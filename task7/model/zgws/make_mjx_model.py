#!/usr/bin/env python3
"""從官方 `zgws.xml` 產生 MJX 可訓練的模型 `zgws_mjx.xml`。

## 為什麼需要這支

官方 MJCF 的碰撞幾何是網格，實測：

| 碰撞 geom | 頂點數 |
|---|---|
| `BASE_LINK` | **98,569** |
| 四個 `*_FOOT_LINK`（輪） | 各 **31,730** |

MJX 的 plane–convex 碰撞是**逐頂點**計算的。2048 個平行環境 × 98,569 頂點，
單一中間張量就是數百 MB —— Colab GPU 上不是慢到不能用就是 OOM。
`mjx.put_model` 本身雖然過得去（實測 2.6 s），MuJoCo 自己就會警告
`coplanar face with more than 20 vertices ... may lead to performance issues`。

**現有 CPG 模擬跑得動不能當作 MJX 可行的證據** —— 那是 CPU MuJoCo、單一環境。

## 改了哪三件事

1. **碰撞幾何**：四輪網格 → 圓柱（半徑取實算的 `WHEEL_RADIUS`，半寬與中心由
   編譯後的 `geom_aabb` 實算，不手填）；`BASE_LINK` 網格 → 方塊（同樣由 aabb 實算）。
   已確認 MJX 的碰撞函式表支援 `(PLANE, CYLINDER)`。
2. **致動器**：16 個 `<motor>` → 12 個 `<position kp kv>` ＋ 4 個 `<velocity kv>`。
   `<position>` 每個**物理步**（500 Hz）算一次 PD，剛好等於原廠 `controller_dt`；
   若改在控制步（50 Hz）算，kp=120 會是完全不同的系統。
3. **拿掉所有網格**：視覺網格（`contype=0 conaffinity=0 density=0`）一併移除，
   連 `<asset><mesh>` 也不留。這樣 XML 自己就是完整模型，**Colab clone 完即可用**，
   不必為了 54 MB 的 STL 去抓 2.1 GB 的官方發布包。

## 不改的事

質量、慣量、關節限位、`frictionloss`、`armature`、site、sensor 全部原樣。
每個 body 都有明寫的 `<inertial>`，所以換 geom 不影響質量分佈 ——
這一點由 `test_mass_and_inertia_unchanged` 釘死，不靠推論。

## 用法

    conda run --no-capture-output -n rbtdog python task7/model/zgws/make_mjx_model.py

產生物**進版控**（沒有網格相依，檔案小且可重現，有 `test_generator_is_reproducible`
釘住「重跑會得到逐字元相同的檔案」）。
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "zgws.xml"
DST = HERE / "zgws_mjx.xml"
SCENE_SRC = HERE / "scene_flat.xml"
SCENE_DST = HERE / "scene_flat_mjx.xml"

sys.path.insert(0, str(HERE.parents[1] / "inference"))
from max_model import (KD3, KD_WHEEL, KP3, LEGS, PREFIX,  # noqa: E402
                       TAU_MAX3, TAU_MAX_WHEEL, WHEEL_RADIUS)

# MJX 沒有提早收斂，solver 迭代數是**固定成本**。MuJoCo 預設 100/50 在 MJX 上是災難。
# 已實測對 walk 的影響可忽略（travel 0.1471→0.1467、bounce 16.7→17.1、support 3.204→3.206）。
SOLVER_ITERATIONS, SOLVER_LS_ITERATIONS = 6, 6

# 圓柱預設軸是 +z，輪關節軸是 +y。繞 x 轉 90° 把 z 帶到 −y（圓柱對稱，正負不影響）。
_Q_Z_TO_Y = "0.7071068 0.7071068 0 0"

JOINT3 = ("ABAD", "HIP", "KNEE")


def _bbox_in_body_frame(m: mujoco.MjModel, mesh_name: str):
    """回傳網格在 **body（＝geom）座標系**的 (中心, 半邊長)。

    ⚠️ **不要用 `m.geom_aabb`。** MuJoCo 編譯時會把網格頂點重新表示在該網格的
    **主慣量軸**座標系，並把補償變換存進 `mesh_pos` / `mesh_quat`。
    `geom_aabb` 是在那個旋轉過的框裡量的，軸會被置換 ——
    實測 `BASE_LINK` 的 aabb 半邊長是 `[0.104, 0.118, 0.417]`，
    看起來像一台「高 0.83 m、長 0.21 m」的狗；轉回 body 框才是
    `[0.417, 0.115, 0.104]`（長 0.83 m）。

    輪子更凶險：aabb 說輪軸在 x，但關節軸是 y。照 aabb 建圓柱會做出一個
    **躺倒 90° 的輪子** —— 它照樣跑得動、照樣有接觸力，只是接觸的是輪緣側面。
    四個診斷指標（超限／飽和／IK 縮限／相位鎖定）全都會是乾淨的。

    所以這裡自己把頂點轉回去：`p_body = mesh_pos + R(mesh_quat) · p_mesh`。
    """
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    assert mid >= 0, f"找不到網格 {mesh_name}"
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.mesh_quat[mid])
    w = v @ rot.reshape(3, 3).T + m.mesh_pos[mid]
    lo, hi = w.min(0), w.max(0)
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def _fmt(v) -> str:
    """XML 數值格式。固定 6 位小數 —— 產生物要進版控，格式必須可重現。"""
    return " ".join(f"{float(x):.6f}" for x in np.atleast_1d(v))


def build(src: str = str(SRC), dst: str = str(DST)) -> dict:
    """讀官方 XML，寫出 MJX 版，回傳替換摘要。"""
    m = mujoco.MjModel.from_xml_path(src)     # 只用來查 aabb，不改它
    tree = ET.parse(src)
    root = tree.getroot()
    summary = {"wheels": [], "base": None, "removed_mesh_geoms": 0}

    # --- 1. 拿掉整個 <asset>（裡面只有 mesh） ---
    for asset in root.findall("asset"):
        root.remove(asset)

    # --- 2. 走訪所有 body，替換 mesh geom ---
    def walk(parent):
        for body in parent.findall("body"):
            for g in list(body.findall("geom")):
                if g.get("type") != "mesh":
                    continue
                mesh = g.get("mesh")
                collidable = g.get("contype") != "0"
                body.remove(g)
                summary["removed_mesh_geoms"] += 1
                if not collidable:
                    continue                       # 視覺網格：拿掉就好
                c, h = _bbox_in_body_frame(m, mesh)
                new = ET.SubElement(body, "geom")
                new.set("name", f"{mesh}_COLL")
                if mesh.endswith("_FOOT_LINK"):
                    # 輪：圓柱，軸沿 +y（＝輪關節軸）。
                    # 半徑取 x/z 兩個半邊長的平均 —— 用**量到的**而不是 WHEEL_RADIUS，
                    # 是為了讓圓柱盡量貼合原網格，G1 對照才能把「換形狀」這個變因隔乾淨。
                    # 但仍與 WHEEL_RADIUS 交叉比對：差太多代表網格被換過或讀錯了。
                    radius = float((h[0] + h[2]) / 2.0)
                    assert abs(radius - WHEEL_RADIUS) < 5e-4, (
                        f"{mesh} 量到的半徑 {radius:.5f} 與 max_model.WHEEL_RADIUS "
                        f"{WHEEL_RADIUS} 差超過 0.5 mm —— 網格換了還是讀錯框？")
                    assert abs(h[0] - h[2]) < 1e-4, (
                        f"{mesh} 的 x/z 半邊長不相等（{h[0]:.5f} / {h[2]:.5f}），"
                        "這不是一個以 y 為軸的圓盤 —— 座標框八成弄錯了")
                    new.set("type", "cylinder")
                    new.set("size", _fmt([radius, h[1]]))
                    new.set("quat", _Q_Z_TO_Y)
                    summary["wheels"].append(
                        {"mesh": mesh, "radius": radius, "half_width": float(h[1]),
                         "pos": c.tolist(), "bbox_half": h.tolist()})
                else:
                    new.set("type", "box")         # 機身
                    new.set("size", _fmt(h))
                    summary["base"] = {"mesh": mesh, "half": h.tolist(), "pos": c.tolist()}
                new.set("pos", _fmt(c))
                new.set("rgba", "1 1 1 1")
            walk(body)

    for wb in root.findall("worldbody"):
        walk(wb)

    # --- 2b. 關掉機器人的自碰撞 ---
    # ⚠️ 這是**被 MJX 逼出來的第四項改動**，不是順手做的最佳化。
    #    MJX 的碰撞函式表沒有 (CYLINDER, BOX)，而輪圓柱與腿上的 hip/knee 方塊
    #    會被列成候選配對 → `mjx.put_model` 直接拋
    #    `NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_BOX) collisions not implemented`。
    #
    #    做法是標準的：機器人所有碰撞體 `contype=1 conaffinity=0`，地板維持 (1,1)。
    #    配對條件 `(contype1 & conaffinity2) | (contype2 & conaffinity1)` 於是
    #    對「機器人 vs 機器人」恆為 0、對「機器人 vs 地板」為 1。
    #
    #    代價：原始模型允許的自碰撞（例如機身撞到自己的膝）在訓練模型裡不會發生。
    #    G1 對照會驗「原始模型走這個步態時本來就沒有自碰撞」，若成立則此項無影響。
    for wb in root.findall("worldbody"):
        for g in wb.iter("geom"):
            if g.get("contype") == "0":          # 視覺 geom 已在上面移除，這裡保險
                continue
            g.set("contype", "1")
            g.set("conaffinity", "0")

    # --- 3. 換致動器 ---
    for a in root.findall("actuator"):
        root.remove(a)
    act = ET.SubElement(root, "actuator")
    for leg in LEGS:
        p = PREFIX[leg]
        for j, (kp, kd, tau) in enumerate(zip(KP3, KD3, TAU_MAX3)):
            e = ET.SubElement(act, "position")
            e.set("name", f"{p}_{JOINT3[j]}_LINK")
            e.set("joint", f"{p}_{JOINT3[j]}_JOINT")
            e.set("kp", f"{kp:.1f}")
            e.set("kv", f"{kd:.1f}")
            e.set("forcerange", f"{-tau:.1f} {tau:.1f}")
        e = ET.SubElement(act, "velocity")
        e.set("name", f"{p}_FOOT_LINK")
        e.set("joint", f"{p}_FOOT_JOINT")
        e.set("kv", f"{KD_WHEEL:.1f}")
        e.set("forcerange", f"{-TAU_MAX_WHEEL:.1f} {TAU_MAX_WHEEL:.1f}")

    # --- 4. solver 選項 ---
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(0, opt)
    opt.set("iterations", str(SOLVER_ITERATIONS))
    opt.set("ls_iterations", str(SOLVER_LS_ITERATIONS))

    # --- 5. 寫檔 ---
    root.set("model", "zgws_mjx")
    ET.indent(tree, space="  ")
    header = ("<!--\n"
              "  ⚠️ 這個檔案是產生的，不要手改。\n"
              "  產生器：task7/model/zgws/make_mjx_model.py（跑它會逐字元覆蓋本檔）\n"
              "  來源：  task7/model/zgws/zgws.xml（官方 MATRiX v0.1.2 原檔）\n"
              "  用途：  MJX / Colab 訓練。碰撞網格換成原始形狀、力矩致動器換成位置伺服、\n"
              "          完全不依賴 STL —— clone 完就能用，不必抓 2.1 GB 發布包。\n"
              "  ⚠️ 它與 zgws.xml **不是同一個物理模型**。差異的量化對照見\n"
              "     task7/docs/MJX模型對照_2026-08-27.md，引用數字要標明是哪一個。\n"
              "-->\n")
    Path(dst).write_text(header + ET.tostring(root, encoding="unicode") + "\n")
    return summary


def build_scene(dst: str = str(SCENE_DST)) -> None:
    """把 scene_flat.xml 的 include 換成 MJX 版，其餘原樣。"""
    text = SCENE_SRC.read_text()
    assert '<include file="zgws.xml"/>' in text, "scene_flat.xml 的 include 行變了"
    text = text.replace('<include file="zgws.xml"/>', '<include file="zgws_mjx.xml"/>')
    text = text.replace("zgws flat scene", "zgws flat scene (mjx)")
    Path(dst).write_text(
        "<!-- ⚠️ 產生檔，不要手改。產生器：make_mjx_model.py -->\n" + text)


if __name__ == "__main__":
    s = build()
    build_scene()
    print(f"[產生] {DST}")
    print(f"  移除網格 geom {s['removed_mesh_geoms']} 個")
    for w in s["wheels"]:
        print(f"  輪 {w['mesh']:>16}  半徑 {w['radius']:.4f}（WHEEL_RADIUS {WHEEL_RADIUS}）"
              f"  半寬 {w['half_width']:.4f} m  中心 y={w['pos'][1]:+.4f}")
    b = s["base"]
    print(f"  機身 {b['mesh']}  半邊長 {[round(v, 4) for v in b['half']]} m")
    print(f"[產生] {SCENE_DST}")
