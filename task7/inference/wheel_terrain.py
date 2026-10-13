"""純 RL 自訓 v1 的地形與場景（spec docs/superpowers/specs/2026-10-13-rl-wheel-v1-design.md §4）。

大圖：三列（rough／boxes／stairs）× 十欄（難度 0–9）的 8 m × 8 m 方塊，格距 0.1 m，外圍 20 m 平地（他的 border_width 200；2 m 時離邊只 6 m、開出圖外會自由落體）；
全圖基準高 BASE_H（反向樓梯往下挖，hfield 資料仍 ≥ 0）。
場景：`scene_flat_mjx_v3f.xml` 去平面、加 hfield、**全部碰撞幾何改膠囊**（輪子／機身／腿；MJX 3.10 不支援 hfield–cylinder，
hfield 對凸體要對 subgrid 每個稜柱跑 GJK、對膠囊是閉式解；A100 實測網格輪＋方塊機身 1024 envs 只 5.4k steps/s 且不隨環境數增加）、
輪速伺服 kv 1.0（他的 IdealPD）、四個輪輪 `<pair>`。CPU 驗收工具用 `make_flat_pairs_scene_xml()`（圓柱輪子＋pair）。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model" / "zgws"

DX = 0.1                                   # 格距 m（＝他的 horizontal_scale）
TILE = 8.0                                 # 方塊邊長 m
NT = int(round(TILE / DX))                 # 每方塊 80 格
LEVELS = 10
KINDS = ("rough", "boxes", "stairs")
BORDER = 20.0                              # 大圖外圍平地 m（他的 border_width 200；2 m 時 ~22% 回合開出圖外）
NB = int(round(BORDER / DX))               # 200 格
NCOL = 2 * NB + LEVELS * NT + 1            # 1201（x）
NROW = 2 * NB + len(KINDS) * NT + 1        # 641（y）
X0 = -(NCOL - 1) * DX / 2                  # -60.0
Y0 = -(NROW - 1) * DX / 2                  # -32.0
BASE_H = 1.0                               # 全圖基準高 m
ZMAX = 2.0                                 # hfield 高度量程 m（資料 0..1 ↔ 0..ZMAX）
TILE_BORDER = 0.25                         # 每方塊邊緣壓平寬 m（他的 border_width）
STAIR_TREAD, STAIR_STEPS = 0.3, 10         # 踏面 m、階數（中央平台 ±1 m 到邊 4 m ＝ 3 m / 0.3）
WHEEL_R, WHEEL_HW = 0.096, 0.005
PAIRS = (("FAR", "FBL"), ("RAR", "RBL"), ("FAR", "RAR"), ("FBL", "RBL"))
CYL = 'type="cylinder" size="0.095999 0.005000" quat="0.7071068 0.7071068 0 0"'
WHEEL_CAP = f'type="capsule" size="{WHEEL_R} {WHEEL_HW}" quat="0.7071068 0.7071068 0 0"'     # 軸向 y 同圓柱
BASE_BOX = 'type="box" size="0.416641 0.115123 0.104083"'
BASE_CAP = 'type="capsule" size="0.105 0.31" quat="0.7071068 0 0.7071068 0"'               # 軸 z→x；半徑 ≈ 方塊 y/z 半寬，0.31+0.105 ≈ x 半長
UPPER_RE = re.compile(r'(<geom size=")0\.03 0\.0075 0\.11(" pos="[^"]*" quat="[^"]*") type="box"')        # 大腿方塊長軸 z ＝ 膠囊軸
LOWER_RE = re.compile(r'(<geom size=")0\.015 0\.015 0\.11(" pos="0\.005 0 -0\.09" quat="0\.99995 0 -0\.00999983 0") type="box"')
ROBOT_SRC, ROBOT_OUT = MODEL_DIR / "zgws_mjx_v3f.xml", MODEL_DIR / "zgws_mjx_wheel.xml"
SCENE_SRC, SCENE_OUT = MODEL_DIR / "scene_flat_mjx_v3f.xml", MODEL_DIR / "scene_wheel_mjx.xml"
FLAT_PAIRS_OUT = MODEL_DIR / "scene_flat_pairs_v3f.xml"


def level_amp(kind: str, level: int) -> float:
    """難度 0–9 → 幅度：rough 0.01–0.06（他的 noise_range）、boxes 0.025–0.10（grid_height_range）、stairs 階高 0.03–0.08。"""
    u = level / (LEVELS - 1)
    lo, hi = {"rough": (0.01, 0.06), "boxes": (0.025, 0.10), "stairs": (0.03, 0.08)}[kind]
    return lo + u * (hi - lo)


def tile_pattern(kind: str, level: int, seed: int) -> np.ndarray:
    """(NT+1)×(NT+1) 相對高度 m（row ↔ y、col ↔ x），方塊中心 (0,0)，邊緣 TILE_BORDER 壓平到 0。"""
    rng = np.random.default_rng(seed)
    n = NT + 1
    xs = (np.arange(n) - NT / 2) * DX
    X, Y = np.meshgrid(xs, xs)
    if kind == "flat":
        h = np.zeros((n, n))
    else:
        amp = level_amp(kind, level)
        if kind == "rough":
            h = np.round(rng.uniform(0.0, amp, (n, n)) / 0.01) * 0.01
        elif kind == "boxes":
            cell = 0.45
            ci = np.floor((X + TILE / 2) / cell).astype(int); cj = np.floor((Y + TILE / 2) / cell).astype(int)
            nc = int(np.ceil(TILE / cell)) + 1
            hc = rng.uniform(-amp, amp, (nc, nc))
            h = hc[cj, ci]
            h[(np.abs(X) < 1.0) & (np.abs(Y) < 1.0)] = 0.0
        elif kind == "stairs":
            ring = np.ceil(np.maximum(np.maximum(np.abs(X), np.abs(Y)) - 1.0, 0.0) / STAIR_TREAD - 1e-9)   # 0 ＝ 中央平台
            h = amp * np.clip(STAIR_STEPS - ring, 0, STAIR_STEPS)
            if rng.integers(2) == 1:
                h = -h                                                                                    # 反向：中央是坑、邊緣 0
        else:
            raise ValueError(kind)
    edge = (np.abs(X) > TILE / 2 - TILE_BORDER) | (np.abs(Y) > TILE / 2 - TILE_BORDER)
    h[edge] = 0.0
    return h


def tile_origin(kind_idx: int, level: int):
    """方塊中心世界座標 (x, y)。"""
    return (X0 + (NB + NT / 2 + level * NT) * DX, Y0 + (NB + NT / 2 + kind_idx * NT) * DX)


def build_map(seed: int = 0, flat: bool = False) -> np.ndarray:
    """H[NROW, NCOL] 絕對高度 m（含 BASE_H）。方塊 seed = seed*1000 + kind_idx*100 + level。flat=True 全圖 BASE_H（對齊驗證用）。"""
    H = np.full((NROW, NCOL), BASE_H)
    if flat:
        return H
    for ki, kind in enumerate(KINDS):
        for lv in range(LEVELS):
            i0, j0 = NB + ki * NT, NB + lv * NT
            H[i0:i0 + NT + 1, j0:j0 + NT + 1] += tile_pattern(kind, lv, seed * 1000 + ki * 100 + lv)
    return H


def height_np(H: np.ndarray, x: float, y: float) -> float:
    """雙線性查高（MuJoCo hfield 三角化是線性的，差別只在格內對角線）。"""
    fx = (x - X0) / DX; fy = (y - Y0) / DX
    j = int(np.clip(np.floor(fx), 0, NCOL - 2)); i = int(np.clip(np.floor(fy), 0, NROW - 2))
    tx = float(np.clip(fx - j, 0, 1)); ty = float(np.clip(fy - i, 0, 1))
    return float((1 - tx) * (1 - ty) * H[i, j] + tx * (1 - ty) * H[i, j + 1] + (1 - tx) * ty * H[i + 1, j] + tx * ty * H[i + 1, j + 1])


def load_hfield(m, H: np.ndarray) -> None:
    assert m.nhfield == 1 and m.hfield_nrow[0] == H.shape[0] and m.hfield_ncol[0] == H.shape[1]
    assert H.min() >= 0.0 and H.max() <= ZMAX
    m.hfield_data[:] = (H / ZMAX).ravel()


def pairs_xml() -> str:
    return "  <contact>\n" + "".join(f'    <pair geom1="{a}_FOOT_LINK_COLL" geom2="{b}_FOOT_LINK_COLL"/>\n' for a, b in PAIRS) + "  </contact>\n"


def add_wheel_pairs(xml: str) -> str:
    assert "<contact>" not in xml
    return xml.replace("</mujoco>", pairs_xml() + "</mujoco>")


def _write(path: Path, text: str) -> str:
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return str(path)


def make_robot_xml() -> str:
    """zgws_mjx_v3f.xml → 全部碰撞幾何改膠囊（輪／機身／大腿／小腿）、輪速伺服 kv 0.1 → 1.0。"""
    src = ROBOT_SRC.read_text(encoding="utf-8")
    assert src.count(CYL) == 4 and src.count('kv="0.1"') == 4, (src.count(CYL), src.count('kv="0.1"'))
    assert src.count(BASE_BOX) == 1, src.count(BASE_BOX)
    out = src.replace(CYL, WHEEL_CAP).replace('kv="0.1"', 'kv="1.0"').replace(BASE_BOX, BASE_CAP)
    out, n_up = UPPER_RE.subn(r'\g<1>0.03 0.11\g<2> type="capsule"', out)
    out, n_lo = LOWER_RE.subn(r'\g<1>0.015 0.11\g<2> type="capsule"', out)
    assert n_up == 4 and n_lo == 4, (n_up, n_lo)
    assert 'type="box"' not in out and 'type="cylinder"' not in out
    out = ("<!-- ⚠️ 產生檔，不要手改。產生器：inference/wheel_terrain.py（zgws_mjx_v3f.xml 的膠囊版：全部碰撞幾何改膠囊——"
           "MJX hfield 對膠囊是閉式解，對凸體要跑 GJK；A100 實測 1024 envs 只有 5.4k steps/s 且不隨環境數增加；輪 kv 1.0＝他的 IdealPD） -->\n" + out)
    return _write(ROBOT_OUT, out)


def make_scene_xml() -> str:
    """訓練場景：hfield 大圖 ＋ 膠囊碰撞機器人 ＋ 輪輪 pair。hfield 資料在載入後用 load_hfield 填。"""
    make_robot_xml()
    src = SCENE_SRC.read_text(encoding="utf-8")
    hf = f'    <hfield name="terrain" nrow="{NROW}" ncol="{NCOL}" size="{(NCOL - 1) * DX / 2} {(NROW - 1) * DX / 2} {ZMAX} 0.5"/>\n'
    assert "<asset>\n" in src and 'name="floor"' in src and '<include file="zgws_mjx_v3f.xml"/>' in src
    src = src.replace("<asset>\n", "<asset>\n" + hf, 1)
    assert src.count("<statistic ") == 1 and "timestep" not in src
    src = src.replace("<statistic ", '<option timestep="0.005"/>\n  <statistic ', 1)    # 控制 0.02 s ＝ 4 × 0.005（MJCF 的 <option> 可重複、屬性合併）
    src = re.sub(r'<geom name="floor"[^>]*/>', '<geom name="floor" type="hfield" hfield="terrain" pos="0 0 0" material="groundplane"/>', src)
    src = src.replace('<include file="zgws_mjx_v3f.xml"/>', '<include file="zgws_mjx_wheel.xml"/>')
    src = src.replace("<!-- ⚠️ 產生檔，不要手改。產生器：make_mjx_model.py -->",
                      "<!-- ⚠️ 產生檔，不要手改。產生器：inference/wheel_terrain.py（純 RL 自訓 v1 訓練場景） -->", 1)
    return _write(SCENE_OUT, add_wheel_pairs(src))


def make_flat_pairs_scene_xml() -> str:
    """CPU 驗收工具 `--self-collision`：平地 v3f（圓柱輪）＋輪輪 pair。"""
    src = SCENE_SRC.read_text(encoding="utf-8")
    src = src.replace("<!-- ⚠️ 產生檔，不要手改。產生器：make_mjx_model.py -->",
                      "<!-- ⚠️ 產生檔，不要手改。產生器：inference/wheel_terrain.py（scene_flat_mjx_v3f.xml ＋ 輪輪 pair） -->", 1)
    return _write(FLAT_PAIRS_OUT, add_wheel_pairs(src))


if __name__ == "__main__":
    print(make_scene_xml()); print(make_flat_pairs_scene_xml())
