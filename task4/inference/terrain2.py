"""地形 v2：統一 hfield（平台 + 0–15° 斜坡 + 粗糙度漸變凹凸）與雙線性 gz。"""
import numpy as np
import mujoco

# --- 幾何參數（見 spec §3）---
PLATFORM_HALF = 1.0
TERR_X_MAX = 6.0
TERR_WY = 3.0
AMP_MAX = 0.08
_d1 = 1.5 * np.tan(np.radians(5.0))
_d2 = 1.5 * np.tan(np.radians(10.0))
_d3 = 2.0 * np.tan(np.radians(15.0))
KNOTS_X = np.array([-6.0, -4.0, -2.5, -1.0, 1.0, 2.5, 4.0, 6.0], np.float64)
KNOTS_Z = np.array([-(_d1 + _d2 + _d3), -(_d1 + _d2), -_d1, 0.0,
                    0.0, _d1, _d1 + _d2, _d1 + _d2 + _d3], np.float64)


def slope_z(x):
    return np.interp(x, KNOTS_X, KNOTS_Z)


def amp_at(x):
    return AMP_MAX * np.clip((np.abs(x) - PLATFORM_HALF) / 2.0, 0.0, 1.0)


def bump(x, y):
    # 多正弦疊加，正規化到 ~[-1,1]；確定性（幾何靜態）
    s = (np.sin(2.1 * x) * np.cos(1.7 * y)
         + 0.5 * np.sin(3.7 * x + 1.0) * np.cos(2.9 * y + 0.5)
         + 0.3 * np.sin(5.3 * x + 2.0) * np.cos(4.1 * y))
    return s / 1.8


def build_height_grid(ncol=161, nrow=81):
    xs = np.linspace(-TERR_X_MAX, TERR_X_MAX, ncol)
    ys = np.linspace(-TERR_WY, TERR_WY, nrow)
    X, Y = np.meshgrid(xs, ys)                      # (nrow, ncol)
    Hg = slope_z(X) + amp_at(X) * bump(X, Y)
    return xs, ys, Hg


XS, YS, H = build_height_grid()


def gz_from(xp, xs, ys, Hg, x, y):
    """array-agnostic 雙線性內插；xp = numpy 或 jax.numpy。均勻網格→直接算索引。"""
    nx = xs.shape[0]; ny = ys.shape[0]
    fx = (x - xs[0]) / (xs[-1] - xs[0]) * (nx - 1)
    fy = (y - ys[0]) / (ys[-1] - ys[0]) * (ny - 1)
    fx = xp.clip(fx, 0.0, nx - 1 - 1e-6)
    fy = xp.clip(fy, 0.0, ny - 1 - 1e-6)
    ix = xp.floor(fx).astype(xp.int32); iy = xp.floor(fy).astype(xp.int32)
    tx = fx - ix; ty = fy - iy
    h00 = Hg[iy, ix]; h01 = Hg[iy, ix + 1]
    h10 = Hg[iy + 1, ix]; h11 = Hg[iy + 1, ix + 1]
    return (h00 * (1 - tx) * (1 - ty) + h01 * tx * (1 - ty)
            + h10 * (1 - tx) * ty + h11 * tx * ty)


def gz_np(x, y):
    return gz_from(np, XS, YS, H, np.asarray(x, np.float64), np.asarray(y, np.float64))


def build_terrain2_model(scene_path):
    spec = mujoco.MjSpec.from_file(scene_path)
    floor = next(g for g in spec.geoms if g.name == "floor")
    hmin = float(H.min()); hmax = float(H.max())
    data01 = ((H - hmin) / (hmax - hmin)).astype(np.float64)   # [0,1] row-major
    hf = spec.add_hfield()
    hf.name = "terrain2"
    hf.nrow = H.shape[0]; hf.ncol = H.shape[1]
    hf.size = [TERR_X_MAX, TERR_WY, (hmax - hmin), 0.5]
    hf.userdata = data01.flatten().tolist()
    floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
    floor.hfieldname = "terrain2"
    floor.pos = [0.0, 0.0, hmin]                # data=0(最低) 對到世界 z=hmin → 平台(H=0) 落在 z=0
    # 安全底網：加一塊大 plane 在 z=-10
    net = spec.worldbody.add_geom()
    net.name = "safety_net"; net.type = mujoco.mjtGeom.mjGEOM_PLANE
    net.size = [0.0, 0.0, 0.05]; net.pos = [0.0, 0.0, -10.0]
    net.rgba = [0.3, 0.3, 0.3, 0.0]
    return spec.compile()
