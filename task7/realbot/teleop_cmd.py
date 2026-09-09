"""遙控指令 (vx, vy, wz) → 逐腿 CPG 振幅 (mu_x, mu_y)。純標準函式庫，狗上與模擬共用同一份。

原理（cpg.foot_targets）：fx = 2(rx−1)/(MU_MAX−MU_MIN) − 1，mu 1.5 → 0（原地踏步）、1.8 → +0.6（A 基準前進）、
1.2 → −0.6（倒退）；fy 同理給側向步幅。旋轉＝左右腿 mu_x 反號。

    mu_x[leg] = 1.5 + GAIN·clip(vx + TURN_SIGN[leg]·wz, −1, 1)
    mu_y[leg] = 1.5 + GAIN·clip(vy, −1, 1)

vx/vy/wz 是正規化指令 ∈ [−1, 1]（1 = A 基準的 mu 1.8）。正負號慣例由 `inference/cpg_teleop_sim.py`
在模擬驗證（+vx 前進、+vy 左移、+wz 左轉），寫在下面的常數裡，狗上直接沿用。
"""
from __future__ import annotations

NEUTRAL = 1.5
NEUTRAL_X = 1.46              # ★ 原地踏步的 mu_x 配平：x_off −30 mm 是為前進配的，mu_x 1.5 踏步會前爬 0.054 m/s；
                              #   1.46 → 0.012 m/s（模擬 6 s 掃描，cpg_teleop_sim 2026-09-09）
GAIN = 0.3                    # 1.5 ± 0.3 → 1.2..1.8（A 基準 mu_x 1.8）
GAIN_Y = 0.18                 # 側向步幅小一點：0.3 時 roll 峰 7.7°、膝 60、誤差 0.50（模擬）
LEGS = ("fl", "fr", "bl", "br")
# 旋轉 = 兩件事各一半（模擬 2026-09-09：純左右步幅差只有 ±2°/s，因為輪子會跟著滾；
#   前後腿側向反向 ±10°/s）：+wz 左轉（偏航角增加）。若反了改這裡，不要改別處。
TURN_SIGN = {"fl": -1.0, "bl": -1.0, "fr": +1.0, "br": +1.0}      # 左右腿 mu_x 反向（左腿減）
ROT_Y_SIGN = {"fl": +1.0, "fr": +1.0, "bl": -1.0, "br": -1.0}     # 前腿左步、後腿右步
VY_SIGN = +1.0                # +vy 左移；模擬驗證定案
SLEW = 0.015                  # 每個 50 Hz 步 mu 最多變 0.015（滿幅 0.6 要 0.8 s）。
                              # ⚠️ 0.04（0.3 s）時前進直接切倒退會摔（模擬 roll 49°）


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else (hi if x > hi else x)


def mu_targets(vx: float, vy: float, wz: float) -> tuple[dict, dict]:
    """→ (mu_x{leg}, mu_y{leg})。mu_x ∈ [1.16, 1.76]（中性 1.46）、mu_y ∈ [1.32, 1.68]。"""
    mux = {l: NEUTRAL_X + GAIN * _clip(vx + 0.5 * TURN_SIGN[l] * wz) for l in LEGS}
    muy = {l: NEUTRAL + GAIN_Y * _clip(VY_SIGN * vy) + GAIN * 0.5 * ROT_Y_SIGN[l] * _clip(wz)
           for l in LEGS}
    return mux, muy


def neutral() -> tuple[dict, dict]:
    return mu_targets(0.0, 0.0, 0.0)


def slew(cur: dict, tgt: dict, rate: float = SLEW) -> dict:
    """逐腿斜率限制：承重腿不能被瞬間改步幅（同 sway 的 4 mm/步原則）。"""
    out = {}
    for l in LEGS:
        d = tgt[l] - cur[l]
        out[l] = cur[l] + (rate if d > rate else (-rate if d < -rate else d))
    return out


# 鍵盤配置（M9 --teleop 與模擬腳本共用）：按住動、放開停
KEYMAP = {
    "w": (1.0, 0.0, 0.0),     # 前進
    "s": (-1.0, 0.0, 0.0),    # 後退
    "a": (0.0, 1.0, 0.0),     # 左平移
    "d": (0.0, -1.0, 0.0),    # 右平移
    "q": (0.0, 0.0, 1.0),     # 左轉（原地）
    "e": (0.0, 0.0, -1.0),    # 右轉（原地）
}
ESTOP_KEY = " "


class CmdShaper:
    """把鍵盤指令整形成安全的 mu 目標：斜率限制 ＋ **反向要先回中性踏步停 `DWELL` 步**。

    模擬（cpg_teleop_sim 2026-09-09）：前進 4 s 直接切倒退，即使斜率放到 0.8 s 仍 roll 17.6°、膝 95 N·m、
    偏航 +36°（踉蹌）；經過 1 s 中性踏步再換向就乾淨。每個 50 Hz 步呼叫一次 `step()`。
    """

    DWELL = 50        # 反向前在中性停留的步數（1.0 s ≈ 1.4 個步態週期）

    def __init__(self):
        self.mux, self.muy = neutral()
        self.cmd = (0.0, 0.0, 0.0)      # 目前正在執行的指令
        self.pending = None             # 反向時排隊的新指令
        self.dwell = 0

    @staticmethod
    def _reverses(a, b) -> bool:
        return any(x * y < -1e-9 for x, y in zip(a, b))

    def step(self, vx: float, vy: float, wz: float) -> tuple[dict, dict]:
        want = (float(vx), float(vy), float(wz))
        if self._reverses(self.cmd, want):
            self.cmd, self.pending, self.dwell = (0.0, 0.0, 0.0), want, self.DWELL
        elif self.pending is not None:
            if want == (0.0, 0.0, 0.0) or self._reverses(self.pending, want):
                self.pending = None if want == (0.0, 0.0, 0.0) else want
                if self.pending is None:
                    self.dwell = 0
            else:
                self.pending = want
        else:
            self.cmd = want
        if self.pending is not None:
            self.dwell -= 1
            if self.dwell <= 0:
                self.cmd, self.pending = self.pending, None
        tx, ty = mu_targets(*self.cmd)
        self.mux, self.muy = slew(self.mux, tx), slew(self.muy, ty)
        return self.mux, self.muy
