# 報告用論文三線表（表 1 可用範圍、表 2 純 RL vs v3.6f）→ outputs/report_tables/*.svg|png（SVG 文字可編輯）。改數字改這裡重跑。
# 論文三線表（booktabs）風格的可編輯 SVG ＋ PNG
import unicodedata, html, subprocess, os, re
OUT = "/home/huang/rbtdog_sim/task7/outputs/report_tables"
FONT = "Liberation Serif, Times New Roman, Noto Serif CJK TC, Noto Serif CJK HK, Noto Serif CJK JP, serif"

def cw(ch, fs):
    return fs * (1.0 if unicodedata.east_asian_width(ch) in "WF" else 0.5)

def tokens(para):
    return re.findall(r'[⺀-鿿豈-﫿＀-￯　-〿]|\s+|[^\s⺀-鿿豈-﫿＀-￯　-〿]+', para)

def wrap(text, width, fs):
    lines = []
    for para in text.split("\n"):
        cur, w = "", 0.0
        for tok in tokens(para):
            tw = sum(cw(ch, fs) for ch in tok)
            if w + tw > width and cur and tok not in "）、，。；：」』）":
                lines.append(cur.rstrip()); cur, w = "", 0.0
                if tok.isspace(): continue
            cur += tok; w += tw
        lines.append(cur.rstrip())
    return lines

def esc(t): return html.escape(t)

def booktabs(fname, caption, cols, widths, groups, note="", align=None):
    """groups: list of list-of-rows；群組之間畫細線。cell 以 '\n' 換行。align: 每欄 'l'/'c'。"""
    fs, lh, pad, M = 19, 26, 9, 48
    W = sum(widths) + 2 * M
    align = align or ["l"] * len(cols)
    parts = []
    y = 40
    # caption（論文：表號粗體＋標題，置上）
    cap_lines = wrap(caption, W - 2 * M, 20)
    for li, line in enumerate(cap_lines):
        parts.append(f'<text x="{W/2}" y="{y + li*28}" text-anchor="middle" font-size="20" fill="#000">{esc(line)}</text>')
    y += 28 * len(cap_lines) + 6
    top = y
    parts.append(f'<line x1="{M}" y1="{y}" x2="{W-M}" y2="{y}" stroke="#000" stroke-width="1.6"/>')   # toprule
    y += pad
    x = M
    hl = [wrap(c, w - 2*pad, fs) for c, w in zip(cols, widths)]
    n = max(len(l) for l in hl)
    for ci, (lines, w) in enumerate(zip(hl, widths)):
        for li, line in enumerate(lines):
            ax = x + pad if align[ci] == "l" else x + w/2
            anc = "start" if align[ci] == "l" else "middle"
            parts.append(f'<text x="{ax}" y="{y + lh*(li+0.78)}" text-anchor="{anc}" font-size="{fs}" font-weight="700" fill="#000">{esc(line)}</text>')
        x += w
    y += n * lh + pad
    parts.append(f'<line x1="{M}" y1="{y}" x2="{W-M}" y2="{y}" stroke="#000" stroke-width="0.9"/>')   # midrule
    for gi, rows in enumerate(groups):
        if gi > 0:
            parts.append(f'<line x1="{M}" y1="{y}" x2="{W-M}" y2="{y}" stroke="#000" stroke-width="0.5"/>')
        y += pad * 0.6
        def draw_cells(cells, ci0, y0):
            x = M + sum(widths[:ci0])
            n = 0
            for k, cell in enumerate(cells):
                ci = ci0 + k; w = widths[ci]
                lines = wrap(cell, w - 2*pad, fs); n = max(n, len(lines))
                for li, line in enumerate(lines):
                    ax = x + pad if align[ci] == "l" else x + w/2
                    anc = "start" if align[ci] == "l" else "middle"
                    fw = ' font-weight="700"' if ci == 0 else ""
                    parts.append(f'<text x="{ax}" y="{y0 + lh*(li+0.78)}" text-anchor="{anc}" font-size="{fs}"{fw} fill="#000">{esc(line)}</text>')
                x += w
            return n * lh
        for row in rows:
            if isinstance(row, dict):
                hl_ = draw_cells(row["left"], 0, y)
                yr = y
                for sub in row["right"]:
                    yr += draw_cells(sub, len(row["left"]), yr) + 4
                y = max(y + hl_ + 4, yr)
            else:
                y += draw_cells(row, 0, y) + 4
        y += pad * 0.6
    parts.append(f'<line x1="{M}" y1="{y}" x2="{W-M}" y2="{y}" stroke="#000" stroke-width="1.6"/>')   # bottomrule
    if note:
        for line in wrap(note, W - 2*M, 16):
            y += 24
            parts.append(f'<text x="{M}" y="{y}" font-size="16" fill="#000">{esc(line)}</text>')
    H = y + 40
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">'
           f'<rect width="{W}" height="{H}" fill="#fff"/>' + "".join(parts) + '</svg>')
    p = os.path.join(OUT, fname + ".svg"); open(p, "w", encoding="utf-8").write(svg)
    subprocess.run(["rsvg-convert", "-z", "1.6", "-o", os.path.join(OUT, fname + ".png"), p], check=True)
    print(p, W, H)

# ── 表 1：可用範圍 ──
booktabs("表1_純RL步態可用範圍",
    "表 1　純 RL 步態（rl16）各動作可用指令範圍與實機驗證（trip24，2026-09-30）。訓練範圍 ±1.0；模擬掃描 6 s（MuJoCo，含實機摩擦），取指令後 1 s 起平均。",
    ["動作", "模擬可用範圍", "模擬效果（範圍內）", "實機驗證點", "實機效果", "與模擬對照"],
    [110, 220, 330, 130, 380, 330],
    [
     [["直走 前進", "+0.1 ～ +1.0 m/s", "純輪行；達成率 105–117%", "+0.3", "0.20 m/s（67%），輪 2.0–2.2 rad/s，腿不動", "慢 30%：實機輪無摩擦補償（effort 恆 0）"],
      ["直走 後退", "−0.1 ～ −1.0 m/s", "105–113%", "—", "未測", "—"]],
     [{"left": ["左平移", "+0.35 ～ +0.8 m/s\n（≤ 0.3 不動；1.0 超命令速護欄）", "踏步 1.4–2.0 步/s，抬 10–33 mm；達成率 38%（0.35）→ 90%（0.8）"],
       "right": [["+0.3", "不動", "死區與模擬一致"],
                 ["+0.5", "踏步 1.1–1.2 步/s，抬 12–29 mm，roll 峰 2.4°，自轉偏移 −4°/s", "有踏步，與模擬一致。實機 vs 模擬：步頻 1.1–1.2 vs 1.8 步/s；抬腳 12–29 vs 18 mm；髖外展（ABAD）追蹤誤差 0.40 vs 0.30 rad"]]},
      ["右平移", "−0.35 ～ −0.8 m/s", "1.8–2.0 步/s，抬 13–31 mm；47%（0.35）→ 82%（0.8）", "−0.5", "踏步 1.3–1.7 步/s，抬 20–31 mm，roll 峰 3.0°，自轉偏移 +2°/s", "實機 vs 模擬：步頻 1.3–1.7 vs 1.8 步/s；抬腳 20–31 vs 20 mm；髖外展（ABAD）追蹤誤差 0.43 vs 0.29 rad"]],
     [{"left": ["左旋轉", "+0.4 ～ +1.0 rad/s", "達成率 68%（0.4）、76%（0.5）、94–98%（0.8–1.0）"],
       "right": [["+0.5", "+21.4°/s（75%）", "一致（模擬 76%）"],
                 ["+1.0", "+52.9°/s（92%），9 s 穩定，roll ≤ 0.7°", "一致（模擬 98%）"]]},
      {"left": ["右旋轉", "−0.5 ～ −1.0 rad/s\n（−0.4 僅 39%）", "85%（0.5）、86–99%（0.8–1.0）"],
       "right": [["−0.5", "3.5 s 後塌：右傾 4.7°，ABAD 至 0.523 止點，平均 −7.8°/s（27%）", "模擬邊際：側傾 1° 或摩擦 ×1.5 即塌"],
                 ["−1.0", "−55.4°/s（97%），9 s 穩定，roll 2.3–2.7°", "一致（模擬 99%）"]]}],
    ],
    note="達成率＝實測／指令。實機平移無里程計，以步頻與足端離地（FK）描述。護欄用量全程：ABAD 誤差峰 0.48（上限 0.77）、力矩峰 32 N·m（上限 50/70）、推論中位 1.0 ms。")

# ── 表 2：對照 ──
booktabs("表2_純RL_vs_CPG-RL_v3.6f_對照",
    "表 2　純 RL（rl16）與 CPG-RL v3.6f（mirror）實機效果對照。純 RL：trip24；v3.6f：trip22／23，右向動作以鏡像執行。",
    ["動作", "指標", "純 RL（rl16）", "CPG-RL v3.6f", "較優"],
    [110, 230, 410, 410, 110],
    [
     [["直走", "實測速度（指令 0.3）", "0.20 m/s（67%）", "0.29 m/s（97%）", "v3.6f"],
      ["", "方式", "純輪行，腿不動", "純輪行", "—"]],
     [["左平移", "可用指令", "≥ 0.35（實測 0.5）", "≤ 0.15（0.2 超護欄）", "RL"],
      ["", "步態", "踏步 1.1–1.2 步/s，抬 12–29 mm", "側滑，0.3–1.0 步/s，抬 9–26 mm", "RL"],
      ["", "ABAD 力矩／誤差", "24 N·m／0.40", "41 N·m／0.70（¼ 時間釘止點）", "RL"]],
     [["右平移", "步態", "踏步 1.3–1.7 步/s，抬 20–31 mm", "側滑，抬 12–23 mm", "RL"],
      ["", "ABAD 力矩／誤差", "25 N·m／0.43", "40 N·m／0.68", "RL"]],
     [["左旋轉", "最大偏航率", "+53°/s（指令 1.0，92%）", "+79°/s（指令 1.3，106%）", "v3.6f"],
      ["", "低速指令", "0.5：75%", "0.8：90–94%（抬腳 15–27 mm，近拖地）", "v3.6f"]],
     [["右旋轉", "最大偏航率", "−55°/s（指令 −1.0，97%）", "−79°/s（指令 −1.3，107%）", "v3.6f"],
      ["", "低速指令", "−0.5：塌（ABAD 至止點）", "−0.8：−42～−46°/s", "v3.6f"],
      ["", "左右對稱", "獨立訓練；右轉餘裕較小", "鏡像，對稱", "v3.6f"]],
     [["整體", "力矩峰", "32 N·m", "54 N·m（膝）", "RL"],
      ["", "推論時間（中位）", "1.0 ms", "2.0–2.3 ms", "RL"],
      ["", "站姿高度", "0.49 m", "0.54 m", "v3.6f"],
      ["", "摔倒／中止", "0／0（12 趟）", "0／0（16 趟）", "—"]],
    ],
    note="可展示範圍：純 RL vx 0.3、vy ±0.5、wz ±1.0；v3.6f vx 0.3、vy ±0.15、wz ±0.8／±1.3。",
    align=["l", "l", "l", "l", "c"])
