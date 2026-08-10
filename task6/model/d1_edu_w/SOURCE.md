# D1 EDU 輪足版（ZSL-1w）模型資產來源

## 網格與運動學

- 來源：https://github.com/zsibot/genisom_model  目錄 **`zsl-1w/`**（輪足版）
- commit：`e6aa98e22d38ae3fdf4d448f79820295e78e83a5`
- 授權：BSD-3-Clause（見同目錄 `LICENSE`）
- 取用內容：17 個 STL 視覺網格；質量/慣量/關節參數抄自同目錄 `urdf/ZSL-1W.urdf`
- 未取用：上游 `meshes/` 另有 `BASE_LINK_ori.STL`，URDF 未引用，故未複製。

## 為什麼不是官方 repo

智元官方 SDK repo（https://github.com/AgibotTech/agibot_D1_Edu-Ultra , commit `db8accd`）
**不含任何 URDF / mesh / 模擬模型**，只有 `demo/`、`docs/`、`include/`、`lib/`、`site/`。

## 輪足版與點足版的差異（同一產品線的兩種構型）

| 項目 | ZSL-1 點足 | **ZSL-1w 輪足（本專案採用）** |
|---|---|---|
| URDF 自由度 | 12 | 16（每腿多一顆連續旋轉的輪）|
| 總質量 | 15.19 kg | **20.56 kg** |
| 足端 | 0.063 kg 圓球 | **0.901 kg 輪**（半徑 71 mm、寬 48 mm）|
| 小腿質量 | 0.207 kg | 0.765 kg |
| 膝關節軸 | `0 -1 0` | **`0 1 0`** |

## 與官網規格的對帳

官網 D1 Pro/Edu 標示 15.5 kg，本 URDF 為 20.56 kg。差額 5.06 kg
≈ 4 ×[(0.901 − 0.063) + (0.765 − 0.207)] = 5.58 kg，正好是輪子與加粗小腿相對點足版的增重。
判定：**官網數字是點足版**，兩份資料不衝突。

其餘可對帳項目（輪足版與點足版共用）：

| 項目 | zsibot URDF | 官網 D1 Edu 規格 |
|---|---|---|
| abad 行程 | ±28.0° | ±28° |
| hip 行程 | −66.0°~170.0° | −170°~66°（符號約定相反）|
| knee 行程 | −156.0°~−34.5° | 35°~156° |
| 機身長 | 0.5555 m | 站立 0.635 m（含腿）|
| 機型代號 | `ZSL-1W` | SDK 輪足版代號 `zsl-1w` |
