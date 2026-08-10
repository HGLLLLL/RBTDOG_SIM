# D1 EDU（ZSL-1）模型資產來源

## 網格與運動學

- 來源：https://github.com/zsibot/genisom_model  目錄 `zsl-1/`
- commit：`e6aa98e22d38ae3fdf4d448f79820295e78e83a5`
- 授權：BSD-3-Clause（見同目錄 `LICENSE`）
- 取用內容：17 個 STL 視覺網格；質量/慣量/關節參數抄自同目錄 `urdf/ZSL-1.urdf`

## 為什麼不是官方 repo

智元官方 SDK repo（https://github.com/AgibotTech/agibot_D1_Edu-Ultra , commit `db8accd`）
**不含任何 URDF / mesh / 模擬模型**，只有 `demo/`、`docs/`、`include/`、`lib/`、`site/`。

## 來源一致性佐證

| 項目 | zsibot URDF | 官網 D1 Edu 規格 |
|---|---|---|
| 總質量 | 15.19 kg | 15.5 kg（含電池）|
| abad 行程 | ±28.0° | ±28° |
| hip 行程 | −66.0°~170.0° | −170°~66°（符號約定相反）|
| knee 行程 | −156.0°~−34.5° | 35°~156° |
| 機身長 | 0.5555 m | 站立 0.635 m（含腿）|
| 機型代號 | `ZSL-1` | SDK 點足版代號 `zsl-1` |

六項全對，判定為同一台機器。
