# MJPC 建置問題與解法（Arch + CMake 4 + GCC 16）

> 產出日期：2026-07-01
> 環境：EndeavourOS（Arch 系）· CMake 4.3.2 · GCC 16.1.1 · MuJoCo MPC（`google-deepmind/mujoco_mpc`，內附 MuJoCo 3.2.3）
> 背景：在此機器上編譯 MJPC 時連續踩到 4 個問題。**根源都是「Arch 滾動更新的最前沿工具鏈」比 MJPC 原始碼假設的還新**，觸發舊碼的相容性與未定義行為問題。
> 本文件記錄每個問題的「症狀 / 根因 / 解法」，並在最前面給出一份已驗證可用的完整建置流程。

---

## 0. 已驗證可用的完整建置流程（照這個做就不會踩雷）

```bash
# (1) 安裝建置工具（Arch DB 若過期會 404，用 -Syu 一起裝）
sudo pacman -Syu base-devel cmake git ninja

# (2) 取得原始碼
cd /home/huang/rbtdog_sim
git clone --depth 1 https://github.com/google-deepmind/mujoco_mpc.git
cd mujoco_mpc

# (3) 第一次 configure：把相依抓下來（含雷①的旗標）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG -fno-strict-aliasing" \
      -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -fno-strict-aliasing"

# (4) 修改 build/_deps/mujoco-src/cmake/MujocoOptions.cmake（雷②、雷③輔助）：
#     - 移除 EXTRA_COMPILE_OPTIONS 裡的 "-Werror" 那一行
#     - 把 LTO 那段：
#         if(NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION AND (...))
#           set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
#         endif()
#       改成：
#         set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# (5) 重跑一次 (3) 的 cmake configure（讓改動生效）

# (6) 編譯（一次性，約 15 分鐘）
cmake --build build -j8          # 記憶體不足改 -j4

# (7) 執行（以後每次直接跑這行即可，不用再編）
./build/bin/mjpc
```

> **關鍵旗標速記**：`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`（雷①）、刪 `-Werror`（雷②）、**`-fno-strict-aliasing`（雷③，最關鍵）**。
> LTO 關閉與 `-O2` 屬「保險」非必要，留著無害。

---

## 雷①：CMake 4.x 太新，configure 失敗

**症狀**
```
CMake Error at build/_deps/qhull-src/CMakeLists.txt:70 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

**根因**
CMake 4.x 移除了對「宣告 `cmake_minimum_required(VERSION < 3.5)`」的相容性。MJPC 的相依 **qhull** 仍用舊語法，於是被擋下。

**解法**
configure 時加旗標，告訴 CMake 用 3.5 的政策相容模式：
```bash
cmake ... -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

---

## 雷②：GCC 16 太嚴 + MuJoCo `-Werror`，編譯中斷

**症狀**
```
engine_print.c:158:12: error: assignment discards ‘const’ qualifier from
  pointer target type [-Werror=discarded-qualifiers]
cc1: all warnings being treated as errors
ninja: build stopped: subcommand failed.
```

**根因**
MuJoCo 內附原始碼（`engine_print.c`）有一處寫法在**新版 GCC 16** 下會產生 `discarded-qualifiers` 警告；而 MuJoCo 的 CMake 用 `-Werror` 把所有警告當錯誤，於是編譯中斷。這是「新編譯器更嚴格 vs 舊碼」的典型狀況。

**解法（全域關掉 MuJoCo 的 -Werror，一勞永逸）**
編輯 `build/_deps/mujoco-src/cmake/MujocoOptions.cmake`，把 `EXTRA_COMPILE_OPTIONS` 裡的 `-Werror` 那一行刪掉：
```cmake
  set(EXTRA_COMPILE_OPTIONS
      -Werror        # ← 刪掉這行
      -Wall
      ...
```
改完重跑 `cmake --build build -j8`（ninja 會自動重設定後續編譯）。

> 補充：也可只精準修 `engine_print.c:158` 把 `char* c;` 改成 `const char* c;`，但因 GCC 16 可能還有其他處會中招，直接關 `-Werror` 較省事。

---

## 雷③：GCC 16 + strict-aliasing → 一開 Quadruped 就 segfault（最關鍵）

**症狀**
編譯成功、`mjpc` 也能啟動並印出版本資訊，但**一載入 Quadruped 任務就 segfault**：
```
Thread 12 "mjpc" received signal SIGSEGV, Segmentation fault.
0x... in mjpc::QuadrupedFlat::ResidualFn::FootStep(...)
  at quadruped.cc:676
#1 ... QuadrupedFlat::ResidualFn::Residual(...) at quadruped.cc:113
#2 ... mj_sensorAcc () from libmujoco.so.3.2.3
```

**診斷過程（用 gdb）**
- 崩在 `quadruped.cc:676`：`double footphase = 2*mjPI*kGaitPhase[gait][foot];`
- gdb 印出變數：`current_gait_ = 0`（正確，kGaitStand），但傳進來的 `gait = 1492025856`（垃圾），**且每次執行值都不同**。
- `gait` 來自 `GetGait()` → `static_cast<A1Gait>(ReinterpretAsInt(current_gait_))`。
- `ReinterpretAsInt` 的實作：
  ```cpp
  int ReinterpretAsInt(double value) {
    return *std::launder(reinterpret_cast<const int*>(&value));  // 用 int* 讀 double
  }
  ```

**根因**
這是 **strict-aliasing（嚴格別名）未定義行為**：透過 `int*` 去讀一個 `double` 物件，違反 C/C++ 的型別別名規則。MJPC 用這個「位元雙關（type punning）」把離散選單索引塞進 double 參數。

- 舊版 GCC 不會出事；
- 但 **GCC 16 在 `-O2/-O3`（預設開 `-fstrict-aliasing`）下**，會假設 `int*` 與 `double` 不會指向同一塊記憶體，因而把這段型別雙關**錯誤最佳化**，讀出的是垃圾（`current_gait_=0` 卻讀成 15 億）。
- `gait` 變垃圾 → `kGaitPhase[gait]`（合法範圍只有 0~4）嚴重越界 → 讀到未映射記憶體 → SIGSEGV。

**解法（唯一有效）**
編譯時加 **`-fno-strict-aliasing`**，關閉型別別名假設，讓位元雙關如預期運作。**必須放在 `-O2` 之後**（否則 `-O2` 隱含的 `-fstrict-aliasing` 會蓋過它）：
```bash
cmake ... \
  -DCMAKE_C_FLAGS_RELEASE="-O2 -DNDEBUG -fno-strict-aliasing" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -fno-strict-aliasing"
```

**踩過但無效的嘗試（記錄下來避免重走）**
- ❌ 關閉 LTO（`CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF`）→ 仍崩。
- ❌ `-O3` 降到 `-O2` → 仍崩。
- ✅ 加 `-fno-strict-aliasing` → 解決。
（結論：病根是 strict-aliasing，與 LTO/最佳化等級無關。上面兩項留著無害，但非必要。）

**驗證方式（無需開 GUI，無頭 shell 即可重現與驗證）**
```bash
timeout 25 gdb -batch -ex "set debuginfod enabled off" -ex run \
  --args ./build/bin/mjpc
# 修正前：數秒內 SIGSEGV 於 FootStep
# 修正後：跑滿 25 秒無 SIGSEGV（被 timeout 結束，exit 124）
```

---

## 雷⑤：GUI 執行中，視窗失焦被遮住、切回來就 segfault（KWin + Mesa）

**症狀**
mjpc 開起來能正常跑、機器狗會走；但**切到別的視窗一陣子（原視窗被遮住/失焦），再切回來時整個崩掉關閉**。console 只留 `Segmentation fault (core dumped)`。

**診斷（用 coredumpctl 撈 core dump backtrace）**
崩潰在**主執行緒的渲染路徑**，且進到 Mesa 驅動內部：
```
thread (main):
#0  libgallium-26.1.3 (Mesa Intel GL 驅動內部)
#1  mjr_render          (libmujoco 渲染)
#2  mujoco::Simulate::Render()
#3  mjpc::MjpcApp::Start() → main
```
（注意：規劃執行緒此時是安穩跑 iLQG，**沒有**再崩在 FootStep → 雷③確實已修好。）

**根因**
環境：**KDE Plasma on Wayland（kwin_wayland）＋ Intel Iris Xe ＋ Mesa 26.1.3**，mjpc 走 **XWayland**。KWin 對「被遮住/失焦的 XWayland 視窗」會暫停送畫面或使 GL 緩衝失效，切回來重繪時 Mesa 在 `mjr_render` 崩潰。屬**驅動/合成器堅固性問題，非 MJPC 程式碼的錯**（Mesa 26.1.3 是最前沿版本）。

**解法（本機實測①有效）**
- **① 關閉 Mesa 多執行緒 GL（有效）**：
  ```bash
  MESA_GLTHREAD=false /home/huang/rbtdog_sim/mujoco_mpc/build/bin/mjpc
  ```
- ② 若①無效，用軟體渲染（最穩，較慢）：`LIBGL_ALWAYS_SOFTWARE=1 ./build/bin/mjpc`
- ③ 實務閃避：使用 GUI 時別把視窗整個遮住/最小化。

**重要**：此崩潰只影響**互動 GUI**。任務3 走 MJPC **Python API + EGL 離屏渲染**（不經 KWin 視窗），不受此問題影響。

---

## 雷④（附帶）：Debug build 時 abseil 缺 `<cstdint>`

> 只有在**改用 `-DCMAKE_BUILD_TYPE=Debug`** 時才會遇到；一般 Release/RelWithDebInfo 不會。記錄備查。

**症狀**
```
abseil-cpp-src/absl/container/internal/container_memory.h:66:27:
  error: ‘uintptr_t’ does not name a type [-Wtemplate-body]
```

**根因**
abseil 有一段 debug-only 程式碼（`NDEBUG` 未定義時才編）用到 `uintptr_t` 卻沒 `#include <cstdint>`。新版 GCC 16 header 不再遞移包含 `<cstdint>`，於是報錯。Release / RelWithDebInfo 因定義了 `NDEBUG` 會跳過該段，所以沒事。

**解法**
- 直接**不要用純 Debug build**；要除錯符號改用 `-DCMAKE_BUILD_TYPE=RelWithDebInfo`（`-O2 -g -DNDEBUG`），同樣有符號又能編過。
- 若真的需要 Debug：在該 abseil 檔補 `#include <cstdint>`（可能要補多處）。

---

## 附帶：MuJoCo Python 離屏渲染的 framebuffer 上限（任務1）

**症狀**
```
ValueError: Image width 800 > framebuffer width 640.
```

**根因**
MuJoCo 模型預設離屏 framebuffer 是 640×480，`mujoco.Renderer` 要求的解析度不能超過它。與顯示卡能力無關。

**解法**
- 渲染時寬高 ≤ 640×480；或
- 在模型 XML 加大：`<visual><global offwidth="1920" offheight="1080"/></visual>`。
- 離屏渲染在 Intel Iris Xe 上用 `MUJOCO_GL=egl` 可正常運作。

---

## 總結：這些雷的共通點與心法

1. **共通根因**：Arch 是滾動更新，工具鏈（CMake 4、GCC 16）走在最前沿，比 MJPC 這種「pin 舊版相依」的專案假設的還新，於是撞出相容性/UB 問題。
2. **心法**：遇到「編不過」先看是 **CMake 政策**（雷①）還是 **編譯器警告當錯誤**（雷②）；遇到「編得過但執行崩」且崩在型別雙關相關處，優先懷疑 **strict-aliasing**（雷③），加 `-fno-strict-aliasing`。
3. **一次性成本**：以上全部解完後，`mjpc` 只需編譯一次，之後直接執行 `./build/bin/mjpc`，重開機也不用重編。
