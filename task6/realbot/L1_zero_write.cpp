// L1_zero_write.cpp —— 零增益寫入測試（安全驗證「寫得進 /spline_shm 且 daemon 認」）
// =============================================================================
// 目的：證明第三方程式能寫入馬達指令白板、且 daemon 會轉發我們的幀，
//       同時保證馬達力矩恆為 0（狗不會動）。
//
// ★ 安全三重保險：
//   1) kp=kd=t_ff=0 → 力矩 = kp*(p_des-p)+kd*(v_des-v)+t_ff ≡ 0（數學保證零出力）
//   2) 開跑前預檢：若偵測到 cmd 旗標仍在跳動（mc_ctrl 沒停），直接中止，不寫。
//   3) 需 --confirm 參數，且需 root（sudo）。程式結束後不再設旗標
//      → daemon watchdog 約 10 週期後自動清掉馬達指令，維持癱軟。
//
// 前提（執行前必須成立）：
//   - mc_ctrl 已停（否則兩個 producer 搶同一塊 cmd）
//   - 狗【平躺在地上】（停 mc_ctrl 後會癱軟；平躺才不會摔/掉）
//   - 旁邊有人、手放電源開關
//
// 執行： sudo ./L1_zero_write --confirm
// 還原： 測完 reboot 機器人（mc_ctrl 會自動重新起來）
// =============================================================================

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ctime>

enum { CONSUMER_CONTROL = 0, CONSUMER_OTHER = 1 };

typedef struct { float p_des, v_des, kp, kd, t_ff; } __attribute__((packed)) joint_control_t;
typedef struct { joint_control_t abad, hip, knee, foot; int32_t flags; } __attribute__((packed)) leg_control_t;
typedef struct { leg_control_t legs[4]; uint32_t consumer_flags[2]; } __attribute__((packed)) spline_cmd_data_t;
typedef struct { int32_t flags; float p, v, t; } __attribute__((packed)) joint_state_t;
typedef struct { joint_state_t abad, hip, knee, foot; } __attribute__((packed)) leg_state_t;
typedef struct { leg_state_t legs[4]; uint32_t consumer_flags[2]; } __attribute__((packed)) spline_state_data_t;
typedef struct { spline_cmd_data_t cmd; spline_state_data_t state; } __attribute__((packed)) spline_data_t;

static std::atomic<bool> g_stop(false);
static void on_sigint(int) { g_stop.store(true); }
static uint64_t now_ns() {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static void sleep_ns(long ns) { struct timespec t{0, ns}; nanosleep(&t, nullptr); }

// 把整塊 cmd.legs 全部填 0（含全部增益/前饋），每條腿 flags=1
static void write_zero_cmd(volatile spline_data_t* d) {
  for (int i = 0; i < 4; ++i) {
    volatile joint_control_t* J[4] = {
      &d->cmd.legs[i].abad, &d->cmd.legs[i].hip,
      &d->cmd.legs[i].knee, &d->cmd.legs[i].foot };
    for (int j = 0; j < 4; ++j) {
      J[j]->p_des = 0.f; J[j]->v_des = 0.f;
      J[j]->kp = 0.f;    J[j]->kd = 0.f;   J[j]->t_ff = 0.f;   // ★ 零增益零前饋
    }
    d->cmd.legs[i].flags = 1;   // 每條腿 flags 固定為 1
  }
}

int main(int argc, char** argv) {
  bool confirm = false;
  for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "--confirm")) confirm = true;
  if (!confirm) {
    fprintf(stderr,
      "用法： sudo %s --confirm\n"
      "前提：mc_ctrl 已停、狗平躺地上、旁邊有人手放電源開關。\n", argv[0]);
    return 1;
  }
  std::signal(SIGINT, on_sigint);

  int fd = shm_open("/spline_shm", O_RDWR, 0666);   // 需 root
  if (fd == -1) { fprintf(stderr, "開不了 /spline_shm（要 sudo？）：%s\n", strerror(errno)); return 1; }
  void* p = mmap(nullptr, 1024 * 10, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { fprintf(stderr, "mmap 失敗：%s\n", strerror(errno)); return 1; }
  volatile spline_data_t* d = (volatile spline_data_t*)p;

  // ---- 預檢：確認 mc_ctrl 真的停了（cmd 旗標不應再自己跳動）----
  printf("[*] 預檢 0.4 秒：確認沒有別人在寫 cmd ...\n");
  uint32_t prev = d->cmd.consumer_flags[CONSUMER_CONTROL];
  long trans = 0; uint64_t t0 = now_ns();
  while (now_ns() - t0 < 400000000ull) {
    uint32_t c = d->cmd.consumer_flags[CONSUMER_CONTROL];
    if (c != prev) { trans++; prev = c; }
  }
  if (trans > 4) {
    fprintf(stderr,
      "✗ 中止：偵測到 cmd 旗標仍在跳動（%ld 次）→ mc_ctrl 可能還在跑。\n"
      "  請先停掉它再執行，例如：\n"
      "    sudo pkill -f mc_ctrl        （或用對應的 systemctl stop）\n"
      "  然後確認 `ps aux | grep mc_ctrl` 只剩 grep 自己。\n", trans);
    munmap(p, 1024 * 10);
    return 2;
  }
  printf("[*] 預檢通過（旗標轉變 %ld 次，判定 mc_ctrl 已停）。\n", trans);
  printf("[*] 開始送【全零增益】指令 3 秒。力矩數學上恆為 0，狗不會動。\n\n");

  // ---- 主迴圈：全零寫入 + 旗標握手，約 500 Hz ----
  long sent = 0, ack = 0;
  float max_v = 0.f, max_t = 0.f;
  uint64_t start = now_ns();
  while (!g_stop.load() && now_ns() - start < 3000000000ull) {
    write_zero_cmd(d);                                   // 1) 全零填 cmd
    d->cmd.consumer_flags[CONSUMER_CONTROL] = 1;         // 2) 設旗標（只碰 cmd 側）
    sent++;
    sleep_ns(2000000);                                   // 3) 等 2ms 給 daemon 處理
    if (d->cmd.consumer_flags[CONSUMER_CONTROL] == 0) ack++;  //    被清 → daemon 認了

    for (int i = 0; i < 4; ++i) {                        // 4) 讀 state 確認沒動（只讀）
      const volatile joint_state_t* S[4] = {
        &d->state.legs[i].abad, &d->state.legs[i].hip,
        &d->state.legs[i].knee, &d->state.legs[i].foot };
      for (int j = 0; j < 4; ++j) {
        if (std::fabs(S[j]->v) > max_v) max_v = std::fabs(S[j]->v);
        if (std::fabs(S[j]->t) > max_t) max_t = std::fabs(S[j]->t);
      }
    }
  }

  // ---- 收尾：不再設旗標 → watchdog 會清掉馬達指令 → 維持癱軟 ----
  write_zero_cmd(d);
  double secs = (now_ns() - start) / 1e9;

  printf("=== 結果（%.1f 秒）===\n", secs);
  printf("送出幀數 sent      = %ld\n", sent);
  printf("被 daemon 清掉 ack = %ld  （%.0f%% → daemon 有消費我們的寫入）\n",
         ack, sent ? 100.0 * ack / sent : 0.0);
  printf("觀察期間最大關節速度 = %.4f rad/s   （應接近 0）\n", max_v);
  printf("觀察期間最大關節力矩 = %.4f N·m     （應接近 0）\n", max_t);
  printf("\n判讀：ack>0 代表『寫得進去且 daemon 認』；max_v/max_t≈0 代表『全程零出力沒動』。\n");
  printf("兩者同時成立 = 寫入路徑打通，且完全安全。\n");
  printf("[*] 程式已結束、不再設旗標，watchdog 會讓馬達維持癱軟。測完請 reboot 還原。\n");
  munmap(p, 1024 * 10);
  return 0;
}
