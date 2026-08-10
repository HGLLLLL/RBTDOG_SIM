// shm_handshake.cpp —— L0：唯讀觀察 /spline_shm 的握手旗標節奏
// ★ 全程 O_RDONLY + PROT_READ，只讀不寫，零風險。
// 目的：親眼確認 mc_ctrl 正以高頻在寫 cmd 並設旗標、daemon 持續清它，
//       理解 L1 要接手的就是「設 cmd.consumer_flags[CONTROL] 這個旗標」。

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ctime>

typedef struct { float p_des, v_des, kp, kd, t_ff; } __attribute__((packed)) joint_control_t;
typedef struct { joint_control_t abad, hip, knee, foot; int32_t flags; } __attribute__((packed)) leg_control_t;
typedef struct { leg_control_t legs[4]; uint32_t consumer_flags[2]; } __attribute__((packed)) spline_cmd_data_t;
typedef struct { int32_t flags; float p, v, t; } __attribute__((packed)) joint_state_t;
typedef struct { joint_state_t abad, hip, knee, foot; } __attribute__((packed)) leg_state_t;
typedef struct { leg_state_t legs[4]; uint32_t consumer_flags[2]; } __attribute__((packed)) spline_state_data_t;
typedef struct { spline_cmd_data_t cmd; spline_state_data_t state; } __attribute__((packed)) spline_data_t;

static uint64_t now_ns() {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

int main() {
  int fd = shm_open("/spline_shm", O_RDONLY, 0666);   // 唯讀
  if (fd == -1) { fprintf(stderr, "開不了 /spline_shm: %s\n", strerror(errno)); return 1; }
  void* p = mmap(nullptr, 1024 * 10, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { fprintf(stderr, "mmap 失敗: %s\n", strerror(errno)); return 1; }
  const volatile spline_data_t* d = (const volatile spline_data_t*)p;

  printf("[*] 唯讀觀察握手旗標 2 秒（只讀不寫）\n");
  printf("[*] mc_ctrl 目前對 leg0 下的指令：hip  kp=%.1f  kd=%.1f  p_des=%.4f  t_ff=%.3f\n",
         (float)d->cmd.legs[0].hip.kp, (float)d->cmd.legs[0].hip.kd,
         (float)d->cmd.legs[0].hip.p_des, (float)d->cmd.legs[0].hip.t_ff);
  printf("[*] （kp 若明顯>0，代表 mc_ctrl 正在用力控制關節）\n\n");

  uint32_t prev_cmd_ctrl = d->cmd.consumer_flags[0];
  uint32_t prev_st_ctrl  = d->state.consumer_flags[0];
  long samples = 0, cmdctrl_set = 0, trans_cmdctrl = 0, trans_stctrl = 0;

  uint64_t t0 = now_ns(), t;
  while ((t = now_ns()) - t0 < 2000000000ull) {
    uint32_t c  = d->cmd.consumer_flags[0];
    uint32_t sc = d->state.consumer_flags[0];
    if (c) cmdctrl_set++;
    if (c  != prev_cmd_ctrl) { trans_cmdctrl++; prev_cmd_ctrl = c; }
    if (sc != prev_st_ctrl)  { trans_stctrl++;  prev_st_ctrl = sc; }
    samples++;
  }
  double secs = (now_ns() - t0) / 1e9;

  printf("=== 2 秒觀察結果 ===\n");
  printf("取樣次數：%ld （每秒約 %.1f 萬次，我們取樣遠快於旗標變化）\n",
         samples, samples / secs / 1e4);
  printf("\ncmd.consumer_flags[CONTROL]（mc_ctrl 設、daemon 清）：\n");
  printf("  被設起來的比例：%.1f%%\n", 100.0 * cmdctrl_set / samples);
  printf("  0<->1 轉變：%ld 次 → 約 %.0f 個寫入週期/秒（mc_ctrl 寫白板的頻率）\n",
         trans_cmdctrl, trans_cmdctrl / 2.0 / secs);
  printf("\nstate.consumer_flags[CONTROL] 轉變：%ld 次（daemon 更新狀態的節奏）\n", trans_stctrl);
  printf("目前 state.consumer_flags = [%u, %u]   ← index1(CONSUMER_OTHER) 是留給你的那格\n",
         d->state.consumer_flags[0], d->state.consumer_flags[1]);

  printf("\n[*] 判讀：若上面看到 mc_ctrl 以每秒數百~上千次在寫 cmd、旗標不停 0<->1，\n");
  printf("    就代表握手機制活著。L1 要做的就是「停掉 mc_ctrl，改由你來設這個旗標」。\n");
  return 0;
}
