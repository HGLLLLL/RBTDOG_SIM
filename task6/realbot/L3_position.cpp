// L3_position.cpp —— 位置控制測試（懸空輪子），移植自 zsl-1 python lowlevel_demo 的邏輯
// =============================================================================
// 目的：在你這台輪足實機上驗證「位置控制」——命令輪子轉到某角度並【定住】，
//       而不是像 L2 那樣一直轉。這是 CPG-RL policy 部署最常用的控制方式。
//
// 移植自官方點足 python 範例的三個範式：
//   1) 讀「當前角度」當起點             （避免 p_des 一開始就跟實際差很多 → 力矩突跳）
//   2) 用 ratio 把 p_des 從起點【內插】到目標（平滑過渡）
//   3) 結束用 kp=0、kd=大 的【卸力收尾】（軟軟停住，不硬鎖）
//
// ★ 安全設計：
//   - 只動 leg0.foot（輪子），其餘 15 關節壓零增益
//   - 內插讓每一步的位置誤差都很小 → 力矩自然小
//   - kp 用溫和值(預設 8，非範例的 80；80 是給撐體重的腿關節)
//   - 雙保險：實際轉速 > 3 rad/s 或 實際力矩 > 5 N·m → 立即歸零中止
//   - 卸力收尾 + watchdog 兜底
//
// ★ 前提：leg0 輪子【懸空離地】、mc_ctrl 已停(SIGSTOP)、旁邊有人手放電源。
//
// 用法： sudo ./L3_position --confirm [--leg 0] [--delta 0.5] [--kp 8] [--kd 0.5] [--hold 2]
// =============================================================================

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
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

// ---- 安全上限 ----
static const float KP_CAP        = 30.0f;  // 這個輪子測試的 kp 上限（腿關節之後吊掛才用 80）
static const float KD_CAP        = 3.0f;
static const float DELTA_CAP     = 3.0f;   // 目標轉動角度上限 rad
static const float RUNAWAY_VEL   = 3.0f;   // 實際轉速超過 → 中止
static const float TORQUE_ABORT  = 5.0f;   // 實際力矩超過 → 中止
static const float RAMP_SEC      = 2.0f;   // 內插時間（p_des 從起點爬到目標）

static std::atomic<bool> g_stop(false);
static void on_sigint(int) { g_stop.store(true); }
static uint64_t now_ns() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; }
static void sleep_ns(long ns) { struct timespec t{0, ns}; nanosleep(&t, nullptr); }

static void zero_all(volatile spline_data_t* d) {
  for (int i = 0; i < 4; ++i) {
    volatile joint_control_t* J[4] = { &d->cmd.legs[i].abad, &d->cmd.legs[i].hip,
                                       &d->cmd.legs[i].knee, &d->cmd.legs[i].foot };
    for (int j = 0; j < 4; ++j) { J[j]->p_des=0; J[j]->v_des=0; J[j]->kp=0; J[j]->kd=0; J[j]->t_ff=0; }
    d->cmd.legs[i].flags = 1;
  }
}

int main(int argc, char** argv) {
  bool confirm = false; int leg = 0;
  float delta = 0.5f, kp = 8.0f, kd = 0.5f; double hold = 2.0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--confirm")) confirm = true;
    else if (!strcmp(argv[i], "--leg")   && i+1<argc) leg   = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--delta") && i+1<argc) delta = atof(argv[++i]);
    else if (!strcmp(argv[i], "--kp")    && i+1<argc) kp    = atof(argv[++i]);
    else if (!strcmp(argv[i], "--kd")    && i+1<argc) kd    = atof(argv[++i]);
    else if (!strcmp(argv[i], "--hold")  && i+1<argc) hold  = atof(argv[++i]);
  }
  if (!confirm) {
    fprintf(stderr, "用法： sudo %s --confirm [--leg 0..3] [--delta 0.5] [--kp 8] [--kd 0.5] [--hold 2]\n"
                    "前提：輪子懸空離地、mc_ctrl 已停、旁邊有人手放電源。\n", argv[0]);
    return 1;
  }
  if (leg<0||leg>3) { fprintf(stderr,"leg 0..3\n"); return 1; }
  if (std::fabs(delta) > DELTA_CAP) { fprintf(stderr,"delta 絕對值上限 %.1f rad\n", DELTA_CAP); return 1; }
  if (kp<=0 || kp>KP_CAP) { fprintf(stderr,"kp 範圍 0~%.1f\n", KP_CAP); return 1; }
  if (kd<0 || kd>KD_CAP)  { fprintf(stderr,"kd 範圍 0~%.1f\n", KD_CAP); return 1; }
  std::signal(SIGINT, on_sigint);

  int fd = shm_open("/spline_shm", O_RDWR, 0666);
  if (fd == -1) { fprintf(stderr,"開不了 /spline_shm（要 sudo？）：%s\n", strerror(errno)); return 1; }
  void* p = mmap(nullptr, 1024*10, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { fprintf(stderr,"mmap 失敗：%s\n", strerror(errno)); return 1; }
  volatile spline_data_t* d = (volatile spline_data_t*)p;

  // ---- 預檢：mc_ctrl 已停 ----
  printf("[*] 預檢 0.4 秒：確認 mc_ctrl 已停 ...\n");
  uint32_t prev = d->cmd.consumer_flags[CONSUMER_CONTROL]; long trans=0; uint64_t tc=now_ns();
  while (now_ns()-tc < 400000000ull) { uint32_t c=d->cmd.consumer_flags[CONSUMER_CONTROL];
    if (c!=prev){trans++;prev=c;} }
  if (trans > 4) { fprintf(stderr,"✗ 中止：cmd 旗標仍在跳動(%ld) → mc_ctrl 沒停。\n", trans);
    munmap(p,1024*10); return 2; }

  // ---- 讀起點角度（移植自範例的 init_q）----
  float p0 = (float)d->state.legs[leg].foot.p;
  float p_target = p0 + delta;
  printf("[*] 預檢通過。leg%d 輪子位置控制：起點 %.4f → 目標 %.4f rad（Δ=%.2f），kp=%.1f kd=%.2f\n",
         leg, p0, p_target, delta, kp, kd);
  printf("[*] %.1f 秒內平滑內插到目標，再撐住 %.1f 秒。★ 輪子有懸空嗎？\n\n", RAMP_SEC, hold);

  long sent=0; float max_v=0, max_t=0, last_p=p0; const char* abort_reason=nullptr;
  uint64_t start = now_ns();
  double total = RAMP_SEC + hold;
  while (!g_stop.load()) {
    double el = (now_ns()-start)/1e9;
    if (el > total) break;
    double ratio = el < RAMP_SEC ? el / RAMP_SEC : 1.0;      // 內插比例 0→1
    float p_des = p0 + (float)ratio * delta;                 // 平滑內插

    zero_all(d);
    volatile joint_control_t* w = &d->cmd.legs[leg].foot;
    w->p_des = p_des; w->kp = kp; w->kd = kd; w->v_des = 0; w->t_ff = 0;
    d->cmd.consumer_flags[CONSUMER_CONTROL] = 1;
    sent++;
    sleep_ns(2000000);

    float v = std::fabs((float)d->state.legs[leg].foot.v);
    float t = std::fabs((float)d->state.legs[leg].foot.t);
    last_p = (float)d->state.legs[leg].foot.p;
    if (v > max_v) max_v = v;
    if (t > max_t) max_t = t;
    if (v > RUNAWAY_VEL) { abort_reason = "轉速暴衝"; break; }
    if (t > TORQUE_ABORT) { abort_reason = "力矩超限"; break; }
  }

  // ---- 卸力收尾（移植自範例：kp=0、kd 大，軟軟停住）----
  for (int k = 0; k < 150 && !abort_reason; ++k) {   // ~0.3s
    zero_all(d);
    volatile joint_control_t* w = &d->cmd.legs[leg].foot;
    w->kd = 3.0f;   // 只留阻尼
    d->cmd.consumer_flags[CONSUMER_CONTROL] = 1;
    sleep_ns(2000000);
  }
  zero_all(d);

  printf("=== 結果 ===\n");
  if (abort_reason) printf("⚠️ 觸發保護（%s），已歸零中止！\n", abort_reason);
  printf("送出幀數     = %ld\n", sent);
  printf("起點角度     = %.4f rad\n", p0);
  printf("目標角度     = %.4f rad\n", p_target);
  printf("最後實際角度 = %.4f rad   ← 接近目標 = 位置控制成功「轉到並定住」\n", last_p);
  printf("誤差         = %.4f rad\n", last_p - p_target);
  printf("過程最大轉速 = %.3f rad/s\n", max_v);
  printf("過程最大力矩 = %.3f N·m\n", max_t);
  printf("\n判讀：最後角度 ≈ 目標角度 = 位置控制在你這台實機驗證成功。\n");
  printf("（可在撐住階段用手輕撥輪子，它會彈回目標 = 位置剛度的手感）\n");
  printf("[*] 已卸力收尾，watchdog 兜底。測完 SIGCONT 解凍 mc_ctrl 還原。\n");
  munmap(p, 1024*10);
  return abort_reason ? 3 : 0;
}
