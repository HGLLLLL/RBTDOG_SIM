// L2_wheel_spin.cpp —— 慢轉「單顆輪子」測試（第一個真正驅動馬達的測試）
// =============================================================================
// 目的：證明「非零指令真的能驅動馬達」——用最低能量、最可控的方式：只慢轉一顆輪子。
//
// ★ 為什麼相對安全：
//   - 只動【一顆輪子(foot)】，其他 15 個關節全部壓在零增益（限位關節完全不出力）
//   - 輪子是連續旋轉關節，沒有限位可撞
//   - 速度控制：目標很慢(預設 0.5 rad/s)，且 1 秒內緩慢加速，啟動力矩僅約 kd*v_des≈0.25 N·m
//   - 暴衝保護：每輪讀回實際轉速，超過 RUNAWAY 上限立即歸零中止
//
// ★ 執行前必須成立（硬前提）：
//   - 狗的【肚子墊高、四顆輪子懸空離地】（貼地會讓狗爬行/滑走）
//   - mc_ctrl 已凍結(SIGSTOP)或停止（程式會自己預檢，沒停就拒跑）
//   - 手指/線材/頭髮遠離輪子；旁邊有人、手放電源開關
//
// 執行： sudo ./L2_wheel_spin --confirm [--leg 0] [--vel 0.5] [--secs 3]
// 還原： SIGCONT 解凍 mc_ctrl（或 reboot）
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

// ---- 安全上限（保守）----
static const float VEL_CAP      = 1.5f;   // 允許設定的最大目標轉速 rad/s
static const float KD_CAP       = 6.0f;   // kd 上限
static const float TFF_CAP      = 2.0f;   // 前饋力矩上限 N·m（仍遠低於 28）
static const float RUNAWAY_VEL  = 3.0f;   // 實際轉速超過此值 → 立即中止
static const float RAMP_SEC     = 1.0f;   // 目標速度從 0 爬到設定值的時間

static std::atomic<bool> g_stop(false);
static void on_sigint(int) { g_stop.store(true); }
static uint64_t now_ns() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; }
static void sleep_ns(long ns) { struct timespec t{0, ns}; nanosleep(&t, nullptr); }

// 全部關節填零增益（安全底色）
static void zero_all(volatile spline_data_t* d) {
  for (int i = 0; i < 4; ++i) {
    volatile joint_control_t* J[4] = { &d->cmd.legs[i].abad, &d->cmd.legs[i].hip,
                                       &d->cmd.legs[i].knee, &d->cmd.legs[i].foot };
    for (int j = 0; j < 4; ++j) { J[j]->p_des=0; J[j]->v_des=0; J[j]->kp=0; J[j]->kd=0; J[j]->t_ff=0; }
    d->cmd.legs[i].flags = 1;
  }
}

int main(int argc, char** argv) {
  bool confirm = false; int leg = 0; float vel = 0.5f; double secs = 3.0; float kd = 0.5f; float tff = 0.0f;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--confirm")) confirm = true;
    else if (!strcmp(argv[i], "--leg")  && i+1<argc) leg = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--vel")  && i+1<argc) vel = atof(argv[++i]);
    else if (!strcmp(argv[i], "--kd")   && i+1<argc) kd  = atof(argv[++i]);
    else if (!strcmp(argv[i], "--tff")  && i+1<argc) tff = atof(argv[++i]);
    else if (!strcmp(argv[i], "--secs") && i+1<argc) secs = atof(argv[++i]);
  }
  if (!confirm) {
    fprintf(stderr, "用法： sudo %s --confirm [--leg 0..3] [--vel 0.5] [--kd 0.5] [--tff 0] [--secs 3]\n"
                    "前提：四顆輪子【懸空離地】、mc_ctrl 已停、旁邊有人手放電源。\n", argv[0]);
    return 1;
  }
  if (leg < 0 || leg > 3) { fprintf(stderr, "leg 必須 0..3\n"); return 1; }
  if (vel > VEL_CAP) { fprintf(stderr, "vel 上限 %.1f rad/s，太快不給跑\n", VEL_CAP); return 1; }
  if (vel <= 0) { fprintf(stderr, "vel 要 > 0\n"); return 1; }
  if (kd > KD_CAP) { fprintf(stderr, "kd 上限 %.1f，太大不給跑\n", KD_CAP); return 1; }
  if (kd <= 0) { fprintf(stderr, "kd 要 > 0\n"); return 1; }
  if (tff < 0 || tff > TFF_CAP) { fprintf(stderr, "tff 範圍 0~%.1f N·m\n", TFF_CAP); return 1; }
  std::signal(SIGINT, on_sigint);

  int fd = shm_open("/spline_shm", O_RDWR, 0666);
  if (fd == -1) { fprintf(stderr, "開不了 /spline_shm（要 sudo？）：%s\n", strerror(errno)); return 1; }
  void* p = mmap(nullptr, 1024*10, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { fprintf(stderr, "mmap 失敗：%s\n", strerror(errno)); return 1; }
  volatile spline_data_t* d = (volatile spline_data_t*)p;

  // ---- 預檢：確認沒有別人在寫 cmd（mc_ctrl 已停）----
  printf("[*] 預檢 0.4 秒：確認 mc_ctrl 已停 ...\n");
  uint32_t prev = d->cmd.consumer_flags[CONSUMER_CONTROL]; long trans = 0; uint64_t t0 = now_ns();
  while (now_ns() - t0 < 400000000ull) { uint32_t c = d->cmd.consumer_flags[CONSUMER_CONTROL];
    if (c != prev) { trans++; prev = c; } }
  if (trans > 4) { fprintf(stderr, "✗ 中止：cmd 旗標仍在跳動(%ld) → mc_ctrl 沒停。先 SIGSTOP/停掉它。\n", trans);
    munmap(p, 1024*10); return 2; }
  printf("[*] 預檢通過。慢轉 leg%d 輪子：目標 %.2f rad/s，kd=%.2f，t_ff=%.2f（啟動力矩約 %.2f N·m），其餘零增益。\n",
         leg, vel, kd, tff, tff + kd * vel);
  printf("[*] ★ 再次確認：四顆輪子有懸空離地嗎？（貼地會讓狗爬走）\n\n");

  long sent = 0; float max_seen_v = 0.f, max_seen_t = 0.f; bool runaway = false;
  uint64_t start = now_ns();
  while (!g_stop.load() && now_ns() - start < (uint64_t)(secs * 1e9)) {
    double el = (now_ns() - start) / 1e9;
    float v_des = vel * (float)(el < RAMP_SEC ? el / RAMP_SEC : 1.0);  // 緩慢爬升

    zero_all(d);                                   // 先把 16 關節全壓零增益
    volatile joint_control_t* w = &d->cmd.legs[leg].foot;   // 只對這顆輪子做速度控制
    w->p_des = 0; w->kp = 0;
    w->v_des = v_des; w->kd = kd; w->t_ff = tff;   // 力矩 = t_ff + kd*(v_des - v)；t_ff 提供掙脫力，kd 限速

    d->cmd.consumer_flags[CONSUMER_CONTROL] = 1;   // 設旗標（只碰 cmd 側）
    sent++;
    sleep_ns(2000000);                             // 2ms

    float av = std::fabs((float)d->state.legs[leg].foot.v);
    float at = std::fabs((float)d->state.legs[leg].foot.t);
    if (av > max_seen_v) max_seen_v = av;
    if (at > max_seen_t) max_seen_t = at;
    if (av > RUNAWAY_VEL) { runaway = true; break; }   // 暴衝保護
  }

  // ---- 收尾：全零 + 停止設旗標 → watchdog 讓馬達癱軟 ----
  zero_all(d);
  double dur = (now_ns() - start) / 1e9;

  printf("=== 結果（%.1f 秒）===\n", dur);
  if (runaway) printf("⚠️ 觸發暴衝保護（實際轉速 > %.1f rad/s），已緊急歸零中止！\n", RUNAWAY_VEL);
  printf("送出幀數           = %ld\n", sent);
  printf("目標轉速           = %.3f rad/s\n", vel);
  printf("實際最大轉速       = %.3f rad/s   ← 接近目標 = 輪子確實照指令轉了\n", max_seen_v);
  printf("實際最大力矩       = %.3f N·m     ← 慢轉應該很小\n", max_seen_t);
  printf("\n判讀：實際轉速接近目標 = 『非零指令能驅動馬達』獲證（L2 目標達成）。\n");
  printf("[*] 已歸零收尾，watchdog 會讓馬達癱軟。測完 SIGCONT 解凍 mc_ctrl 或 reboot 還原。\n");
  munmap(p, 1024*10);
  return runaway ? 3 : 0;
}
