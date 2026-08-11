// L0_cmd_probe.cpp — 唯讀擷取原廠 mc_ctrl 寫進 /spline_shm 的「指令區」增益
//
// 為什麼要有這支：L0_shm_probe 只讀 state（p/v/t），但原廠運控每個週期還會把
//   joint_control_t{p_des, v_des, kp, kd, t_ff} 寫進同一塊 mapping 的 cmd 區。
//   那就是原廠對「這台輪足機」實際使用的伺服增益 —— 官方 GitHub 查無此數據，
//   但它一直在共享記憶體裡。輪足版沒有 LowLevel demo，這是唯一的一手來源。
//
// 編譯（狗上）： g++ -O2 -o L0_cmd_probe L0_cmd_probe.cpp -lrt
//
// 用法：
//   ./L0_cmd_probe                   人眼模式，10 Hz
//   ./L0_cmd_probe --secs 10         取樣 10 秒後印統計（每關節 kp/kd 的相異值 + 範圍）★ 主用
//   ./L0_cmd_probe --secs 10 --csv   同時把每筆原始樣本吐成 CSV（給離線分析）
//
// ★ 安全：全程 O_RDONLY + PROT_READ，不寫入、不需 sudo、不必停 mc_ctrl。

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ctime>
#include <map>

// ---- 結構定義（與 L0_shm_probe.cpp / L1_zero_write.cpp 相同，順序與 packed 不可改）----
typedef struct { float p_des, v_des, kp, kd, t_ff; }
    __attribute__((packed)) joint_control_t;

typedef struct { joint_control_t abad, hip, knee, foot; int32_t flags; }
    __attribute__((packed)) leg_control_t;

typedef struct { leg_control_t legs[4]; uint32_t consumer_flags[2]; }
    __attribute__((packed)) spline_cmd_data_t;

typedef struct { int32_t flags; float p, v, t; }
    __attribute__((packed)) joint_state_t;

typedef struct { joint_state_t abad, hip, knee, foot; }
    __attribute__((packed)) leg_state_t;

typedef struct { leg_state_t legs[4]; uint32_t consumer_flags[2]; }
    __attribute__((packed)) spline_state_data_t;

typedef struct { spline_cmd_data_t cmd; spline_state_data_t state; }
    __attribute__((packed)) spline_data_t;

static void* open_shm_ro(const char* path, size_t size) {
    int fd = shm_open(path, O_RDONLY, 0666);
    if (fd == -1) {
        fprintf(stderr, "[X] 開不了 %s：%s\n", path, strerror(errno));
        return nullptr;
    }
    void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "[X] mmap %s 失敗：%s\n", path, strerror(errno));
        return nullptr;
    }
    return p;
}

static const char* LEG_NAME[4] = {"leg0(FR)", "leg1(FL)", "leg2(RR)", "leg3(RL)"};
static const char* JN[4]       = {"abad", "hip ", "knee", "foot"};

static const joint_control_t* jc(const spline_data_t* d, int leg, int j) {
    const leg_control_t* L = &d->cmd.legs[leg];
    switch (j) {
        case 0:  return &L->abad;
        case 1:  return &L->hip;
        case 2:  return &L->knee;
        default: return &L->foot;
    }
}

static uint64_t now_ns() {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

// ---- 統計：kp/kd 幾乎必為離散常數，所以直接記「相異值 → 出現次數」 ----
struct Acc {
    std::map<float, long> kp_hist, kd_hist;
    float p_min = INFINITY, p_max = -INFINITY;   // p_des 範圍
    float t_min = INFINITY, t_max = -INFINITY;   // t_ff 範圍
    // 交叉驗證用：把 state 一起讀進來，檢查 t ?= kp*(p_des-p) + kd*(v_des-v) + t_ff
    double err_sum = 0, tau_pred_sum = 0, tau_real_sum = 0, resid_max = 0;
    long  n = 0;
};

static const joint_state_t* js(const spline_data_t* d, int leg, int j) {
    const leg_state_t* L = &d->state.legs[leg];
    switch (j) {
        case 0:  return &L->abad;
        case 1:  return &L->hip;
        case 2:  return &L->knee;
        default: return &L->foot;
    }
}

static void print_human(const spline_data_t* d) {
    printf("\033[2J\033[H");
    printf("=== /spline_shm 【指令區 cmd】唯讀 —— 原廠 mc_ctrl 正在下的增益 ===\n\n");
    for (int i = 0; i < 4; ++i) {
        printf("[%s]  flags=%d\n", LEG_NAME[i], d->cmd.legs[i].flags);
        for (int j = 0; j < 4; ++j) {
            const joint_control_t* c = jc(d, i, j);
            printf("  %s  p_des=%+8.4f  v_des=%+8.4f  kp=%8.3f  kd=%7.3f  t_ff=%+7.3f\n",
                   JN[j], c->p_des, c->v_des, c->kp, c->kd, c->t_ff);
        }
    }
    printf("\ncmd.consumer_flags = [%u, %u]  （會跳動＝mc_ctrl 正在寫）\n",
           d->cmd.consumer_flags[0], d->cmd.consumer_flags[1]);
    printf("\n(Ctrl+C 離開)\n");
    fflush(stdout);
}

int main(int argc, char** argv) {
    double secs = 0.0;
    bool csv = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--secs") && i + 1 < argc) secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--csv")) csv = true;
    }

    void* sp = open_shm_ro("/spline_shm", 1024 * 10);
    if (!sp) return 1;
    const spline_data_t* d = (const spline_data_t*)sp;

    if (secs <= 0.0) {                     // 人眼模式
        while (true) {
            print_human(d);
            struct timespec ts = {0, 100000000L};
            nanosleep(&ts, nullptr);
        }
    }

    // ---- 取樣模式：1 kHz 掃，記相異值 ----
    Acc acc[4][4];
    uint32_t prev_flag = d->cmd.consumer_flags[0];
    long flag_changes = 0;

    if (csv) {
        printf("t_ns");
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                printf(",l%d_%s_pdes,l%d_%s_vdes,l%d_%s_kp,l%d_%s_kd,l%d_%s_tff",
                       i, JN[j], i, JN[j], i, JN[j], i, JN[j], i, JN[j]);
        printf("\n");
    }

    uint64_t t0 = now_ns();
    uint64_t end = t0 + (uint64_t)(secs * 1e9);
    while (now_ns() < end) {
        uint32_t f = d->cmd.consumer_flags[0];
        if (f != prev_flag) { prev_flag = f; ++flag_changes; }

        if (csv) printf("%llu", (unsigned long long)(now_ns() - t0));
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const joint_control_t* c = jc(d, i, j);
                Acc& a = acc[i][j];
                a.kp_hist[c->kp]++;
                a.kd_hist[c->kd]++;
                if (c->p_des < a.p_min) a.p_min = c->p_des;
                if (c->p_des > a.p_max) a.p_max = c->p_des;
                if (c->t_ff  < a.t_min) a.t_min = c->t_ff;
                if (c->t_ff  > a.t_max) a.t_max = c->t_ff;
                const joint_state_t* s = js(d, i, j);
                double tau_pred = c->kp * (c->p_des - s->p)
                                + c->kd * (c->v_des - s->v) + c->t_ff;
                double resid = fabs(tau_pred - s->t);
                a.err_sum      += fabs(c->p_des - s->p);
                a.tau_pred_sum += tau_pred;
                a.tau_real_sum += s->t;
                if (resid > a.resid_max) a.resid_max = resid;
                a.n++;
                if (csv) printf(",%.5f,%.5f,%.4f,%.4f,%.5f",
                                c->p_des, c->v_des, c->kp, c->kd, c->t_ff);
            }
        }
        if (csv) printf("\n");

        struct timespec ts = {0, 1000000L};   // 1 kHz
        nanosleep(&ts, nullptr);
    }

    // ---- 報告 ----
    fprintf(stderr, "\n===== 取樣 %.1f 秒，樣本 %ld 筆 =====\n", secs, acc[0][0].n);
    fprintf(stderr, "cmd.consumer_flags[0] 變化 %ld 次（≈ %.0f Hz）—— 0 代表 mc_ctrl 沒在寫\n\n",
            flag_changes, flag_changes / secs);

    for (int i = 0; i < 4; ++i) {
        fprintf(stderr, "[%s]\n", LEG_NAME[i]);
        for (int j = 0; j < 4; ++j) {
            Acc& a = acc[i][j];
            fprintf(stderr, "  %s  kp:", JN[j]);
            for (auto& kv : a.kp_hist)
                fprintf(stderr, " %.3f(%.0f%%)", kv.first, 100.0 * kv.second / a.n);
            fprintf(stderr, "   kd:");
            for (auto& kv : a.kd_hist)
                fprintf(stderr, " %.3f(%.0f%%)", kv.first, 100.0 * kv.second / a.n);
            fprintf(stderr, "   p_des[%+.3f,%+.3f]  t_ff[%+.3f,%+.3f]\n",
                    a.p_min, a.p_max, a.t_min, a.t_max);
            fprintf(stderr, "        追蹤誤差 %5.2f°  預測力矩 %+6.3f  實測力矩 %+6.3f N·m"
                            "  (殘差最大 %.3f)\n",
                    a.err_sum / a.n * 57.29578,
                    a.tau_pred_sum / a.n, a.tau_real_sum / a.n, a.resid_max);
        }
    }
    return 0;
}
