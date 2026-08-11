// shm_probe.cpp — D1 EDU 輪足版 /spline_shm 唯讀偵察工具
//
// 編譯（機器上）： g++ -O2 -o shm_probe shm_probe.cpp -lrt
// 交叉編譯：       aarch64-linux-gnu-g++ -O2 -o shm_probe shm_probe.cpp -lrt
//
// 用法：
//   ./shm_probe              人眼可讀模式，10 Hz 刷新
//   ./shm_probe --csv        CSV 模式，1 kHz，輸出到 stdout
//   ./shm_probe --raw        傾印原始 bytes（結構對不上時用）
//
// ★ 安全設計：全程 O_RDONLY + PROT_READ，不加 O_CREAT。
//   讀不到就是讀不到，絕不會憑空建一塊假的騙自己。

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ctime>

// ---- 結構定義（照抄官方 lowlevel.h，順序與 packed 不可改）----
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

typedef struct {
    size_t timestamp;   // 奈秒
    float acc[3];       // m/s^2
    float gyro[3];      // rad/s
    float q[4];         // w, x, y, z（旋轉順序 zyx）
} __attribute__((packed)) nav_imu_t;

// ---- 唯讀開啟共享記憶體（刻意不加 O_CREAT）----
static void* open_shm_ro(const char* path, size_t size) {
    int fd = shm_open(path, O_RDONLY, 0666);
    if (fd == -1) {
        fprintf(stderr, "[X] 開不了 %s：%s\n", path, strerror(errno));
        fprintf(stderr, "    → 先用 `ls -l /dev/shm/` 確認檔案在不在\n");
        fprintf(stderr, "    → 若是 Permission denied，試試 sudo\n");
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

static const char* LEG_NAME[4] = {"leg0", "leg1", "leg2", "leg3"};
// ↑ 刻意不寫 FR/FL/RR/RL —— 因為官方四份文件互相矛盾。
//   請用步驟 0-7 手動扳動關節，實測出真正的對應再改這裡。

static void decode_flags(int32_t f, char* out, size_t n) {
    // 2026-08-11 修正：原本這裡減了 40，是錯的。實機驗證原始值直接就是攝氏度：
    //   腿關節讀到 42~47、輪子讀到 29~31（開機數分鐘後，物理上合理）。
    //   減 40 之後會變成 2~7°C / -10°C，室溫下不可能，曾害我們誤判「溫度資料不可信」。
    //   失聯的節點溫度與電壓都會讀到 0（daemon 歸零），可據此判斷馬達是否掉線。
    int temp = (f >> 8) & 0xFF;          // 直接是 °C，0 = 失聯
    int volt = (f >> 16) & 0xFF;         // 0 ~ 255 V，0 = 失聯
    snprintf(out, n, "en=%d %s%s%s%s%s T=%d°C V=%dV",
             f & 1,
             (f >> 1) & 1 ? "過壓 " : "",
             (f >> 2) & 1 ? "過流 " : "",
             (f >> 3) & 1 ? "過溫 " : "",
             (f >> 4) & 1 ? "超速 " : "",
             (f >> 5) & 1 ? "雙編碼器故障 " : "",
             temp, volt);
}

static void print_human(const spline_data_t* d, const nav_imu_t* imu) {
    printf("\033[2J\033[H");   // 清畫面
    printf("=== /spline_shm 唯讀偵察 ===\n\n");
    const char* JN[4] = {"ABAD", "HIP ", "KNEE", "FOOT"};
    for (int i = 0; i < 4; ++i) {
        const joint_state_t* js[4] = {
            &d->state.legs[i].abad, &d->state.legs[i].hip,
            &d->state.legs[i].knee, &d->state.legs[i].foot };
        printf("[%s]\n", LEG_NAME[i]);
        for (int j = 0; j < 4; ++j) {
            char fb[128];
            decode_flags(js[j]->flags, fb, sizeof fb);
            printf("  %s  p=%+8.4f rad  v=%+8.4f rad/s  t=%+7.3f N·m  | %s\n",
                   JN[j], js[j]->p, js[j]->v, js[j]->t, fb);
        }
        printf("\n");
    }
    printf("state.consumer_flags = [%u, %u]   (index0=mc_ctrl 的, index1=你的)\n",
           d->state.consumer_flags[0], d->state.consumer_flags[1]);
    if (imu) {
        printf("\nIMU  ts=%zu ns\n", imu->timestamp);
        printf("  quat(wxyz) = %+.4f %+.4f %+.4f %+.4f\n",
               imu->q[0], imu->q[1], imu->q[2], imu->q[3]);
        printf("  acc  = %+.3f %+.3f %+.3f m/s^2\n",
               imu->acc[0], imu->acc[1], imu->acc[2]);
        printf("  gyro = %+.3f %+.3f %+.3f rad/s\n",
               imu->gyro[0], imu->gyro[1], imu->gyro[2]);
    }
    printf("\n(Ctrl+C 離開)  ★ 用手扳單一關節，看哪個 leg 索引在動\n");
    fflush(stdout);
}

static void print_csv_header() {
    printf("t_ns");
    const char* JN[4] = {"abad", "hip", "knee", "foot"};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            printf(",leg%d_%s_p,leg%d_%s_v,leg%d_%s_t,leg%d_%s_flags",
                   i, JN[j], i, JN[j], i, JN[j], i, JN[j]);
    printf(",imu_ts,qw,qx,qy,qz,ax,ay,az,gx,gy,gz\n");
}

static void print_csv_row(const spline_data_t* d, const nav_imu_t* imu, uint64_t t) {
    printf("%llu", (unsigned long long)t);
    for (int i = 0; i < 4; ++i) {
        const joint_state_t* js[4] = {
            &d->state.legs[i].abad, &d->state.legs[i].hip,
            &d->state.legs[i].knee, &d->state.legs[i].foot };
        for (int j = 0; j < 4; ++j)
            printf(",%.6f,%.6f,%.6f,%d", js[j]->p, js[j]->v, js[j]->t, js[j]->flags);
    }
    if (imu)
        printf(",%zu,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
               imu->timestamp, imu->q[0], imu->q[1], imu->q[2], imu->q[3],
               imu->acc[0], imu->acc[1], imu->acc[2],
               imu->gyro[0], imu->gyro[1], imu->gyro[2]);
    else
        printf(",0,0,0,0,0,0,0,0,0,0,0");
    printf("\n");
}

int main(int argc, char** argv) {
    bool csv = false, raw = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--csv")) csv = true;
        if (!strcmp(argv[i], "--raw")) raw = true;
    }

    printf("[*] 結構大小自我檢查：spline_data_t = %zu bytes（應該遠小於 10240）\n",
           sizeof(spline_data_t));

    void* sp = open_shm_ro("/spline_shm", 1024 * 10);
    if (!sp) return 1;
    void* ip = open_shm_ro("/imu_shm", 1024 * 1);   // 沒有也不致命

    const spline_data_t* d   = (const spline_data_t*)sp;
    const nav_imu_t*     imu = (const nav_imu_t*)ip;

    if (raw) {   // 結構對不上時，肉眼看原始 bytes
        const unsigned char* b = (const unsigned char*)sp;
        for (int i = 0; i < 256; ++i) {
            printf("%02x ", b[i]);
            if (i % 16 == 15) printf("\n");
        }
        return 0;
    }

    if (csv) print_csv_header();

    // CSV 模式 1 kHz，人眼模式 10 Hz
    long period_ns = csv ? 1000000L : 100000000L;
    while (true) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        uint64_t t = (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec;

        if (csv) print_csv_row(d, imu, t);
        else     print_human(d, imu);

        struct timespec ts = {0, period_ns};
        nanosleep(&ts, nullptr);
    }
    return 0;
}