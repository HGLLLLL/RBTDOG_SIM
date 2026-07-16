"""MJX 地形環境本機 CPU 冒煙測試。run: conda run -n rbtdog python task4/inference/tests/test_env2_smoke.py"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import cpg2 as C
import go2_terrain2_env as E


def main():
    jinvs = C.leg_ik_consts("mujoco_menagerie/unitree_go2/scene_mjx.xml")
    env = E.Go2Terrain2Env(jinvs)
    assert env.observation_size == 80 and env.action_size == 16
    for seed in [0, 1, 2]:
        s = jax.jit(env.reset)(jax.random.PRNGKey(seed))
        assert s.obs.shape == (80,), s.obs.shape
        assert float(s.done) == 0.0, "reset 不應立即 done"
        s2 = jax.jit(env.step)(s, jnp.zeros(16))
        r = float(s2.reward)
        assert np.isfinite(r), f"reward 非有限 {r}"
        # 站立零動作數步後仍在地形上（rel_h 合理、未穿透）
        st = s
        for _ in range(20):
            st = jax.jit(env.step)(st, jnp.zeros(16))
        rel_h = float(st.metrics["rel_h"])
        assert 0.10 < rel_h < 0.45, f"rel_h 異常(可能穿透/飛起) {rel_h}"
    print("PASS test_env2_smoke")


if __name__ == "__main__":
    main()
