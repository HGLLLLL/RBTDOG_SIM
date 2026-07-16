"""cpg2：新舊版本動作映射 + 每腿抬腳。run: conda run -n rbtdog python task4/inference/tests/test_cpg2.py"""
import sys, numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import jax.numpy as jnp
import cpg2 as C


def main():
    assert C.detect_mode(12) == "fixed" and C.detect_mode(16) == "learnable"
    # fixed：action 12 → gc 全部 == G_C(0.08)
    mux, muy, om, gc = C.action_to_cpg_cmd(jnp.zeros(12), "fixed")
    assert gc.shape == (4,) and np.allclose(np.array(gc), C.G_C)
    assert mux.shape == (4,) and muy.shape == (4,) and om.shape == (4,)
    # learnable：action 16、gc 落在 [GC_MIN,GC_MAX]
    a = jnp.zeros(16)
    _, _, _, gc0 = C.action_to_cpg_cmd(a, "learnable")
    assert np.allclose(np.array(gc0), (C.GC_MIN + C.GC_MAX) / 2, atol=1e-6)  # tanh(0)=0→中點
    big = jnp.array(([0, 0, 0, 10.0]) * 4, dtype=jnp.float32)   # 第4欄 tanh→1 → GC_MAX
    _, _, _, gcmax = C.action_to_cpg_cmd(big, "learnable")
    assert np.allclose(np.array(gcmax), C.GC_MAX, atol=1e-4)
    # foot offsets：擺動相(sin>0) dz == gc*sinθ
    c = C.cpg_init()
    c = {**c, "theta": jnp.full(4, jnp.pi / 2)}                 # sinθ=1
    gc_test = jnp.array([0.05, 0.10, 0.03, 0.15])
    off = C.cpg_foot_offsets(c, gc_test)
    assert np.allclose(np.array(off[:, 2]), np.array(gc_test), atol=1e-6), off[:, 2]
    print("PASS test_cpg2")


if __name__ == "__main__":
    main()
