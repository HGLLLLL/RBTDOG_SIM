"""obs2：76/80 維相容。run: conda run -n rbtdog python task4/inference/tests/test_obs2.py"""
import sys, numpy as np
sys.path.insert(0, "/home/huang/rbtdog_sim/task4/inference")
import jax.numpy as jnp
import cpg2 as C
import obs2 as O


def _dummy(last_len):
    z3 = jnp.zeros(3); z12 = jnp.zeros(12); z4 = jnp.zeros(4)
    return O.build_obs(z3, z3, z3, z12, z12, jnp.zeros(3),
                       jnp.zeros(last_len), z4, C.cpg_init())


def main():
    assert _dummy(12).shape == (76,), _dummy(12).shape
    assert _dummy(16).shape == (80,), _dummy(16).shape
    # 欄位順序：前 3 = grav
    grav = jnp.array([0.1, 0.2, 0.3])
    o = O.build_obs(grav, jnp.zeros(3), jnp.zeros(3), jnp.zeros(12), jnp.zeros(12),
                    jnp.zeros(3), jnp.zeros(16), jnp.zeros(4), C.cpg_init())
    assert np.allclose(np.array(o[:3]), np.array(grav))
    print("PASS test_obs2")


if __name__ == "__main__":
    main()
