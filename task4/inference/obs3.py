# === v3 模組（與 notebook cpg_rl_terrain3_colab 的 %%writefile cell 同步；改動請兩邊一起改）===
"""觀測建構：欄位順序與 v1 一致，last_action 長度決定 76/80。"""
import jax.numpy as jnp


def build_obs(grav, blin, gyro, dq, dqvel, cmd, last_action, contact, c):
    return jnp.concatenate([
        grav, blin, gyro,
        dq, dqvel,
        cmd, last_action, contact,
        c["rx"], c["rx_d"], c["ry"], c["ry_d"],
        jnp.sin(c["theta"]), jnp.cos(c["theta"]),
    ])