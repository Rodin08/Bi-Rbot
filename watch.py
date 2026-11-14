import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import Bi_rBot_env


def main():
    models = Path("models")

    if not models.exists():
        sys.exit(
            "No models/ folder. Start watch.py from inside Bi_rBot/, or train first."
        )

    finished = [p for p in models.iterdir() if (p / "final.zip").exists()]

    if not finished:
        sys.exit("No final.zip found")

    RUN = max(finished, key=lambda p: p.stat().st_mtime)
    print("Watching:", RUN)

    raw_env = Bi_rBot_env.Bi_rBot_Env()
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(str(RUN / "vecnormalize.pkl"), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()

    model = PPO.load(RUN / "final.zip")

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        while viewer.is_running():
            assert isinstance(obs, np.ndarray)
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            viewer.sync()
            time.sleep(Bi_rBot_env.FRAME_SKIP * raw_env.model.opt.timestep)


if __name__ == "__main__":
    main()
