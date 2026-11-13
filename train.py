from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from torch import cpu

from Bi_rBot_env import Bi_rBot_Env

N_ENVS = 8
TOTAL_STEPS = 2000000
SAVE_EVERY = 100000
RUN_NAME = "PPO_" + datetime.now().strftime("%d,%m,%Y,%H:%M:%S")

if __name__ == "__main__":
    env = VecNormalize(make_vec_env(Bi_rBot_Env, N_ENVS, vec_env_cls=SubprocVecEnv))
    model = PPO(
        "MlpPolicy",
        env,
        batch_size=264,
        verbose=1,
        tensorboard_log="runs/",
        device="cpu",
    )
    checkpoint = CheckpointCallback(
        SAVE_EVERY // N_ENVS, "models/" + RUN_NAME, "ppo", save_vecnormalize=True
    )
    model.learn(TOTAL_STEPS, checkpoint, tb_log_name=RUN_NAME, progress_bar=True)
    model.save("models/" + RUN_NAME + "/final")
    env.save("models/" + RUN_NAME + "/vecnormalize.pkl")
