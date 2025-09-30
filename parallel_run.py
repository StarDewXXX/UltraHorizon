import asyncio
import os
import random
import time
import numpy as np
import argparse
from multiprocessing import Process
from concurrent.futures import ProcessPoolExecutor

from interface import run_single_experiment  # 或 interface_runner，取决于你定义的位置
from envs.grid_env.env import MysteryGridEnvironment
from envs.seq_env.env import SequenceExploreEnvironment
from envs.bio_env.env import GeneticsLabEnvironment
from OpenManus.app.config import config

# 映射 ENV 参数
ENV_MAP = {
    "grid": MysteryGridEnvironment,
    "seq": SequenceExploreEnvironment,
    "bio": GeneticsLabEnvironment,
}


def run_experiment_process(env_cls, required_steps, exp_id, exp_dir, max_run_steps, free):
    os.environ["EXP_DIR"] = exp_dir
    seed = int(time.time() * 1000) % (2**32) + exp_id
    random.seed(seed)
    np.random.seed(seed)
    asyncio.run(run_single_experiment(env_cls, required_steps, exp_id, exp_dir, max_run_steps, free))


def run_all_experiments_mp(env_cls, required_steps, n, parent_dir, max_concurrency, max_run_steps, free):
    with ProcessPoolExecutor(max_workers=max_concurrency) as executor:
        futures = []
        for i in range(n):
            exp_dir = os.path.join(parent_dir, f"experiment-{i+1}")
            os.makedirs(exp_dir, exist_ok=True)
            futures.append(executor.submit(run_experiment_process, env_cls, required_steps, i+1, exp_dir, max_run_steps, free))
        for f in futures:
            f.result()  # 等待所有完成


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple experiments with configurable parameters.")
    parser.add_argument("--env", type=str, choices=["grid", "seq", "bio"], default="seq", help="Choose environment: grid / seq / bio")
    parser.add_argument("--steps", type=int, default=50, help="Required environment steps")
    parser.add_argument("--index", type=str, default="part1", help="Experiment index")
    parser.add_argument("--n_experiments", type=int, default=16, help="Total number of experiments")
    parser.add_argument("--max_concurrency", type=int, default=4, help="Maximum concurrency")
    parser.add_argument("--exp_folder", type=str, default=4, help="Experiment log save path")
    parser.add_argument("--free", type=bool, default=False, help="Whether to specify the agent's number of interactions. If true, the agent can submit the final answer at any time")


    args = parser.parse_args()

    env_cls = ENV_MAP[args.env]
    required_steps = args.steps
    index = args.index
    n_experiments = args.n_experiments
    max_concurrency = args.max_concurrency
    exp_folder = args.exp_folder
    free = args.free

    if args.env == "bio":
        max_run_steps = int(required_steps * 8)
    else:
        max_run_steps = int(required_steps * 4)

    # 从 config 获取模型名
    cfg = config.llm
    MODEL = cfg.get("default", cfg["default"]).model.split("/")[-1]
    window_size = int(os.getenv("WINDOW_SIZE", 200))

    parent_dir = f"user/{exp_folder}/{env_cls.__name__}_{MODEL}_steps_{required_steps}_wdsize_{window_size}_{index}"

    print("\n")
    print(f"[*] [Agent LLM]: {MODEL}\n[Environment]: {env_cls.__name__}\n[Required Steps]: {required_steps}\n[Output Parent Dir]: {parent_dir}")
    # input("Press Enter to continue...")

    run_all_experiments_mp(env_cls, required_steps, n_experiments, parent_dir, max_concurrency, max_run_steps, free)
