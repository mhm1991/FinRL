"""
FinRL XAU/USD Part 2: Train

Trains Stable-Baselines3 agents with FinRL's standard single-asset trading
environment. This is a FinRL-first baseline, not a leveraged Forex execution
environment.
"""

from __future__ import annotations

import argparse

from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from FinRL_XAUUSD_common import DATA_DIR
from FinRL_XAUUSD_common import MODEL_DIR
from FinRL_XAUUSD_common import RESULTS_DIR
from FinRL_XAUUSD_common import build_env_kwargs
from FinRL_XAUUSD_common import ensure_xau_dirs
from FinRL_XAUUSD_common import load_split
from FinRL_XAUUSD_common import parse_algorithms

MODEL_PARAMS = {
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    },
    "sac": {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
    "td3": {
        "batch_size": 100,
        "buffer_size": 100000,
        "learning_rate": 0.001,
    },
    "ddpg": {
        "batch_size": 128,
        "buffer_size": 50000,
        "learning_rate": 0.001,
    },
    "a2c": {
        "n_steps": 5,
        "ent_coef": 0.01,
        "learning_rate": 0.0007,
    },
}


def build_model_params(
    algo: str,
    *,
    ppo_n_steps: int,
    ppo_batch_size: int,
) -> dict:
    params = MODEL_PARAMS[algo].copy()
    if algo == "ppo":
        params["n_steps"] = ppo_n_steps
        params["batch_size"] = ppo_batch_size
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FinRL agents for XAU.")
    parser.add_argument("--algo", default="ppo", help="ppo, sac, td3, ddpg, a2c, or all")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument(
        "--ppo-n-steps",
        type=int,
        default=2048,
        help="PPO rollout length before each update.",
    )
    parser.add_argument(
        "--ppo-batch-size",
        type=int,
        default=128,
        help="PPO minibatch size. Should divide --ppo-n-steps for one-env training.",
    )
    parser.add_argument("--initial-amount", type=float, default=100000.0)
    parser.add_argument("--hmax", type=int, default=100)
    parser.add_argument("--cost-pct", type=float, default=0.0002)
    parser.add_argument("--reward-scaling", type=float, default=1e-4)
    parser.add_argument("--train-file", default=str(DATA_DIR / "xau_train.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_xau_dirs()

    train = load_split(args.train_file)
    env_kwargs = build_env_kwargs(
        train,
        initial_amount=args.initial_amount,
        hmax=args.hmax,
        cost_pct=args.cost_pct,
        reward_scaling=args.reward_scaling,
    )
    print(
        "XAU train environment:",
        f"stock_dim={env_kwargs['stock_dim']}",
        f"state_space={env_kwargs['state_space']}",
    )

    train_env = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = train_env.get_sb_env()

    for algo in parse_algorithms(args.algo):
        model_params = build_model_params(
            algo,
            ppo_n_steps=args.ppo_n_steps,
            ppo_batch_size=args.ppo_batch_size,
        )
        if algo == "ppo" and args.timesteps < model_params["n_steps"]:
            print(
                "Warning: --timesteps is lower than PPO --ppo-n-steps; "
                f"Stable-Baselines3 will still collect one full "
                f"{model_params['n_steps']}-step rollout."
            )
        agent = DRLAgent(env=env_train)
        model = agent.get_model(algo, model_kwargs=model_params)
        model.set_logger(
            configure(str(RESULTS_DIR / algo), ["stdout", "csv", "tensorboard"])
        )

        print(f"Training {algo} for {args.timesteps} timesteps")
        trained = agent.train_model(
            model=model,
            tb_log_name=f"xau_{algo}",
            total_timesteps=args.timesteps,
        )
        model_path = MODEL_DIR / f"agent_{algo}"
        trained.save(str(model_path))
        print(f"Saved {algo} model to {model_path}.zip")


if __name__ == "__main__":
    main()
