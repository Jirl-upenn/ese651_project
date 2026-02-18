# grade_race.py — Gradescope evaluation script for drone racing.
# Modeled after play_race.py but instrumented for grading.
#
# Usage (called by run_autograder):
#   python autograder/grade_race.py \
#       --checkpoint /path/to/model_1000.pt \
#       --agent_pkl  /path/to/params/agent.pkl \
#       --results_path /autograder/results/results.json

"""Launch Isaac Sim Simulator first."""

import sys
import os
import json
import argparse
import pickle
import traceback

# --- Ensure local rsl_rl is on the path before anything else ---
local_rsl_path = os.path.abspath("src/third_parties/rsl_rl_local")
if os.path.exists(local_rsl_path):
    sys.path.insert(0, local_rsl_path)

# ── CLI args (parsed before AppLauncher so --help works) ──────────
parser = argparse.ArgumentParser(description="Gradescope drone-race evaluator")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt checkpoint")
parser.add_argument("--agent_pkl", type=str, default="", help="Path to params/agent.pkl (optional)")
parser.add_argument("--results_path", type=str, default="/autograder/results/results.json",
                    help="Where to write the Gradescope results JSON")

from isaaclab.app import AppLauncher

# AppLauncher needs its own args on the same parser
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless, single-env operation
args_cli.headless = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import heavy deps (Isaac Sim is running) ──────────
import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# Register the custom quadcopter task
import src.isaac_quad_sim2real.tasks  # noqa: F401

# ── Constants ─────────────────────────────────────────────────────
TASK_NAME = "Isaac-Quadcopter-Race-v0"
TRACK_NAME = "powerloop"
NUM_GATES = 7  # powerloop track has 7 gates
CRASH_THRESHOLD = 100  # _crashed > 100 means crashed


def write_results(path: str, output: str, detail: str, passed: bool):
    """Write a Gradescope-compatible results.json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results = {
        "output": output,
        "tests": [
            {
                "name": "Race Evaluation",
                "output": detail,
                "status": "passed" if passed else "failed",
            }
        ],
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    results_path = args_cli.results_path

    try:
        # ── 1. Build environment config ───────────────────────────
        env_cfg = parse_env_cfg(
            TASK_NAME,
            device=args_cli.device,
            num_envs=1,
            use_fabric=True,
        )
        env_cfg.is_train = False
        env_cfg.track_name = TRACK_NAME
        env_cfg.debug_vis = False
        env_cfg.seed = 42
        # Disable motor noise for deterministic evaluation
        if hasattr(env_cfg, "max_motor_noise_std"):
            env_cfg.max_motor_noise_std = 0.0

        # ── 2. Build agent config ────────────────────────────────
        if args_cli.agent_pkl and os.path.isfile(args_cli.agent_pkl):
            with open(args_cli.agent_pkl, "rb") as f:
                agent_cfg = pickle.load(f)
        else:
            # Fall back to the registered default
            agent_cfg = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")

        agent_cfg_dict = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)

        # ── 3. Create gym environment ────────────────────────────
        env = gym.make(TASK_NAME, cfg=env_cfg)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        env = RslRlVecEnvWrapper(env)

        # ── 4. Load model ────────────────────────────────────────
        ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
        ppo_runner.load(args_cli.checkpoint, load_optimizer=False)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # ── 5. Run evaluation loop ───────────────────────────────
        obs = env.get_observations()
        if hasattr(obs, "get"):
            obs = obs["policy"]

        # Access the underlying QuadcopterEnv
        raw_env = env.unwrapped
        strategy = raw_env.strategy
        max_timesteps = int(raw_env.max_episode_length)

        prev_gates_passed = 0
        gate_events = []   # list of (gate_number, timestep)
        crash_info = None  # (x, y, z, timestep) or None
        lap_timestep = None

        for step in range(max_timesteps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)
                if hasattr(obs, "get"):
                    obs = obs["policy"]

            current_timestep = int(raw_env.episode_length_buf[0].item())
            n_gates = int(strategy._n_gates_passed[0].item())
            crashed_counter = int(raw_env._crashed[0].item())

            # ── Gate pass detection ──────────────────────────────
            if n_gates > prev_gates_passed:
                for g in range(prev_gates_passed + 1, n_gates + 1):
                    gate_events.append((g, current_timestep))
                prev_gates_passed = n_gates

            # ── Lap detection ────────────────────────────────────
            if n_gates >= NUM_GATES and lap_timestep is None:
                lap_timestep = current_timestep

            # ── Crash detection ──────────────────────────────────
            if crashed_counter > CRASH_THRESHOLD and crash_info is None:
                pos = raw_env._robot.data.root_link_pos_w[0].tolist()
                crash_info = (pos[0], pos[1], pos[2], current_timestep)

            # ── Episode done (died or timeout) ───────────────────
            if dones.any():
                break

        # ── 6. Build report ──────────────────────────────────────
        lines = [f"Track: {TRACK_NAME} ({NUM_GATES} gates)"]

        for gate_num, ts in gate_events:
            lines.append(f"Gate {gate_num} passed at timestep {ts}")

        if crash_info:
            x, y, z, ts = crash_info
            lines.append(f"CRASH at timestep {ts} | Position: ({x:.2f}, {y:.2f}, {z:.2f})")

        if lap_timestep is not None:
            lines.append(f"\nLap completed in {lap_timestep} timesteps")
            passed = True
        else:
            lines.append(f"\nGates passed: {prev_gates_passed}/{NUM_GATES}")
            if crash_info:
                lines.append("Result: Crashed before completing the lap")
            else:
                lines.append("Result: Timed out before completing the lap")
            passed = False

        detail = "\n".join(lines)
        if lap_timestep is not None:
            summary = f"Lap: {lap_timestep} ts"
        elif crash_info:
            summary = f"Gates: {prev_gates_passed}/{NUM_GATES} | Crashed at ts {crash_info[3]}"
        else:
            summary = f"Gates: {prev_gates_passed}/{NUM_GATES} | Timed out"

        write_results(results_path, summary, detail, passed)

        env.close()

    except Exception:
        tb = traceback.format_exc()
        write_results(
            results_path,
            "Grading script encountered an error.",
            f"Error during evaluation:\n{tb}",
            False,
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
