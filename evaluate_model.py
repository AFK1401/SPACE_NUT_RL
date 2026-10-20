"""
Formal Benchmark Evaluation for Rocket Lander
============================================
Runs 100 consecutive evaluation episodes on unseen physics seeds with frozen weights.
Generates statistical metrics: Mean Score, Std Dev, Min/Max, and Touchdown Success Rate.
"""

import time
import gymnasium as gym
import numpy as np
import torch
from quickstart_lander import RocketQNetwork, DEVICE

def evaluate(model_path="rocket_lander_model.pth", num_episodes=100):
    print("=" * 70)
    print(f"  RUNNING FORMAL BENCHMARK TEST: {num_episodes} UNSEEN FLIGHTS")
    print(f"  Model: '{model_path}' | Device: {DEVICE}")
    print("=" * 70)

    # 1. Load trained model in pure evaluation mode (zero exploration, frozen weights)
    model = RocketQNetwork(state_dim=8, action_dim=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Fast headless environment (no graphics overhead)
    env = gym.make("LunarLander-v3")

    scores = []
    durations = []
    landings = 0
    crashes = 0

    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            steps += 1

        scores.append(score)
        durations.append(steps)

        # In LunarLander, a score >= 200 is considered a successful soft landing.
        if score >= 200.0:
            landings += 1
        elif score < 0.0:
            crashes += 1

        print(f"\rTesting Flight {ep:3d}/{num_episodes} | Current Score: {score:7.2f} | Running Avg: {np.mean(scores):7.2f}", end="")

    total_time = time.time() - start_time
    env.close()

    # Calculate statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    avg_steps = np.mean(durations)
    success_rate = (landings / num_episodes) * 100
    crash_rate = (crashes / num_episodes) * 100

    print("\n\n" + "=" * 70)
    print("                    OFFICIAL BENCHMARK REPORT CARD                    ")
    print("=" * 70)
    print(f"Total Test Flights   : {num_episodes} (brand new seeds)")
    print(f"Elapsed Time         : {total_time:.2f} seconds ({num_episodes / total_time:.1f} flights/sec)")
    print("-" * 70)
    print(f"Mean Score (Average) : {mean_score:.2f}  [Solved threshold: >= 200.00]")
    print(f"Score Std Deviation  : {std_score:.2f}")
    print(f"Lowest Score (Min)   : {min_score:.2f}")
    print(f"Highest Score (Max)  : {max_score:.2f}")
    print(f"Average Steps/Flight : {avg_steps:.1f} steps (~{avg_steps*0.02:.2f} seconds)")
    print("-" * 70)
    print(f"Safe Landing Rate    : {landings}/{num_episodes} ({success_rate:.1f}%)")
    print(f"Crash Rate           : {crashes}/{num_episodes} ({crash_rate:.1f}%)")
    print("=" * 70)

    if mean_score >= 200.0:
        print("[SUCCESS] VERDICT: OFFICIALLY SOLVED AND CERTIFIED!")
    else:
        print("[NOTICE] VERDICT: Not quite reaching official 200 average threshold yet.")
    print("=" * 70 + "\n")

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": min_score,
        "max_score": max_score,
        "success_rate": success_rate,
        "crash_rate": crash_rate,
        "avg_steps": avg_steps
    }

if __name__ == "__main__":
    evaluate()
