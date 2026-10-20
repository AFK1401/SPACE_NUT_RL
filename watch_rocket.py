"""
Interactive Visual Player for Trained Rocket Lander
===================================================
Run this script to watch your trained rocket land in a persistent Pygame window!
Controls:
  - ESC or Close Window : Exit
  - Space               : Pause / Resume
  - R                   : Instant restart flight
"""

import time
import sys
import gymnasium as gym
import pygame
import torch
from quickstart_lander import RocketQNetwork, DEVICE

def watch(model_path="rocket_lander_model.pth"):
    print("=" * 60)
    print("  LAUNCHING INTERACTIVE FLIGHT VIEWER")
    print("  Controls: [ESC] = Exit | [SPACE] = Pause | [R] = Restart")
    print("=" * 60)

    # Initialize PyTorch Model
    model = RocketQNetwork(state_dim=8, action_dim=4).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded trained weights from '{model_path}'")
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Please train first!")
        return

    model.eval()

    # Create Gym Environment with Human Render Mode
    env = gym.make("LunarLander-v3", render_mode="human")
    clock = pygame.time.Clock()

    episode = 1
    running = True

    while running:
        state, _ = env.reset()
        done = False
        score = 0
        step = 0
        paused = False

        print(f"\n--> Flight #{episode} starting...")

        while not done and running:
            # Handle Pygame window events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif event.key == pygame.K_r:
                        done = True  # Restart this flight

            if paused:
                clock.tick(15)
                continue

            # Model decides action
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            step += 1

            # Cap frame rate at 50 FPS for smooth, realistic viewing
            clock.tick(50)

        if running:
            print(f"--> Flight #{episode} finished! Score: {score:.2f} (Steps: {step})")
            episode += 1
            # Pause 1.5 seconds so you can see the final resting position of the lander
            time.sleep(1.5)

    env.close()
    pygame.quit()
    print("Viewer closed.")

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "rocket_lander_model.pth"
    watch(model_file)
