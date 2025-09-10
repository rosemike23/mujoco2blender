import os
import sys
import argparse
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import msmodel_gym
import numpy as np

def main():

    env = gym.make("msmodel_gym/LocomotionEnv-v1", render_mode='human')
    
    for episode in range(3):
        terminated = truncated = False
        obs = env.reset()
        print(f"\nEpisode {episode + 1}/{3}")
        print("Action space shape:", env.action_space.shape)
        print("Observation space shape:", obs[0].shape)

        while not terminated and not truncated:
            # Use zero action for visualization
            action = env.action_space.sample() * 0.
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            print("Reward:", reward)

    env.close()

if __name__ == "__main__":
    main()
