import os
import sys
import argparse
import gymnasium as gym

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import msmodel_gym
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Test MS-Human-700 environments')
    parser.add_argument('-loco', '--locomotion', action='store_true',
                      help='Test locomotion environment')
    parser.add_argument('-mani', '--manipulation', action='store_true',
                      help='Test manipulation environment')
    parser.add_argument('--render_mode', type=str, default='human',
                      choices=['human', 'rgb_array'],
                      help='Rendering mode. Use "rgb_array" for headless servers')
    parser.add_argument('--episodes', type=int, default=3,
                      help='Number of episodes to run')
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = os.path.abspath(__file__)
    model_path = os.path.dirname(model_path)

    # Select environment based on arguments
    if args.manipulation:
        env_id = "msmodel_gym/ManipulationEnv-v1"
    elif args.locomotion:
        env_id = "msmodel_gym/LocomotionEnv-v1"
    else:
        print("Please specify an environment using -loco or -mani")
        return

    # Create environment with specified render mode
    env = gym.make(env_id, render_mode=args.render_mode)
    print(f"\nTesting {env_id} environment with {args.render_mode} rendering mode")

    for episode in range(args.episodes):
        terminated = truncated = False
        obs = env.reset()
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        print("Action space shape:", env.action_space.shape)
        print("Observation space shape:", obs[0].shape)

        while not terminated and not truncated:
            # Use zero action for visualization
            action = env.action_space.sample() * 0.
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Only call render if mode is 'human' or we want to capture frames
            if args.render_mode in ['human']:
                env.render()
            
            print("Reward:", reward)

    env.close()

if __name__ == "__main__":
    main()
