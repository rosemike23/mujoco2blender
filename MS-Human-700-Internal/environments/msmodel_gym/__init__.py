from gymnasium.envs.registration import register

register(
    id="msmodel_gym/ManipulationEnv-v1",
    entry_point="msmodel_gym.envs:ManipulationEnvV1",
    max_episode_steps=200,
)

register(
    id="msmodel_gym/LocomotionEnv-v1",
    entry_point="msmodel_gym.envs:LocomotionEnvV1",
    max_episode_steps=3000,
)