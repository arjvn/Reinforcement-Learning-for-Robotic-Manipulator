from gym.envs.registration import register

register(
    id='factory-v0',
    entry_point='gym_factory.envs:FactoryEnv',
)