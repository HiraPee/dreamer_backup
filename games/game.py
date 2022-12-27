import gym
env = gym.make('SpaceInvaders-v0')
observation = env.reset()
for t in range(1000):
    #env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    print(reward)
env.close()
