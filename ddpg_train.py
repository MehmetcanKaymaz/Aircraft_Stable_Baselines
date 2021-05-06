import gym
import numpy as np
from environment_gym import Env_Gym

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import FeedForwardPolicy

class CustomDDPGPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs,
                                           layers=[64, 64, 64],
                                           layer_norm=False,
                                           feature_extraction="mlp")
                                           
class CustomDDPGPolicy_2(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy_2, self).__init__(*args, **kwargs,
                                           layers=[128, 128, 128],
                                           layer_norm=False,
                                           feature_extraction="mlp")                                           

env = Env_Gym()

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
action_noise_2 = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(1) * np.ones(n_actions))

model = DDPG(CustomDDPGPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, gamma=0.995, batch_size=256, actor_lr=0.00001, critic_lr=0.00002,observation_range=(-30, 30),normalize_observations=True)

"""
model_2 = DDPG(CustomDDPGPolicy_2, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model_3 = DDPG(CustomDDPGPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise_2)
model_4 = DDPG(CustomDDPGPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
"""


model.learn(total_timesteps=10000)

"""
model_2.learn(total_timesteps=100000)
model_3.learn(total_timesteps=100000)
model_4.learn(total_timesteps=200000)
"""

model.save("ddpg_uav_2")

"""
model_2.save("ddpg_copter_2")
model_2.save("ddpg_copter_3")
model_4.save("ddpg_copter_4")
"""

del model # remove to demonstrate saving and loading

"""
del model_2
del model_3
del model_4

"""
print("*************************************\n        Model 1 Result        \n*************************************")

model = DDPG.load("ddpg_uav_2")


obs = env.reset()
while True:
    
    action, _states = model.predict(obs)
    print("Action:{}".format(action))
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    env.render()
"""    
print("*************************************\n        Model 2 Result        \n*************************************")

model_2 = DDPG.load("ddpg_copter_2")


obs = env.reset()
while True:
    
    action, _states = model_2.predict(obs)
    print("Action:{}".format(action))
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    env.render()
        
print("*************************************\n        Model 3 Result        \n*************************************")

model_3 = DDPG.load("ddpg_copter_3")


obs = env.reset()
while True:
    
    action, _states = model_3.predict(obs)
    print("Action:{}".format(action))
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    env.render()
        
print("*************************************\n        Model 4 Result        \n*************************************")

model_4 = DDPG.load("ddpg_copter_4")


obs = env.reset()
while True:
    
    action, _states = model_4.predict(obs)
    print("Action:{}".format(action))
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    env.render()
"""        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
