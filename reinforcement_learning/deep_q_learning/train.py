#!/usr/bin/env python3
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
import gym


WINDOW_LENGTH = 4
ENV_NAME = 'ALE/Breakout-v5'
env = gym.make(ENV_NAME)
INPUT_SHAPE = env.observation_space.shape
nb_actions = env.action_space.n

input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE    
input = Input(input_shape)
x = Permute((2, 3, 1))(input)
x = Convolution2D(32, (8, 8), strides=(4, 4))(x)
x = Activation('relu')(x)
x = Convolution2D(32, (8, 8), strides=(4, 4))(x)
x = Activation('relu')(x)
x = Convolution2D(32, (8, 8), strides=(4, 4))(x)
x = Activation('relu')(x)
x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dense(nb_actions)(x)
x = Activation('linear')(x)
model = Model(inputs=input, outputs=x)

memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, policy=policy, nb_actions=nb_actions, memory=memory, target_model_update=10000, nb_steps_warmup=1000)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

dqn.fit(env, nb_steps=10000)

dqn.save_weights('policy.h5', overwrite=True)
