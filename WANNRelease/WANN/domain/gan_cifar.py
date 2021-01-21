import tensorflow as tf
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
#import cv2
import math

class GANEnv(gym.Env):

  def __init__(self, discrimPath, latentDim):
    """
    discrimPath: string path to tensorflow model of discriminator.
    """

    self.t = 0          # Current batch number
    self.t_limit = 0    # Number of batches if you want to use them (we didn't)
    self.batch   = 100  # Number of images per batch
    self.seed()
    self.viewer = None
    self.inputDim = latentDim

    self.discriminator = tf.keras.models.load_model(discrimPath)

    print("SUccessfully loaded discriminator")

    self.action_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))
    self.observation_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))

    self.state = None

  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def reset(self):
    ''' Initialize State'''    
    #print('Lucky number', np.random.randint(10)) # same randomness?
    self.t = 0 # timestep
    self.state = np.random.normal(size=(self.batch, self.inputDim))
    return self.state
  
  def step(self, action):
    ''' 
    consider the WAN output and feed to the discrimator
    '''
    sigmoid = lambda z : 1 / (1 + np.exp(-z))
    action = sigmoid( np.reshape(action, (-1, 32,32,3)) )

    discriminator_judgment = self.discriminator.predict_on_batch(np.reshape(action, (-1, 32, 32, 3)))
    reward = np.mean(discriminator_judgment)

    if self.t_limit > 0: # We are doing batches
      reward *= (1/self.t_limit) # average
      self.t += 1
      done = False
      if self.t >= self.t_limit:
        done = True

      self.state = np.random.normal(size=(self.batch, self.inputDim))
    else:
      done = True

    obs = self.state
    return obs, reward, done, {}

 
