import tensorflow as tf
import numpy as np
import numpy.linalg.norm as norm

from network import QNetwork
from action import actionlst
from ..feature_extractor.feature_extractor import ContextExtractor, get_histogram 

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
SEED = 6885
ACTION_SIZE = 12
STATE_LENGTH = 5000

# Calculate reward for consecutive timesteps
# based on the features of images
def reward(prev, cur, target):
    l2prev = norm(target - prev)
    l2cur = norm(target - cur)
    return l2prev - l2cur


class Agent:
    def __init__(self):
        # local network for estimate
        # target network for computing target
        self.network_loc = QNetwork()
        self.network_targ = QNetwork()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LR)

        self.memory = ReplayBuffer(ACTION_SIZE, BUFFER_SIZE, BATCH_SIZE, SEED)
        self.t_step = 0

        self.actions = actionlst()

        self.context_extractor = ContextExtractor()

        # Build model so it knows the input shape
        self.network_loc.build(tf.TensorShape([None, STATE_LENGTH,]))
        self.network_targ.build(tf.TensorShape([None, STATE_LENGTH,]))

    def step(self, img, target):
        # Given input image and target image as numpy array
        # Return (s,a,s',r) and the img after action
        
        # Extract features
        ctx_cur = self.context_extractor(img / 255.0)
        color_cur = get_histogram(img)

        # TODO: 1. feed to local network and get action
        #       2. apply action to prev img and extract features
        #       3. calculate reward


    def record(self):
        # Save (s,a,s',r) to replay buffer

    def learn(self):
        pass

    def soft_update(self):
        pass

class ReplayBuffer:
    def __init__(self):
        pass

    def add(self):
        pass

    def sample(self):
        pass

    def __len__(self):
        pass