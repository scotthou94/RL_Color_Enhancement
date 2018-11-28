import tensorflow as tf
import numpy as np
import numpy.linalg as LA

from network import QNetwork
from action import actionlst
from feature_extractor.feature_extractor import ContextExtractor, get_histogram 

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
SEED = 6885
ACTION_SIZE = 12
VGG_SHAPE = 224
VGG_OUTPUT = 4096
COLOR_SHAPE = 96
CAPACITY = 500
STATE_LENGTH = VGG_OUTPUT + COLOR_SHAPE


# Calculate reward for consecutive timesteps based on the features of images
def reward(prev, cur, target):
    l2prev = LA.norm(target - prev)
    l2cur = LA.norm(target - cur)
    return l2prev - l2cur

# Combine color and context, return state feature
def combine(color, context):
    color = color.flatten()
    state = np.concatenate((color, context))
    assert state.shape == (VGG_OUTPUT + COLOR_SHAPE,)
    return state

# Apply action and return result image
def applyChange(actions, choice, img):
    assert 'int' in str(img.dtype)
    return actions[choice](img)

class Agent:
    def __init__(self):
        # local network for estimate
        # target network for computing target
        self.network_loc = QNetwork()
        self.network_targ = QNetwork()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LR)

        self.buffer = ReplayBuffer()
        self.t_step = 0

        self.actions = actionlst()

        self.context_extractor = ContextExtractor()

        # Build model so it knows the input shape
        self.network_loc.build(tf.TensorShape([None, STATE_LENGTH,]))
        self.network_targ.build(tf.TensorShape([None, STATE_LENGTH,]))

    def getState(self, img):
        img_resize = tf.image.resize_images(img, [VGG_SHAPE, VGG_SHAPE]) / 255.0
        ctx = self.context_extractor(img_resize.numpy())
        color = get_histogram(img)
        return combine(color, ctx)

    def step(self, img_prev, target):
        # Given input image and target image as numpy array
        # Return (s,a,s',r) and the img after action
        
        # 1. Extract features
        state_prev = self.getState(img_prev)

        # 2. Feed to local network and get action
        # Add batch dimension to state
        state_prev = np.expand_dims(state_prev, 0)
        state_prev = state_prev.astype(np.float32)
        predicts = self.network_loc(state_prev)
        action = np.argmax(predicts)

        # 3. Apply action and get img_cur, state_cur, state_target
        img_cur = applyChange(self.actions, action, img_prev)
        state_cur = self.getState(img_cur)
        state_target = self.getState(target)

        # 4. Calculate reward
        r = reward(state_prev, state_cur, state_target)

        # Return (s,a,s',r) tuple
        return (state_prev, action, state_cur, r)

    def record(self, state_prev, action, state_cur, reward):
        # Save (s,a,s',r) to replay buffer
        return self.buffer.add(state_prev, action, state_cur, reward)

    def learn(self):
        pass

    def soft_update(self):
        pass

class ReplayBuffer:
    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self, state_prev, action, state_cur, reward):
        if len(self.memory) == self.buffer_size:
            return False
        else:
            experience = [state_prev, action, state_cur, reward]
            self.memory.append(experience)
            return True

    def sample(self):
        # Sample a batch of experiences from buffer
        if len(self.memory) < BATCH_SIZE:
            return False
        else:
            return np.random.choice(self.memory, BATCH_SIZE, replace=False)

    def clear(self):
        self.memory.clear()











