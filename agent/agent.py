from collections import deque

import tensorflow as tf
import numpy as np
import numpy.linalg as LA

from network import QNetwork
from action import actionlst
from feature_extractor.feature_extractor import ContextExtractor, get_histogram 

BUFFER_SIZE = int(1e3)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
VGG_SHAPE = 224
VGG_OUTPUT = 4096
COLOR_SHAPE = 96
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

# Set model2 weights to model1 weights
# I couldn't get seed to work so this is ugly
def setWeights(model1, model2):
    assert len(model1.layers) == len(model2.layers)
    for i in range(len(model1.layers)):
        model2.layers[i].set_weights(model1.layers[i].get_weights())

class Agent:
    def __init__(self):
        # local network for estimate
        # target network for computing target
        self.network_loc = QNetwork()
        self.network_targ = QNetwork()
        setWeights(self.network_loc, self.network_targ)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LR)

        self.buffer = ReplayBuffer()
        self.t_step = 0

        self.actions = actionlst()

        self.context_extractor = ContextExtractor()

        # Build model so it knows the input shape
        self.network_loc.build(tf.TensorShape([None, STATE_LENGTH,]))
        self.network_targ.build(tf.TensorShape([None, STATE_LENGTH,]))

    def __getState(self, img):
        # Private method to get state from an numpy image
        img_resize = tf.image.resize_images(img, [VGG_SHAPE, VGG_SHAPE]) / 255.0
        ctx = self.context_extractor(img_resize.numpy())
        color = get_histogram(img)
        return combine(color, ctx)

    def step(self, img_prev, target):
        # Given input image and target image as numpy array
        # Return (s,a,s',r) and the img after action
        
        # 1. Extract features
        state_prev = self.__getState(img_prev)

        # 2. Feed to local network and get action
        # Add batch dimension to state
        state_prev = np.expand_dims(state_prev, 0)
        state_prev = state_prev.astype(np.float32)
        predicts = self.network_loc(state_prev)
        action = np.argmax(predicts)
        state_prev = np.squeeze(state_prev, 0)

        # 3. Apply action and get img_cur, state_cur, state_target
        img_cur = applyChange(self.actions, action, img_prev)
        state_cur = self.__getState(img_cur)
        state_target = self.__getState(target)

        # 4. Calculate reward
        r = reward(state_prev, state_cur, state_target)

        # Return (s,a,s',r) tuple and img_cur
        return (state_prev, action, state_cur, r), img_cur

    def record(self, state_prev, action, state_cur, reward):
        # Save (s,a,s',r) to replay buffer
        self.buffer.add(state_prev, action, state_cur, reward)

    def learn(self):
        # 1. Sample batch from replay buffer
        # Only sample whole batch
        batch = self.buffer.sample()
        if not batch:
            return False
        # Unpack experiences
        rows = np.arange(batch.shape[0])
        state_ps = batch[rows, 0]
        actions = batch[rows, 1]
        state_cs = batch[rows, 2]
        rs = batch[rows, 3]

        # 2. Compute loss based on q_target and q_estimate
        with tf.GradientTape() as tape:
            q_est = self.network_loc(state_ps)[rows, actions]
            q_targ = self.network_targ(state_cs).max(axis=1)
            target = rs + GAMMA * q_targ
            loss = tf.losses.mean_squared_error(target, q_est)

        # 3. Back prop
        grads = tape.gradient(loss, self.network_loc.variables)
        self.optimizer.apply_gradients(zip(grads, self.network_loc.variables))

        # 4. Soft updates
        self.__soft_update()
        return True

    def __soft_update(self):
        # Slowly update the target network
        # Iterate through all layers and set weights
        for layer_t, layer_loc in zip(self.network_targ, self.network_loc):
            target = layer_t.get_weights()
            loc = layer_loc.get_weights()
            for i in range(len(target)):
                target[i] = (1 - TAU) * target[i] + TAU * loc[i]
            layer_t.set_weights(target)

class ReplayBuffer:
    def __init__(self, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.batch_size = batch_size

    def add(self, state_prev, action, state_cur, reward):
        experience = [state_prev, action, state_cur, reward]
        self.memory.append(experience)

    def sample(self):
        # Sample a batch of experiences from buffer
        if len(self.memory) < BATCH_SIZE:
            return False
        else:
            choices = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
            return np.array(self.memory)[choices]

    def __len__(self):
        return len(self.memory)
