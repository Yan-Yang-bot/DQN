import tensorflow as tf
from cv2 import cvtColor, resize, COLOR_BGR2GRAY
from tensorflow.layers import conv2d, Flatten, dense
from tensorflow.train import RMSPropOptimizer
import gym
import numpy as np
import argparse
import random
from collections import deque, namedtuple

Experience = namedtuple('Experience', 's, a, r, sp, done')
sess = tf.Session()

class ReplayBuffer:
    """ Class for Experience Replay"""
    def __init__(self, replay_mem_size=10000):
        """
        :param replay_mem_size: The maximum buffer size
        """
        self.mem = deque()
        self.replay_mem_size=replay_mem_size

    def add(self, experience):
        """
        Add a transition to the buffer. In case of buffer size overflow,
        remove the oldest transition in record.
        :param experience: the transition (s_t, a_t, r_{t+1}, s_{t+1}, done_flag)
        """
        self.mem.append(experience)
        if len(self.mem)>self.replay_mem_size:
            self.mem.popleft()

    def sample(self, batch_size=32):
        """
        Sample a batch of transitions (s_t, a_t, r_{t+1}, s_{t+1}, done_flag)
        where done_flag indicates whether s_t is a terminal state of an episode
        :param batch_size: Number of transitions in this batch
        :return: A batch of transitions uniformly sampled from the buffer
        """
        row_i = np.random.choice(len(self.mem), batch_size)
        return [self.mem[i] for i in row_i]

class Preprocess:
    """ Class for preprocessing image observations """
    def __init__(self, img_shape, m=4):
        """
        Patch the frame sequence with zero tensors at the beginning.
        :param m: # of most recent frames to be stacked together
        :param img_shape: the shape of each raw frame
        """
        self.history = deque()
        for _ in range(m):
            img = np.zeros(shape=img_shape, dtype=np.uint8)
            self.history.append(img)
        self.last_img=img

    def store(self, img):
        """
        Take the maximum value for each pixel colour value over the frame being encoded and the previous frame
        :param img: the raw frame to be stored
        """
        corrected_img = np.maximum(self.last_img, img)
        self.history.append(corrected_img)
        self.history.popleft()
        self.last_img=img

    def getinput(self):
        """
        Grayscale and resize the m most recent frames and stacks them
        :return: the input from preprocessing the most recent m raw frames
        """
        return np.stack( [resize(cvtColor(img, COLOR_BGR2GRAY), (84,84)) for img in self.history], axis=-1 )

    def storeGet(self, img):
        self.store(img)
        return self.getinput()

class QFunction:
    """ Class for Q function networks """
    def __init__(self, num_action, scope_name):
        """
        Define the same architecture for all Q-Networks, with initialization.
        :param num_action: # of valid actions
        :param scope_name: Define one scope name for each Q-Network (because in the same session)
        """
        with tf.variable_scope(scope_name) as self.scope:
            self.input = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.float32)
            h1 = conv2d(self.input, 32, 8, strides=(4,4), activation="relu")
            h2 = conv2d(h1, 64, 4, strides=(2,2), activation="relu")
            h3 = conv2d(h2, 64, 3, activation="relu")
            h4 = dense(Flatten()(h3), 512, activation="relu")
            self.output = dense(h4, num_action)
        sess.run(tf.initialize_variables(self.scope.trainable_variables()))

    def getVariables(self):
        """
        :return: The current values of all trainable variables in this scope
        """
        return [v.eval(session=sess) for v in self.scope.trainable_variables()]

    def update(self, yy, ss, aa):
        #TODO
        """

        :param yy:
        :param ss:
        :param aa:
        :return:
        """
        y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        a = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        loss = tf.reduce_sum( tf.square( tf.subtract(y, tf.gather_nd(self.output, a)) ) )
        rms = RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)
        opt = rms.minimize(loss)
        sess.run(tf.variables_initializer(rms.variables()))
        loss_value, _ = sess.run([loss, opt], feed_dict={self.input: ss, a: aa, y: yy})
        return loss_value

    def predict(self, input):
        """
        Use the current network state to evaluate the output of the Q-Function
        :param input: the preprocessed frame data
        :return: a tensor with one value for each valid action
        """
        return sess.run([self.output], feed_dict={self.input: np.reshape(input, (1,)+input.shape)})[0]

    def sync(self, values):
        """
        For the Target network to copy the whole set of variables from `qFunc.getVariables`
        :param values: Values got from qFunc.getVariables()
        """
        for (val, var) in zip(values, self.scope.trainable_variables()):
            tf.assign(var, val)

    def eval(self, env):
        returns = []
        for _ in range(15):
            ret = 0
            evalenv = gym.make(env)
            obs = evalenv.reset()
            done = False
            while not done:
                act = self.predict(obs)
                obs, r, done, _ = gym.step(act)
                ret += r
                print('.', end='')
            returns.append(ret)
            print('*', end='')
        print()
        return sum(returns)/5

def run(args):
    env = gym.make(args.env)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    num_action = env.action_space.n

    preprocess = Preprocess(env.observation_space.shape)
    buffer = ReplayBuffer()
    qFunc = QFunction(num_action, "q")
    tFunc = QFunction(num_action, "target")
    tFunc.sync(qFunc.getVariables())

    #TODO: your DQN
    #Step forward first, then update networks
    count = 0
    epsilon = 1.0
    exploration = 0
    ave_rets = []  # Stored every 1500 steps
    for epi in range(args.episodes):
        obs = env.reset()
        input = preprocess.storeGet(obs)
        while not done:
            if random.random()>epsilon:
                act = np.argmax(qFunc.predict(input))
            else:
                act = env.action_space.sample()
                exploration += 1
            newObs, reward, done, info = env.step(act)
            newInput = preprocess.storeGet(newObs)
            experience = Experience(input, act, reward, newInput, done)
            buffer.add(experience)
            count += 1
            if count>=500:
                input = newInput
                yy, ss, aa = [], [], []
                for experience in buffer.sample():
                    ss.append(experience.s)
                    aa.append([experience.a])
                    if experience.done:
                        yy.append([experience.r])
                    else:
                        yy.append([experience.r + args.gamma * np.max(tFunc.predict(experience.sp))])
                # TODO: describe loss graph & do the back propagation in qFunc
                # learning rate: 0.00025, gradient momentum: 0.95
                # squared gradient momentum: 0.95, min squared gradient: 0.01

                loss = qFunc.update(yy, ss, aa)
                if count%1500==0:
                    avereturn = qFunc.eval(args.env)
                    ave_rets.append(avereturn)
                    print("Episode {}, Count {}, Ave-Return {} ===> loss {}".format(epi, count, avereturn loss))
                else:
                    print("Episode {}, Count {} ===> loss {}".format(epi, count, loss))

                '''
                TODO: after learned something (param update)
                anneal epsilon toward 0.1 with total exploration frame number 1,000,000
                '''

                if epsilon>0.1:
                    epsilon -= 1.9e-6
                elif exploration<10000:
                    epsilon=0.1
                else:
                    epsilon=0

                if count%100==0 and count!=500:
                    tFunc.sync(qFunc.getVariables())

            else:
                print("step({}/500)".format(count), end='\r')

    np.asarray(ave_rets).dump('to_plot.npz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--episodes", default=15)
    parser.add_argument("--epsilon", default=1.0)
    parser.add_argument("--gamma", default=0.99)

    #TODO: add your own parameters

    args = parser.parse_args()
    run(args)
