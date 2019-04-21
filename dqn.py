import tensorflow as tf
from cv2 import cvtColor, resize, COLOR_BGR2GRAY
from tensorflow.layers import conv2d, Flatten, dense
from tensorflow.train import RMSPropOptimizer
import gym, sys
import numpy as np
import argparse
import random
from collections import deque, namedtuple

Experience = namedtuple('Experience', 's, a, r, sp, done')
sess = tf.Session()

class ReplayBuffer:
    """ Class for Experience Replay """
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

class NeuralNetwork:
    """ Class for Q/Target function networks """
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
        sess.run(tf.variables_initializer(self.scope.trainable_variables()))

    def initLossGraph(self):
        """
        Describe and initialize the loss graph
        (for Q Function only, not needed for Target Function)
        """
        self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.a = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        #print(self.output, a)
        #output_var = tf.get_variable("output", dtype=tf.float32, initializer=self.output, trainable=False, validate_shape=False)
        #print(output_var)
        self.loss = tf.reduce_sum( tf.square( tf.subtract(self.y, tf.gather_nd(self.output, self.a)) ) )
        rms = RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)
        self.opt = rms.minimize(self.loss)
        sess.run(tf.variables_initializer(rms.variables()))

    def getVariables(self):
        """
        :return: The current values of trainable variables in `self.scope`
        """
        return [v.eval(session=sess) for v in self.scope.trainable_variables()]

    def sync(self, values):
        """
        For the Target network to copy the set of network variables from `qFunc.getVariables`
        :param values: Values got from qFunc.getVariables()
        """
        for (val, var) in zip(values, self.scope.trainable_variables()):
            tf.assign(var, val)

    def update(self, yy, ss, aa):
        """
        Back propagation in qFunc
        :param yy: A list of 32 target values of the sampled transitions
        :param ss: A list of 32 preprocessed input statuses of the sampled transitions
        :param aa: A list of 32 actions chosen in the sampled transitions
        :return: the loss value at this parameter update
        """
        loss_value, _ = sess.run([self.loss, self.opt], feed_dict={self.input: ss, self.a: aa, self.y: yy})
        return loss_value

    def predict(self, input):
        """
        Use the current network state to evaluate the output of the Q-Function
        :param input: the preprocessed frame data
        :return: a tensor with one value for each valid action
        """
        return sess.run([self.output], feed_dict={self.input: np.reshape(input, (1,)+input.shape)})[0]

    def eval(self, env):
        """
        Evaluate the current prediction ability by average return.
        :param env: the environment to be evaluated on
        :return: average return
        """
        returns = []
        for _ in range(5):
            ret = 0
            prep = Preprocess(env.observation_space.shape)
            obs = env.reset()
            done = False
            t = 0
            while not done:
                act = np.argmax(self.predict(prep.storeGet(obs)))
                obs, r, done, _ = env.step(act)
                ret += r
                if t%10==0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                t += 1
            del prep
            returns.append(ret)
            sys.stdout.write('*\n')

        return sum(returns)/5


def run(args):

    # Initializations
    env = gym.make(args.env)
    evalEnv = gym.make(args.env)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    num_action = env.action_space.n

    buffer = ReplayBuffer()
    qFunc = NeuralNetwork(num_action, "q")
    print(qFunc.eval(evalEnv)) #TODO: delete
    tFunc = NeuralNetwork(num_action, "target")
    qFunc.initLossGraph()
    tFunc.sync(qFunc.getVariables())

    #your DQN
    #Step forward first, then update networks
    epsilon = 1.0
    exploration = 0

    count = 0
    epi = 0
    n_steps = int(args.steps)

    ave_rets = []  # Stored an average return every 1500 steps

    while count < n_steps:
        # Settings for starting a new episode
        obs = env.reset()
        preprocess = Preprocess(env.observation_space.shape)
        processedState = preprocess.storeGet(obs)
        done = False
        t=0

        while not done and count < n_steps:
            # epsilon-greedy action selection
            if random.random()>epsilon:
                act = np.argmax(qFunc.predict(processedState))
            else:
                act = env.action_space.sample()
                exploration += 1
            # step forward, get a transition, and store it in the buffer
            newObs, reward, done, info = env.step(act)
            newProcessedState = preprocess.storeGet(newObs)
            experience = Experience(processedState, act, reward, newProcessedState, done)
            buffer.add(experience)

            # update countings
            count += 1
            t += 1

            # for the first 500 steps, skip learning, and do random actions
            # without need to care about what state it is in
            if count<500:
                sys.stdout.write("\r")
                sys.stdout.write("step({}/500)".format(count))

            # after the first 500 steps
            else:
                # sample a batch of transitions from buffer each time and calculate the target
                yy, ss, aa = [], [], []
                for experience in buffer.sample():
                    ss.append(experience.s)
                    aa.append([experience.a])
                    if experience.done:
                        yy.append([experience.r])
                    else:
                        yy.append([experience.r + args.gamma * np.max(tFunc.predict(experience.sp))])

                # use the target-status-action triple of each item in the batch to
                # find the current gradient and optimize the q network
                loss = qFunc.update(yy, ss, aa)

                # logging with average return calculation every 1500 steps
                if count%1500==0:
                    avereturn = qFunc.eval(evalEnv)
                    ave_rets.append(avereturn)
                    print("Episode {}, step {}, Count {}, Ave-Return {} ===> loss {}".format(epi, t, count, avereturn, loss))
                else:
                    print("Episode {}, step {}, Count {}, Reward {}  ===> loss {}".format(epi, t, count, reward, loss))


                # after learned something (param update)
                # anneal epsilon toward 0.1 with total exploration frame number 1,000,000
                # no longer use randomly sampled actions
                # update state representation and epsilon

                processedState = newProcessedState

                if epsilon>0.1:
                    epsilon -= 1.9e-6
                elif exploration<10000:
                    epsilon=0.1
                else:
                    epsilon=0

                # Sync q network params to the target function every 100 steps

                if count%100==0 and count!=500:
                    tFunc.sync(qFunc.getVariables())



        epi += 1
        del preprocess

    np.asarray(ave_rets).dump('to_plot.npz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--steps", default=30000)
    #parser.add_argument("--epsilon", default=1.0)
    parser.add_argument("--gamma", default=0.99)


    args = parser.parse_args()
    run(args)
