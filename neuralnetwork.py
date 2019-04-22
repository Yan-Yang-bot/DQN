import tensorflow as tf
import numpy as np
from tensorflow.layers import conv2d, Flatten, dense
from tensorflow.train import RMSPropOptimizer
import sys

from preprocess import Preprocess

class NeuralNetwork:
    """ Class for Q/Target function networks """

    sess = tf.Session()  # A universal session handler for all instances of this Class

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
        self.sess.run(tf.variables_initializer(self.scope.trainable_variables()))

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
        self.sess.run(tf.variables_initializer(rms.variables()))

    def getVariables(self):
        """
        :return: The current values of trainable variables in `self.scope`
        """
        return [v.eval(session=self.sess) for v in self.scope.trainable_variables()]

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
        :param yy: A list of target values of the sampled transitions
        :param ss: A list of preprocessed input statuses of the sampled transitions
        :param aa: A list of actions chosen in the sampled transitions
        :return: the loss value at this parameter update
        """
        loss_value, _ = self.sess.run([self.loss, self.opt], feed_dict={self.input: ss, self.a: aa, self.y: yy})
        return loss_value

    def predict(self, input):
        """
        Use the current network state to evaluate the output of the Q-Function
        :param input: the preprocessed frame data
        :return: a tensor with one value for each valid action
        """
        return self.sess.run([self.output], feed_dict={self.input: np.reshape(input, (1,)+input.shape)})[0]

    def eval(self, env):
        """
        Evaluate the current prediction ability by average return.
        :param env: the environment to be evaluated on
        :return: average return
        """
        #print(self.getVariables()[0][1])
        N = 5
        returns = []
        for _ in range(N):
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
            sys.stdout.write('*{},{}\n'.format(t,ret))

        return sum(returns)/N
