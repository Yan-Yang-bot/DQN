import numpy as np
import gym, sys, argparse, random
from collections import namedtuple

from replaybuffer import ReplayBuffer
from neuralnetwork import NeuralNetwork
from preprocess import Preprocess

Experience = namedtuple('Experience', 's, a, r, sp, done')

def run(args):

    # Initializations
    env = gym.make(args.env)
    evalEnv = gym.make(args.env)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    num_action = env.action_space.n

    buffer = ReplayBuffer()
    qFunc = NeuralNetwork(num_action, "q")
    #print(qFunc.getVariables()[0][1])
    # print(qFunc.eval(evalEnv)) #TODO: delete
    tFunc = NeuralNetwork(num_action, "target")
    qFunc.initLossGraph()
    tFunc.sync(qFunc.getVariables())

    #your DQN
    #Step forward first, then update networks
    epsilon = 1.0
    exploration = 0

    count = 0
    epi = 0
    global_steps = int(args.steps)
    noop_max = 30
    exploration_max = 10000
    target_rate = 500
    q_rate = 4

    ave_rets = []  # Stored an average return every 1500 steps
    rets = []  # Store 5 returns for calculating the average

    while count < global_steps:
        # Settings for starting a new episode
        obs = env.reset()
        preprocess = Preprocess(env.observation_space.shape)
        processedState = preprocess.storeGet(obs)
        done = False
        t = 0
        noop = True

        return_ = 0

        while not done and count < global_steps:
            # epsilon-greedy action selection
            if random.random() > epsilon:
                act = np.argmax(qFunc.predict(processedState))
            else:
                act = env.action_space.sample()
                exploration += 1

            # no-op at the beginning can't exceed noop_max
            if noop:
                if act == 0:
                    if t >= noop_max:
                        act = 1
                else:
                    noop = False

            # step forward, get a transition, and store it in the buffer
            newObs, reward, done, info = env.step(act)
            return_ += reward
            newProcessedState = preprocess.storeGet(newObs)
            experience = Experience(processedState, act, reward, newProcessedState, done)
            buffer.add(experience)

            # update countings
            count += 1
            t += 1

            # for the first 500 steps, skip learning, and do random actions
            # without need to care about what state it is in
            if count < 5000:
                sys.stdout.write("\r")
                sys.stdout.write("step({}/5000)".format(count))

            # after the first 500 steps
            else:
                # select four actions between two updates
                if count % q_rate == 0:
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

                    # after learned something (param update)
                    # anneal epsilon toward 0.1 with total exploration frame number 1,000,000
                    # no longer use randomly sampled actions
                    # update state representation and epsilon

                    if epsilon > 0.1:
                        epsilon -= 1.9e-7
                    elif exploration < exploration_max:
                        epsilon = 0.1
                    else:
                        epsilon = 0

                # logging with average return calculation every 1000 steps, normal logging every 10 steps
                '''
                if count % 1000 == 0:
                    avereturn = qFunc.eval(evalEnv)
                    ave_rets.append(avereturn)
                    print("Episode {}, step {}, Count {}, Ave-Return {} ===> loss {}".format(epi, t, count, avereturn, loss))
                '''
                if count % 10==0:
                    print("Episode {}, step {}, Count {}, Reward {}  ===> loss {}".format(epi, t, count, reward, loss))


                # Sync q network params to the target function every 1000 steps

                if count % target_rate == 0 and count != 5000:
                    tFunc.sync(qFunc.getVariables())

        rets.append(return_)
        if epi==0:
            rets.extend([return_] * 4)
        else:
            rets = rets[1:]

        ave_rets.append(sum(rets)/5)

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
