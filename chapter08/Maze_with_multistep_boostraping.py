#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import itertools
import heapq

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def addItem(self, item, priority=0):
        if item in self.entry_finder:
            self.removeItem(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def removeItem(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def popTask(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder

# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
class Maze:

    def __init__(self):
        # maze width
        self.WORLD_WIDTH = 9

        # maze height
        self.WORLD_HEIGHT = 6

        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [2, 0]

        # goal state
        self.GOAL_STATES = [[0, 8]]

        # all obstacles
        self.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.oldObstacles = None
        self.newObstacles = None

        # time to change obstacles
        self.changingPoint = None

        # initial state action pair values
        self.stateActionValues = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))

        # max steps
        self.maxSteps = float('inf')
        #self.maxSteps = 5000

        # track the resolution for this maze
        self.resolution = 1

    # extend a state to a higher resolution maze
    # @state: state in lower resolution maze
    # @factor: extension factor, one state will become factor^2 states after extension
    def extendState(self, state, factor):
        newState = [state[0] * factor, state[1] * factor]
        newStates = []
        for i in range(0, factor):
            for j in range(0, factor):
                newStates.append([newState[0] + i, newState[1] + j])
        return newStates

    # extend a state into higher resolution
    # one state in original maze will become @factor^2 states in @return new maze
    def extendMaze(self, factor):
        newMaze = Maze()
        newMaze.WORLD_WIDTH = self.WORLD_WIDTH * factor
        newMaze.WORLD_HEIGHT = self.WORLD_HEIGHT * factor
        newMaze.START_STATE = [self.START_STATE[0] * factor, self.START_STATE[1] * factor]
        newMaze.GOAL_STATES = self.extendState(self.GOAL_STATES[0], factor)
        newMaze.obstacles = []
        for state in self.obstacles:
            newMaze.obstacles.extend(self.extendState(state, factor))
        newMaze.stateActionValues = np.zeros((newMaze.WORLD_HEIGHT, newMaze.WORLD_WIDTH, len(newMaze.actions)))
        newMaze.resolution = factor
        return newMaze

    # take @action in @state
    # @return: [new state, reward]
    def takeAction(self, state, action):
        x, y = state
        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        if [x, y] in self.obstacles:
            x, y = state
        if [x, y] in self.GOAL_STATES:
            reward = 1.0
        else:
            reward = 0.0
        return [x, y], reward

# a wrapper class for parameters of dyna algorithms
class DynaParams:

    def __init__(self):
        # discount
        self.gamma = 0.95

        # probability for exploration
        self.epsilon = 0.01

        # step size
        self.alpha = 0.1

        # weight for elapsed time
        self.timeWeight = 0

        # n-step planning
        self.planningSteps = 5

        # average over several independent runs
        self.runs = 10

        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']

        # threshold for priority queue
        self.theta = 0


# choose an action based on epsilon-greedy algorithm
def chooseAction_old(state, stateActionValues, maze, dynaParams):
    if np.random.binomial(1, dynaParams.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        # when selecting greedily among actions, ties were broken randomly
        winner = np.argwhere(stateActionValues[state[0], state[1], :] == np.amax(stateActionValues[state[0], state[1], :]))
        if winner.shape[0] > 1:
            return np.random.choice(list(np.squeeze(winner)))
        else:
            return np.argmax(stateActionValues[state[0], state[1], :])

# choose an action based on epsilon-greedy algorithm
def chooseAction(state, stateActionValues, maze, dynaParams):
    if np.random.binomial(1, dynaParams.epsilon) == 1:
        return np.random.choice(maze.actions)
    else:
        # return np.argmax(stateActionValues[state[0], state[1], :])
        return np.random.choice(np.flatnonzero(
            stateActionValues[state[0], state[1], :] == np.max(stateActionValues[state[0], state[1], :])))

# Trivial model for planning in Dyna-Q
class TrivialModel:

    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, rand=np.random):
        self.model = dict()
        self.rand = rand

    # feed the model with previous experience
    def feed(self, currentState, action, newState, reward):
        if tuple(currentState) not in self.model.keys():
            self.model[tuple(currentState)] = dict()
        self.model[tuple(currentState)][action] = [list(newState), reward]

    # randomly sample from previous experience
    def sample(self):
        stateIndex = self.rand.choice(range(0, len(self.model.keys())))
        state = list(self.model)[stateIndex]
        actionIndex = self.rand.choice(range(0, len(self.model[state].keys())))
        action = list(self.model[state])[actionIndex]
        newState, reward = self.model[state][action]
        return list(state), action, list(newState), reward

# Time-based model for planning in Dyna-Q+
class TimeModel:

    # @maze: the maze instance. Indeed it's not very reasonable to give access to maze to the model.
    # @timeWeight: also called kappa, the weight for elapsed time in sampling reward, it need to be small
    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self, maze, timeWeight=1e-4, rand=np.random):
        self.rand = rand
        self.model = dict()

        # track the total time
        self.time = 0

        self.timeWeight = timeWeight
        self.maze = maze

    # feed the model with previous experience
    def feed(self, currentState, action, newState, reward):
        self.time += 1
        if tuple(currentState) not in self.model.keys():
            self.model[tuple(currentState)] = dict()

            # Actions that had never been tried before from a state were allowed to be considered in the planning step
            for action_ in self.maze.actions:
                if action_ != action:
                    # Such actions would lead back to the same state with a reward of zero
                    # Notice that the minimum time stamp is 1 instead of 0
                    self.model[tuple(currentState)][action_] = [list(currentState), 0, 1]

        self.model[tuple(currentState)][action] = [list(newState), reward, self.time]

    # randomly sample from previous experience
    def sample(self):
        stateIndex = self.rand.choice(range(0, len(self.model.keys())))
        state = list(self.model)[stateIndex]
        actionIndex = self.rand.choice(range(0, len(self.model[state].keys())))
        action = list(self.model[state])[actionIndex]
        newState, reward, time = self.model[state][action]

        # adjust reward with elapsed time since last vist
        reward += self.timeWeight * np.sqrt(self.time - time)

        return list(state), action, list(newState), reward


# play for an episode for Dyna-Q algorithm
# @stateActionValues: state action pair values, will be updated
# @model: model instance for planning
# @planningSteps: steps for planning
# @maze: a maze instance containing all information about the environment
# @dynaParams: several params for the algorithm
def dynaQ(stateActionValues, model, maze, dynaParams):
    currentState = maze.START_STATE
    steps = 0
    while currentState not in maze.GOAL_STATES:
        # track the steps
        steps += 1
        # get action
        action = chooseAction(currentState, stateActionValues, maze, dynaParams)
        # take action
        newState, reward = maze.takeAction(currentState, action)


        # Q-Learning update
        stateActionValues[currentState[0], currentState[1], action] += \
            dynaParams.alpha * (reward + dynaParams.gamma * np.max(stateActionValues[newState[0], newState[1], :]) -
            stateActionValues[currentState[0], currentState[1], action])

        # feed the model with experience
        model.feed(currentState, action, newState, reward)

        # sample experience from the model
        for t in range(0, dynaParams.planningSteps):
            stateSample, actionSample, newStateSample, rewardSample = model.sample()
            stateActionValues[stateSample[0], stateSample[1], actionSample] += \
                dynaParams.alpha * (rewardSample + dynaParams.gamma * np.max(stateActionValues[newStateSample[0], newStateSample[1], :]) -
                stateActionValues[stateSample[0], stateSample[1], actionSample])

        currentState = newState

        # check whether it has exceeded the step limit
        if steps > maze.maxSteps:
            print(currentState)
            break

    return steps


def temporalDifference(stateActionValues, n, maze, dynaParams):
    # N_step SARSA algorithm for esimtating Q Page 157
    currentState = maze.START_STATE
    # track the time
    t = 0
    # arrays to store states and rewards for an

    currentState_action = currentState.copy()
    action = chooseAction(currentState, stateActionValues, maze, dynaParams)
    currentState_action.append(action)
    states_actions = [tuple(currentState_action)]
    rewards = [0.0]

    # the length of this episode
    T = float('inf')
    while True:
        # track the steps
        if t+1 < T:
            # store new state and new rewards
            newState, reward = maze.takeAction(currentState, action)
            rewards.append(reward)

            if newState in maze.GOAL_STATES:
                T = t+1
            else:
                action = chooseAction(newState, stateActionValues, maze, dynaParams)
                newState_action = newState.copy()
                newState_action.append(action)
                states_actions.append(tuple(newState_action))
                # get the time of the state to update
        tao = t - n + 1
        if tao >= 0:
            returns = 0.0
            # calculate the corresponding rewards
            for i in range(tao+1, min(T, tao + n)+1):
                returns += pow(dynaParams.gamma, i - tao - 1) * rewards[i]
            # add state-action value to the return
            if tao + n < T:
                returns += pow(dynaParams.gamma, n) * stateActionValues[tuple(states_actions[tao + n])]
            state_action_ToUpdate = states_actions[tao]
            # update the state action value
            if not state_action_ToUpdate[:2] in maze.GOAL_STATES:
                stateActionValues[tuple(state_action_ToUpdate)] += dynaParams.alpha * (returns - stateActionValues[tuple(state_action_ToUpdate)])
        if tao == T - 1:
            break

        t += 1
        currentState = newState

        # check whether it has exceeded the step limit
        if t > maze.maxSteps:
            print(currentState)
            break

    return T


def temporalDifference_old(stateActionValues, n, maze, dynaParams):
    # N_step SARSA algorithm for esimtating Q Page 157
    currentState = maze.START_STATE
    # track the time
    t = 0
    # arrays to store states and rewards for an
    action = chooseAction(currentState, stateActionValues, maze, dynaParams)
    states = [currentState]
    actions = [action]
    rewards = [0.0]

    # the length of this episode
    T = float('inf')
    while True:
        # track the steps
        #TODO ??? t+1 < T?
        #if t < T:
        if t+1 < T:
            # store new state and new rewards
            newState, reward = maze.takeAction(currentState, action)

            # store new state and new reward
            states.append(newState)
            rewards.append(reward)

            if newState in maze.GOAL_STATES:
                T = t + 1
            else:
                action = chooseAction(newState, stateActionValues, maze, dynaParams)
                actions.append(action)

        tao = t - n + 1
        if tao >= 0:
            returns = 0.0
            # calculate the corresponding rewards
            for i in range(tao + 1, min(T, tao + n) + 1):
                returns += pow(dynaParams.gamma, i - tao - 1) * rewards[i]
            # add state-action value to the return
            # TODO: tao +n <= T
            if tao + n < T:
                returns += pow(dynaParams.gamma, n) * \
                           stateActionValues[states[tao+n][0], states[tao+n][1], actions[tao+n]]
            stateToUpdate = states[tao]
            # update the state action value
            if not stateToUpdate in maze.GOAL_STATES:
                stateActionValues[stateToUpdate[0], stateToUpdate[1], actions[tao]] += dynaParams.alpha * (
                    returns - stateActionValues[stateToUpdate[0], stateToUpdate[1], actions[tao]])
        if tao == T - 1:
            break

        t += 1
        currentState = newState

        # check whether it has exceeded the step limit
        if t > maze.maxSteps:
            print(currentState)
            break

    return t


def temporalDifference_wzn(stateActionValues, n,  maze, dynaParams):

    # initial starting state
    currentState = maze.START_STATE
    # get action
    currentAction = chooseAction(currentState, stateActionValues, maze, dynaParams)

    # arrays to store states and rewards for an episode
    # space isn't a major consideration, so I didn't use the mod trick
    states = [currentState]
    actions = [currentAction]
    # rewards = [0]
    rewards = [0]

    steps = 0
    # while currentState not in maze.GOAL_STATES:

    # the length of this episode
    T = float('inf')
    while True:

        # track the steps
        steps += 1

        if steps < T:

            # take action
            newState, reward = maze.takeAction(currentState, currentAction)

            # store new state and new reward
            states.append(newState)
            rewards.append(reward)

            newAction = chooseAction(newState, stateActionValues, maze, dynaParams)
            actions.append(newAction)

            if newState in maze.GOAL_STATES:
                T = steps

        # get the time of the state to ft
        updateTime = steps - n
        if updateTime >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(updateTime + 1, min(T, updateTime + n) + 1):
                returns += pow(dynaParams.gamma, t - updateTime - 1) * rewards[t]
            # add state-action value to the return
            upN = updateTime + n
            if upN <= T:
                returns += pow(dynaParams.gamma, n) * stateActionValues[
                    states[(upN)][0], states[(upN)][1], actions[(upN)]]
            stateToUpdate = states[updateTime]
            # update the state-action value
            if not stateToUpdate in maze.GOAL_STATES:
                stateActionValues[stateToUpdate[0], stateToUpdate[1], actions[updateTime]] += dynaParams.alpha * (
                    returns - stateActionValues[stateToUpdate[0], stateToUpdate[1], actions[updateTime]])
                # # Q-Learning update
                # stateActionValues[currentState[0], currentState[1], action] += \
                #     dynaParams.alpha * (reward + dynaParams.gamma * np.max(stateActionValues[newState[0], newState[1], :]) -
                #                         stateActionValues[currentState[0], currentState[1], action])

                # # feed the model with experience
                # model.feed(currentState, action, newState, reward)

                # # sample experience from the model
                # for t in range(0, dynaParams.planningSteps):
                #     stateSample, actionSample, newStateSample, rewardSample = model.sample()
                #     stateActionValues[stateSample[0], stateSample[1], actionSample] += \
                #         dynaParams.alpha * (rewardSample + dynaParams.gamma * np.max(stateActionValues[newStateSample[0], newStateSample[1], :]) -
                #         stateActionValues[stateSample[0], stateSample[1], actionSample])

        if updateTime >= T - 1:
            break

        #check whether it has exceeded the step limit
        if T > maze.maxSteps:
            break

        currentState = newState
        currentAction = newAction

    return T

# Figure 8.3, DynaMaze, use 10 runs instead of 30 runs
def figure8_3():

    # set up an instance for DynaMaze
    dynaMaze = Maze()
    dynaParams = DynaParams()

    runs = 10
    episodes = 50

    # this random seed is for sampling from model
    # we do need this separate random seed to make sure the first episodes for all planning steps are the same
    rand = np.random.RandomState(0)

    # all possible steps
    multisteps = np.power(2, np.arange(2, 4))
    steps = np.zeros((len(multisteps), episodes))
    for run in range(0, runs):
        for stepInd, step in zip(range(len(multisteps)), multisteps):
            print('run: ', run, ' step: ', step, ' alpha: ', dynaParams.alpha)
            # set same random seed for each planning step
            np.random.seed(run)
            # Initiate state value to zeros
            currentStateActionValues = np.copy(dynaMaze.stateActionValues)
            for ep in range(0, episodes):
                # total_step, currentStateActionValues = temporalDifference(currentStateActionValues, step, dynaMaze, dynaParams)
                # steps[stepInd, ep] += total_step
                # print(int(steps[stepInd, ep]))
                stepnum = temporalDifference_wzn(currentStateActionValues, step, dynaMaze, dynaParams)
                #stepnum = temporalDifference(currentStateActionValues, step, dynaMaze, dynaParams)
                print('run:', run, 'episode:', ep, 'steps:', stepnum)
                steps[stepInd, ep] += stepnum


    # averaging over runs
    steps /= runs

    plt.figure(0)
    for i in range(0, len(multisteps)):
        plt.plot(range(0, episodes), steps[i, :], label=str(multisteps[i]) + ' planning steps')
        print(steps[i, :])
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()

    # planningSteps = [0, 5, 50]
    # steps = np.zeros((len(planningSteps), episodes))
    # for run in range(0, runs):
    #     for index, planningStep in zip(range(0, len(planningSteps)), planningSteps):
    #         dynaParams.planningSteps = planningStep
    #
    #         # set same random seed for each planning step
    #         np.random.seed(run)
    #         # Initiate state value to zeros
    #         currentStateActionValues = np.copy(dynaMaze.stateActionValues)
    #
    #         # generate an instance of Dyna-Q model
    #         model = TrivialModel(rand)
    #         for ep in range(0, episodes):
    #             print('run:', run, 'planning step:', planningStep, 'episode:', ep)
    #             steps[index, ep] += dynaQ(currentStateActionValues, model, dynaMaze, dynaParams)
    #             print(int(steps[index, ep]))
    #
    # # averaging over runs
    # steps /= runs
    #
    # plt.figure(0)
    # for i in range(0, len(planningSteps)):
    #     plt.plot(range(0, episodes), steps[i, :], label=str(planningSteps[i]) + ' planning steps')
    # plt.xlabel('episodes')
    # plt.ylabel('steps per episode')
    # plt.legend()



figure8_3()

plt.show()