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

# state A
STATE_A = 0

# state B
STATE_B = 1

# use one terminal state
STATE_TERMINAL = 2

# starts from state A
STATE_START = STATE_A

# possible actions in A
ACTION_A_RIGHT = 0
ACTION_A_LEFT = 1

# possible actions in B, maybe 10 actions
actionsOfB = range(0, 10)

# all possible actions
stateActions = [[ACTION_A_RIGHT, ACTION_A_LEFT], actionsOfB, [0]]

# state action pair values, if a state is a terminal state, then the value is always 0
stateActionValues = [np.zeros(2), np.zeros(len(actionsOfB)), np.zeros(1)]

# set up destination for each state and each action
actionDestination = [[STATE_TERMINAL, STATE_B], [STATE_TERMINAL] * len(actionsOfB)]

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.1

# discount for max value
GAMMA = 1.0

# choose an action based on epsilon greedy algorithm
def chooseAction(state, stateActionValues):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(stateActions[state])
    else:
        values_ = stateActionValues[state]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

# take @action in @state, return the reward
def takeAction(state, action):
    if state == STATE_A:
        return 0
    return np.random.normal(-0.1, 1)

# if there are two state action pair value array, use double Q-Learning
# otherwise use normal Q-Learning
def qLearning(stateActionValues, stateActionValues2=None):
    currentState = STATE_START
    # track the # of action left in state A
    leftCount = 0
    while currentState != STATE_TERMINAL:
        if stateActionValues2 is None:
            currentAction = chooseAction(currentState, stateActionValues)
        else:
            # derive a action form Q1 and Q2
            currentAction = chooseAction(currentState, [item1 + item2 for item1, item2 in zip(stateActionValues, stateActionValues2)])
        if currentState == STATE_A and currentAction == ACTION_A_LEFT:
            leftCount += 1
        reward = takeAction(currentState, currentAction)
        newState = actionDestination[currentState][currentAction]
        if stateActionValues2 is None:
            currentStateActionValues = stateActionValues
            targetValue = np.max(currentStateActionValues[newState])
        else:
            if np.random.binomial(1, 0.5) == 1:
                currentStateActionValues = stateActionValues
                anotherStateActionValues = stateActionValues2
            else:
                currentStateActionValues = stateActionValues2
                anotherStateActionValues = stateActionValues
            bestAction = np.argmax(currentStateActionValues[newState])
            targetValue = anotherStateActionValues[newState][bestAction]

        # Q-Learning update
        currentStateActionValues[currentState][currentAction] += ALPHA * (
            reward + GAMMA * targetValue - currentStateActionValues[currentState][currentAction])
        currentState = newState
    return leftCount


# if there are two state action pair value array, use double Expected-Sarsa
def Sarsa(stateActionValues, stateActionValues2=None, expected=False):
    currentState = STATE_START
    # track the # of action left in state A
    leftCount = 0
    while currentState != STATE_TERMINAL:
        if stateActionValues2 is None:
            currentAction = chooseAction(currentState, stateActionValues)
        else:
            # derive a action form Q1 and Q2
            currentAction = chooseAction(currentState, [item1 + item2 for item1, item2 in zip(stateActionValues, stateActionValues2)])
        if currentState == STATE_A and currentAction == ACTION_A_LEFT:
            leftCount += 1
        reward = takeAction(currentState, currentAction)
        newState = actionDestination[currentState][currentAction]
        if stateActionValues2 is None:
            currentStateActionValues = stateActionValues
            if not expected:
                newAction = chooseAction(newState, currentStateActionValues)
                targetValue = currentStateActionValues[newState][newAction]
            else:
                targetValue = 0
                actionValues = currentStateActionValues[newState]
                bestActions = np.argwhere(actionValues == np.max(actionValues))
                for action in stateActions[newState]:
                    if action in bestActions:
                        targetValue += ((1.0 - EPSILON) / len(bestActions) + EPSILON / len(stateActions[newState])) * currentStateActionValues[newState][action]
                    else:
                        targetValue += EPSILON / len(stateActions[newState]) * currentStateActionValues[newState][action]
        else:
            if np.random.binomial(1, 0.5) == 1:
                currentStateActionValues = stateActionValues
                anotherStateActionValues = stateActionValues2
            else:
                currentStateActionValues = stateActionValues2
                anotherStateActionValues = stateActionValues

            if not expected:
                newAction = chooseAction(newState, currentStateActionValues)
                targetValue = anotherStateActionValues[newState][newAction]
            else:
                targetValue = 0
                actionValues = currentStateActionValues[newState]
                bestActions = np.argwhere(actionValues == np.max(actionValues))
                for action in stateActions[newState]:
                    if action in bestActions:
                        targetValue += ((1.0 - EPSILON) / len(bestActions) + EPSILON / len(stateActions[newState])) * \
                                                                            anotherStateActionValues[newState][action]
                    else:
                        targetValue += EPSILON / len(stateActions[newState]) * anotherStateActionValues[newState][
                            action]

        # Sarsa update
        currentStateActionValues[currentState][currentAction] += ALPHA * (
            reward + GAMMA * targetValue - currentStateActionValues[currentState][currentAction])
        currentState = newState

    return leftCount


# Figure 6.8, 1,000 runs may be enough, # of actions in state B will also affect the curves
def figure6_8():
    # each independent run has 300 episodes
    episodes = 300
    runs = 1000

    leftCountsQ = np.zeros(episodes)
    leftCountsDoubleQ = np.zeros(episodes)
    leftCountsSarsa = np.zeros(episodes)
    leftCountsDoubleSarsa = np.zeros(episodes)
    leftCountsExpectedSarsa = np.zeros(episodes)
    leftCountsDoubleExpectedSarsa = np.zeros(episodes)

    for run in range(0, runs):
        print('run:', run)
        stateActionValuesQ = [np.copy(item) for item in stateActionValues]
        stateActionValuesSarsa = [np.copy(item) for item in stateActionValues]
        stateActionValuesExpectedSarsa = [np.copy(item) for item in stateActionValues]
        stateActionValuesDoubleQ1 = [np.copy(item) for item in stateActionValues]
        stateActionValuesDoubleQ2 = [np.copy(item) for item in stateActionValues]
        stateActionValuesDoubleSarsa1 = [np.copy(item) for item in stateActionValues]
        stateActionValuesDoubleSarsa2 = [np.copy(item) for item in stateActionValues]
        stateActionValuesDoubleExpectedSarsa1 = [np.copy(item) for item in stateActionValues]
        stateActionValuesDoubleExpectedSarsa2 = [np.copy(item) for item in stateActionValues]

        leftCountsQ_ = [0]
        leftCountsSarsa_ = [0]
        leftCountsExpectedSarsa_ = [0]
        leftCountsDoubleQ_ = [0]
        leftCountsDoubleSarsa_ = [0]
        leftCountsDoubleExpectedSarsa_ = [0]

        for ep in range(0, episodes):
            leftCountsQ_.append(leftCountsQ_[-1] + qLearning(stateActionValuesQ))
            leftCountsDoubleQ_.append(leftCountsDoubleQ_[-1] + qLearning(stateActionValuesDoubleQ1, stateActionValuesDoubleQ2))
            leftCountsSarsa_.append(leftCountsSarsa_[-1] + Sarsa(stateActionValuesSarsa, expected=False))
            leftCountsExpectedSarsa_.append(leftCountsExpectedSarsa_[-1] + Sarsa(stateActionValuesExpectedSarsa, expected=True))
            leftCountsDoubleSarsa_.append(leftCountsDoubleSarsa_[-1] + Sarsa(stateActionValuesDoubleSarsa1, stateActionValuesDoubleSarsa2, expected=False))
            leftCountsDoubleExpectedSarsa_.append(leftCountsDoubleExpectedSarsa_[-1] + Sarsa(stateActionValuesDoubleExpectedSarsa1, stateActionValuesDoubleExpectedSarsa2, expected=True))

        del leftCountsQ_[0]
        del leftCountsDoubleQ_[0]
        del leftCountsSarsa_[0]
        del leftCountsExpectedSarsa_[0]
        del leftCountsDoubleSarsa_[0]
        del leftCountsDoubleExpectedSarsa_[0]

        leftCountsQ += np.asarray(leftCountsQ_, dtype='float') / np.arange(1, episodes + 1)
        leftCountsDoubleQ += np.asarray(leftCountsDoubleQ_, dtype='float') / np.arange(1, episodes + 1)
        leftCountsSarsa += np.asarray(leftCountsSarsa_, dtype='float') / np.arange(1, episodes + 1)
        leftCountsExpectedSarsa += np.asarray(leftCountsExpectedSarsa_, dtype='float') / np.arange(1, episodes + 1)
        leftCountsDoubleSarsa += np.asarray(leftCountsDoubleSarsa_, dtype='float') / np.arange(1, episodes + 1)
        leftCountsDoubleExpectedSarsa += np.asarray(leftCountsDoubleExpectedSarsa_, dtype='float') / np.arange(1, episodes + 1)

    leftCountsQ /= runs
    leftCountsDoubleQ /= runs
    leftCountsSarsa /= runs
    leftCountsExpectedSarsa /= runs
    leftCountsDoubleSarsa /= runs
    leftCountsDoubleExpectedSarsa /= runs

    plt.figure()
    plt.plot(leftCountsQ, label='Q-Learning')
    plt.plot(leftCountsDoubleQ, label='Double Q-Learning')
    plt.plot(leftCountsSarsa, label='Sarsa')
    plt.plot(leftCountsExpectedSarsa, label='Expected Sarsa')
    plt.plot(leftCountsDoubleSarsa, label='Double Sarsa')
    plt.plot(leftCountsDoubleExpectedSarsa, label='Double Expected Sarsa')
    plt.plot(np.ones(episodes) * 0.05, label='Optimal')
    plt.xlabel('episodes')
    plt.ylabel('% left actions from A')
    plt.legend()

# it may take a while
figure6_8()
plt.show()