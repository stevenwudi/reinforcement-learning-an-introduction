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
from chapter06.A_star_3D import a_star_search_3D, heuristic

# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
class Maze:
    def __init__(self):
        self.WORLD_WIDTH = 12
        self.WORLD_HEIGHT = 4
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

        # start state
        self.START_STATE = [self.WORLD_HEIGHT-1, 0]
        # goal state
        self.GOAL_STATES = [self.WORLD_HEIGHT-1, 11]
        # all obstacles
        self.cliff = [[self.WORLD_HEIGHT-1, x] for x in range(1, self.WORLD_WIDTH-1)]
        self.obstacles = self.cliff

        self.oldObstacles = None
        self.newObstacles = None

        # time to change obstacles
        self.changingPoint = None

        # initial reward for each action in each state
        self.reward_goal = 0.0
        self.reward_move = -1.0
        self.reward_cliff = -100
        self.actionRewards = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, len(self.actions)))
        # max steps
        self.maxSteps = float('inf')
        self.maxSteps = 10000

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
        if [x, y] in self.cliff:
            x, y = self.START_STATE
            reward = self.reward_cliff
        elif [x, y] in self.GOAL_STATES:
            reward = self.reward_goal
        else:
            reward = self.reward_move
        return [x, y], reward


# a wrapper class for parameters of TD_learning
class TD_learning:
    def __init__(self, maze, expected=False, qLearning=False, stepSize=0.5, epsilon=0.1, gamma=1.0):
        """
        :param stateActionValues: values for state action pair, will be updated
        :param expected: if True, will use expected Sarsa algorithm
        :param stepSize: step size for updating
        :param epsilon: epsilon greedy algorithm
        :param gamma: gamma for Expected Sarsa
        """
        self.maze = maze
        # initial state Action Values --> which are all zeros or it is such a bad choise?
        #self.heuristic = heuristic(maze.START_STATE, maze.GOAL_STATES)
        #self.stateActionValues = np.ones((maze.WORLD_HEIGHT, maze.WORLD_WIDTH, len(maze.actions))) * self.heuristic
        self.stateActionValues = np.zeros((maze.WORLD_HEIGHT, maze.WORLD_WIDTH, len(maze.actions)))
        self.expected = expected
        self.qLearning = qLearning
        self.stepSize = stepSize
        self.epsilon = epsilon
        self.gamma = gamma

    # choose an action based on epsilon greedy algorithm
    def chooseAction(self, state):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(self.maze.actions)
        else:
            values_ = self.stateActionValues[state[0], state[1], :]
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    def play(self):
        """
        play for an episode
        :return:
        """
        currentState = self.maze.START_STATE
        rewards = 0.0
        time_step = 0
        if not self.qLearning:
            currentAction = self.chooseAction(currentState)
            while currentState != self.maze.GOAL_STATES:
                time_step += 1
                newState, reward = self.maze.takeAction(currentState, currentAction)
                newAction = self.chooseAction(newState)
                rewards += reward
                if not self.expected:
                    valueTarget = self.stateActionValues[newState[0], newState[1], newAction]
                elif self.expected:
                    # calculate the expected value of new state
                    valueTarget = 0.0
                    actionValues = self.stateActionValues[newState[0], newState[1], :]
                    bestActions = np.argwhere(actionValues == np.max(actionValues))
                    for action in self.maze.actions:
                        if action in bestActions:
                            valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(self.maze.actions)) \
                                           * self.stateActionValues[newState[0], newState[1], action]
                        else:
                            valueTarget += self.epsilon / len(self.maze.actions) * self.stateActionValues[newState[0], newState[1], action]
                valueTarget *= self.gamma
                # Sarsa update
                self.stateActionValues[currentState[0], currentState[1], currentAction] += self.stepSize * (reward +
                    valueTarget - self.stateActionValues[currentState[0], currentState[1], currentAction])
                currentState = newState
                currentAction = newAction
            print(time_step)

        elif self.qLearning:
            while currentState != self.maze.GOAL_STATES:
                time_step += 1
                currentAction = self.chooseAction(currentState)
                newState, reward = self.maze.takeAction(currentState, currentAction)
                rewards += reward
                # Q-Learning update
                self.stateActionValues[currentState[0], currentState[1], currentAction] += self.stepSize * (reward + self.gamma * np.max(self.stateActionValues[newState[0], newState[1], :] - self.stateActionValues[currentState[0], currentState[1], currentAction]))
                currentState = newState
            print(time_step)

        max_q = np.sum(np.max(self.stateActionValues, axis=2))
        return rewards, max_q


# print optimal policy
def printOptimalPolicy(control_algorithm):
    optimalPolicy = []
    for i in range(0, control_algorithm.maze.WORLD_HEIGHT):
        optimalPolicy.append([])
        for j in range(0, control_algorithm.maze.WORLD_WIDTH):
            if [i, j] == control_algorithm.maze.GOAL_STATES:
                optimalPolicy[-1].append('G')
                continue
            bestAction = np.argmax(control_algorithm.stateActionValues[i, j, :])
            if bestAction == control_algorithm.maze.ACTION_UP:
                optimalPolicy[-1].append('U')
            elif bestAction == control_algorithm.maze.ACTION_DOWN:
                optimalPolicy[-1].append('D')
            elif bestAction == control_algorithm.maze.ACTION_LEFT:
                optimalPolicy[-1].append('L')
            elif bestAction == control_algorithm.maze.ACTION_RIGHT:
                optimalPolicy[-1].append('R')
    for row in optimalPolicy:
        print(row)


def walk_final_grid(control_algorithm):

    came_from = {}
    cost_so_far = {}
    came_from[tuple(control_algorithm.maze.START_STATE)] = None
    cost_so_far[tuple(control_algorithm.maze.START_STATE)] = 0
    # we use the maximum here directly
    control_algorithm.epsilon = 0

    currentState = control_algorithm.maze.START_STATE
    steps = 0
    while currentState != control_algorithm.maze.GOAL_STATES:
        # track the steps
        steps += 1
        # get action
        action = control_algorithm.chooseAction(currentState)
        # take action
        newState, reward = control_algorithm.maze.takeAction(currentState, action)
        cost_so_far[tuple(newState)] = reward
        came_from[tuple(newState)] = tuple(currentState)
        currentState = newState

        # check whether it has exceeded the step limit
        if steps > control_algorithm.maze.maxSteps:
            print(currentState)
            break

    return came_from, cost_so_far


def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]

        if x2 == x1 + 1: r = "\u2193"
        if x2 == x1 - 1: r = "\u2191"
        if y2 == y1 + 1: r = "\u2192"
        if y2 == y1 - 1: r = "\u2190"
    if 'start' in style and id == style['start']: r = "A"
    if 'goal' in style and id == style['goal']: r = "Z"
    if 'path' in style and id in style['path']: r = "@"
    if id in graph.walls: r = "#" + ' ' * (width-1)
    return r


def draw_grid(graph, width=2, **style):
    for x in range(graph.height):
        for y in range(graph.width):
            # Note: DIWU change (x, y) to (y, x) because the notation is different here
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
        print()


class GridWithWeights:
    def __init__(self, width, height, wall_wind):
        self.width = width
        self.height = height
        self.wall_wind = wall_wind
        self.weights = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.height and 0 <= y < self.width

    def in_wind(self, id):
        (x, y) = id
        return self.weights[x, y] < self.wall_wind

    def neighbors(self, id):
        (x, y) = id
        # Voila, time only goes forward, but we can stay in the same position
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1), (x, y)]
        results = filter(self.in_bounds, results)
        # we also need within the wall wind limit
        # results = filter(self.in_wind, results)  # However, with this condition, we might never find a route
        return results

    def cost(self, to_node):
        return self.weights[to_node]


def convert_maze_to_grid(maze):
    """
    This function is mainly for
    :param maze:
    :return:
    """
    # data from main article
    diagram = GridWithWeights(width=maze.WORLD_WIDTH, height=maze.WORLD_HEIGHT, wall_wind=-maze.reward_cliff)
    diagram.walls = [tuple(x) for x in maze.obstacles]
    diagram.weights = np.ones(shape=(maze.WORLD_HEIGHT, maze.WORLD_WIDTH))
    for c in maze.cliff:
        diagram.weights[c[0], c[1]] = -maze.reward_cliff

    return diagram


def figure6_5():
    # set up an instance for DynaMaze
    dynaMaze = Maze()

    # A star algorithm here
    diagram = convert_maze_to_grid(dynaMaze)
    A_star_came_from, A_star_cost_so_far = a_star_search_3D(diagram, start=tuple(dynaMaze.START_STATE), goals=[tuple(dynaMaze.GOAL_STATES)])
    draw_grid(diagram, width=3, point_to=A_star_came_from,
              start=tuple(dynaMaze.START_STATE), goals=tuple(dynaMaze.GOAL_STATES))
    # averaging the reward sums from 10 successive episodes
    averageRange = 10

    # episodes of each run
    nEpisodes = 13

    # perform 20 independent runs
    runs = 10

    rewardsSarsa = np.zeros(nEpisodes)
    rewardsExpectedSarsa = np.zeros(nEpisodes)
    rewardsQLearning = np.zeros(nEpisodes)
    QSarsa = np.zeros(nEpisodes)
    QExpectedSarsa = np.zeros(nEpisodes)
    QQLearning = np.zeros(nEpisodes)
    for run in range(0, runs):
        sarsa = TD_learning(dynaMaze)
        expected_sarsa = TD_learning(dynaMaze, expected=True)
        qLearning = TD_learning(dynaMaze, qLearning=True)
        for i in range(0, nEpisodes):
            # cut off the value by -100 to draw the figure more elegantly
            rewards, max_q = sarsa.play()
            rewardsSarsa[i], QSarsa[i] = rewardsSarsa[i]+max(rewards, -100), QSarsa[i] + max_q
            rewards, max_q = expected_sarsa.play()
            rewardsExpectedSarsa[i], QExpectedSarsa[i] = rewardsExpectedSarsa[i]+max(rewards, -100), QExpectedSarsa[i] + max_q
            rewards, max_q = qLearning.play()
            rewardsQLearning[i], QQLearning[i] = rewardsQLearning[i] + max(rewards, -100), QQLearning[i] + max_q

    # averaging over independent runs
    rewardsSarsa /= runs
    rewardsExpectedSarsa /= runs
    rewardsQLearning /= runs
    QSarsa /= runs
    QExpectedSarsa /= runs
    QQLearning /= runs

    # averaging over successive episodes
    smoothedRewardsSarsa = np.copy(rewardsSarsa)
    smoothedRewardsExpectedSarsa = np.copy(rewardsExpectedSarsa)
    smoothedRewardsQLearning = np.copy(rewardsQLearning)
    for i in range(averageRange, nEpisodes):
        smoothedRewardsSarsa[i] = np.mean(rewardsSarsa[i - averageRange: i + 1])
        smoothedRewardsExpectedSarsa[i] = np.mean(rewardsExpectedSarsa[i - averageRange: i + 1])
        smoothedRewardsQLearning[i] = np.mean(rewardsQLearning[i - averageRange: i + 1])
    # display optimal policy
    print('Sarsa Optimal Policy:')
    printOptimalPolicy(sarsa)
    print('Expected Sarsa Optimal Policy:')
    printOptimalPolicy(expected_sarsa)
    print('Expected Q-Learning Optimal Policy:')
    printOptimalPolicy(qLearning)

    # draw reward curves
    plt.figure(1)
    plt.plot(smoothedRewardsSarsa, label='Sarsa')
    plt.plot(smoothedRewardsExpectedSarsa, label='Expected Sarsa')
    plt.plot(smoothedRewardsQLearning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()

    # draw Q curves
    plt.figure(2)
    plt.plot(QSarsa, label='Sarsa')
    plt.plot(QExpectedSarsa, label='Expected Sarsa')
    plt.plot(QQLearning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Max Q during episode')
    plt.legend()

    print('sarsa')
    came_from, cost_so_far = walk_final_grid(sarsa)
    draw_grid(diagram, width=3, point_to=came_from,
              start=tuple(dynaMaze.START_STATE), goal=tuple(dynaMaze.GOAL_STATES))
    print()
    print('expected_sarsa')
    came_from, cost_so_far = walk_final_grid(expected_sarsa)
    draw_grid(diagram, width=3, point_to=came_from,
              start=tuple(dynaMaze.START_STATE), goal=tuple(dynaMaze.GOAL_STATES))
    print()
    print('qLearning')
    came_from, cost_so_far = walk_final_grid(qLearning)
    draw_grid(diagram, width=3, point_to=came_from,
              start=tuple(dynaMaze.START_STATE), goal=tuple(dynaMaze.GOAL_STATES))
    print()


figure6_5()
plt.show()
