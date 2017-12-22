import numpy as np


class Maze:
    """
    A wrapper class for a maze, containing all the information about the maze.
    Basically it's initialized to DynaMaze by default, however it can be easily adapted
    to other maze
    """
    def __init__(self, width, height, start_state, goal_states, obstacles=[], return_to_start=False,
                 reward_goal=0.0, reward_move=-1.0, reward_obstacle=-100.,
                 stochastic_wind=[], stochastic_wind_direction=None):
        self.WORLD_WIDTH = width
        self.WORLD_HEIGHT = height
        self.stochastic_wind = stochastic_wind
        self.stochastic_wind_direction = stochastic_wind_direction
        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        #self.ACTION_STAY = -1
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]
        #self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_STAY]

        # start state
        self.START_STATE = start_state
        # goal state
        self.GOAL_STATES = goal_states

        # all obstacles
        self.reward_goal = reward_goal
        self.reward_move = reward_move
        self.reward_obstacle = reward_obstacle
        self.obstacles = obstacles
        self.return_to_start = return_to_start
        self.oldObstacles = None
        self.newObstacles = None

        # time to change obstacles
        self.changingPoint = None

        # max steps
        self.maxSteps = float('inf')
        self.maxSteps = 100000

        # track the resolution for this maze
        self.resolution = 1

    def extendState(self, state, factor):
        """
         Extend a state to a higher resolution maze
        :param state: state in lower resolution maze
        :param factor: extension factor, one state will become factor^2 states after extension
        :return:
        """
        newState = [state[0] * factor, state[1] * factor]
        newStates = []
        for i in range(0, factor):
            for j in range(0, factor):
                newStates.append([newState[0] + i, newState[1] + j])
        return newStates

    def extendMaze(self, factor):
        """
        Extend a state into higher resolution
        :param factor: one state in original maze will become @factor^2 states in @return new maze
        :return:
        """
        newMaze = Maze(width=self.WORLD_WIDTH * factor,
                       height=self.WORLD_HEIGHT * factor,
                       start_state=[self.START_STATE[0] * factor, self.START_STATE[1] * factor],
                       goal_states=self.extendState(self.GOAL_STATES[0], factor),
                       return_to_start=self.return_to_start,
                       reward_goal=self.reward_goal,
                       reward_move=self.reward_move,
                       reward_obstacle=self.reward_obstacle
                       )
        newMaze.obstacles = []
        for state in self.obstacles:
            newMaze.obstacles.extend(self.extendState(state, factor))
        newMaze.resolution = factor
        return newMaze

    def wind_blow(self, state):
        x, y = state
        action_times = self.stochastic_wind[x, y]
        # for a in range(action_times):
        #     action = self.stochastic_wind_direction
        #     [x, y], _ = self.takeAction(state, action, [])
        #     state = [x, y]

        ## The following is for Book exe. 6.10
        if action_times:
            action = np.random.choice([0, -1, 1])
            [x, y], _ = self.takeAction(state, action, [])
        return x, y

    def takeAction(self, state, action, stochastic_wind=[]):
        """
        :param state:
        :param action:
        :param stochastic_wind:
        :return:
        """
        x, y = state
        if len(stochastic_wind):
            # this funcionality is mainly for Book Fig.6.3...
            x, y = self.wind_blow(state)

        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        # elif action == self.ACTION_STAY:
        #     x, y = x, y

        if [x, y] in self.obstacles:
            if self.return_to_start:
                x, y = self.START_STATE
            else:
                x, y = state
            reward = self.reward_obstacle
        elif [x, y] in self.GOAL_STATES:
            reward = self.reward_goal
        else:
            reward = self.reward_move
        return [x, y], reward


class Maze_3D:
    """
    A wrapper class for a 3D maze, containing all the information about the maze.
    Maze has the third dimension t
    to other maze
    """
    def __init__(self, width, height, time_length,
                 start_state, goal_states,
                 return_to_start=False,
                 reward_goal=0.0, reward_move=-1.0, reward_obstacle=-100.,
                 stochastic_wind=[], stochastic_wind_direction=None):
        self.WORLD_WIDTH = width
        self.WORLD_HEIGHT = height
        self.TIME_LENGTH = time_length
        self.stochastic_wind = stochastic_wind
        self.stochastic_wind_direction = stochastic_wind_direction
        # all possible actions
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        self.ACTION_STAY = 4
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_STAY]

        # start state
        self.START_STATE = start_state
        # goal state
        self.GOAL_STATES = goal_states

        # all obstacles
        self.reward_goal = reward_goal
        self.reward_move = reward_move
        self.reward_obstacle = reward_obstacle

        self.obstacles_weight = np.zeros(shape=(self.WORLD_HEIGHT, self.WORLD_WIDTH, self.TIME_LENGTH))

        self.return_to_start = return_to_start
        self.oldObstacles = None
        self.newObstacles = None

        # time to change obstacles
        self.changingPoint = None

        # max steps
        self.maxSteps = float('inf')
        self.maxSteps = 100000

        # track the resolution for this maze
        self.resolution = 1

    def extendState(self, state, factor):
        """
         Extend a state to a higher resolution maze
        :param state: state in lower resolution maze
        :param factor: extension factor, one state will become factor^2 states after extension
        :return:
        """
        newState = [state[0] * factor, state[1] * factor]
        newStates = []
        for i in range(0, factor):
            for j in range(0, factor):
                newStates.append([newState[0] + i, newState[1] + j])
        return newStates

    def extendMaze(self, factor):
        """
        Extend a state into higher resolution
        :param factor: one state in original maze will become @factor^2 states in @return new maze
        :return:
        """
        newMaze = Maze(width=self.WORLD_WIDTH * factor,
                       height=self.WORLD_HEIGHT * factor,
                       start_state=[self.START_STATE[0] * factor, self.START_STATE[1] * factor],
                       goal_states=self.extendState(self.GOAL_STATES[0], factor),
                       return_to_start=self.return_to_start,
                       reward_goal=self.reward_goal,
                       reward_move=self.reward_move,
                       reward_obstacle=self.reward_obstacle
                       )
        newMaze.obstacles = []
        for state in self.obstacles:
            newMaze.obstacles.extend(self.extendState(state, factor))
        newMaze.resolution = factor
        return newMaze

    def wind_blow(self, state):
        x, y = state
        action_times = self.stochastic_wind[x, y]
        # for a in range(action_times):
        #     action = self.stochastic_wind_direction
        #     [x, y], _ = self.takeAction(state, action, [])
        #     state = [x, y]

        ## The following is for Book exe. 6.10
        if action_times:
            action = np.random.choice([0, -1, 1])
            [x, y], _ = self.takeAction(state, action, [])
        return x, y

    def takeAction(self, state, action, stochastic_wind=[]):
        """
        :param state:
        :param action:
        :param stochastic_wind:
        :return:
        """
        x, y, t = state
        # the time always goes forward
        t += 1
        if t >= self.TIME_LENGTH:
            # It will be a very undesirable state, we go back to the start state
            x, y, t = self.START_STATE
            reward = self.reward_obstacle
            return [x, y, t], reward

        if len(stochastic_wind):
            # this funcionality is mainly for Book Fig.6.3...
            x, y = self.wind_blow(state)

        if action == self.ACTION_UP:
            x = max(x - 1, 0)
        elif action == self.ACTION_DOWN:
            x = min(x + 1, self.WORLD_HEIGHT - 1)
        elif action == self.ACTION_LEFT:
            y = max(y - 1, 0)
        elif action == self.ACTION_RIGHT:
            y = min(y + 1, self.WORLD_WIDTH - 1)
        elif action == self.ACTION_STAY:
             x, y = x, y

        if self.obstacles_weight[x, y, t]:
            if self.return_to_start:
                x, y, t = self.START_STATE
            else:
                x, y, _ = state
            reward = self.reward_obstacle
        elif [x, y, t] in self.GOAL_STATES:
            reward = self.reward_goal
        else:
            reward = self.reward_move
        return [x, y, t], reward


