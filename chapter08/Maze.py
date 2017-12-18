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
        self.ACTION_STAY = -1
        self.actions = [self.ACTION_UP, self.ACTION_DOWN, self.ACTION_LEFT, self.ACTION_RIGHT]

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


class GridWithWeights:
    def __init__(self, width, height):
        self.weights = []
        self.width = width
        self.height = height
        self.walls = []

    def cost(self, to_node):
        return self.weights[to_node]

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        return results


def convert_maze_to_grid(maze):
    """
    This function is mainly for
    :param maze:
    :return:
    """
    # data from main article
    diagram = GridWithWeights(maze.WORLD_WIDTH, maze.WORLD_HEIGHT)
    diagram.walls = [tuple(x) for x in maze.obstacles]
    return diagram


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
