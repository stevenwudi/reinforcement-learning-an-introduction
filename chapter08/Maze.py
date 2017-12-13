import numpy as np


# A wrapper class for a maze, containing all the information about the maze.
# Basically it's initialized to DynaMaze by default, however it can be easily adapted
# to other maze
class Maze:
    def __init__(self):
        self.WORLD_WIDTH = 9
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

        # all cliff
        self.reward_goal = 0.0
        self.reward_move = -1.0
        self.reward_cliff = -100
        self.cliff = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
        self.oldObstacles = None
        self.newObstacles = None

        # time to change cliff
        self.changingPoint = None

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
        newMaze.cliff = []
        for state in self.cliff:
            newMaze.cliff.extend(self.extendState(state, factor))
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
            #x, y = state
            reward = self.reward_cliff
            #reward = 0
        elif [x, y] in self.GOAL_STATES:
            reward = self.reward_goal
            #reward = 1.0
        else:
            reward = self.reward_move
            #reward = 0
        return [x, y], reward


class GridWithWeights():
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
    diagram.walls = [tuple(x) for x in maze.cliff]
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
