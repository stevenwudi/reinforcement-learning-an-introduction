import heapq
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


def convert_maze_to_grid(maze):
    """
    This function is mainly for
    :param maze:
    :return:
    """
    # data from main article
    diagram = GridWithWeights_3D(maze.WORLD_HEIGHT, maze.WORLD_WIDTH)
    diagram.walls = [tuple(x) for x in maze.obstacles]
    for i in diagram.walls:
        diagram.weights[i] = maze.reward_obstacle * (-1)
    return diagram


def convert_3Dmaze_to_grid(maze):
    """
    This function is mainly for
    :param maze:
    :return:
    """
    # data from main article
    diagram = GridWithWeights_3D(maze.WORLD_HEIGHT, maze.WORLD_WIDTH, maze.TIME_LENGTH)
    diagram.weights = maze.obstacles_weight * maze.reward_obstacle * (-1)
    diagram.weights += maze.reward_move * (-1)
    diagram.wall_wind = maze.reward_obstacle * (-1)
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


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_grid_3d(graph, **style):
    """
    3D plot of a maze
    :param graph:
    :param style:
    :return:
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=11, azim=-19)

    # plot the walls
    walls_idx = np.where(graph.weights >= graph.wall_wind)
    ax.scatter(walls_idx[2], walls_idx[1], graph.height-walls_idx[0], c='b', marker='^')
    ax.set_xlim(0, graph.time_length); ax.set_xlabel('Time'); ax.set_xticks(range(0, graph.time_length))
    ax.set_ylim(0, graph.width); ax.set_ylabel('Width'); ax.set_yticks(range(0, graph.width))
    ax.set_zlim(0, graph.height); ax.set_zlabel('Height'); ax.set_zticks(range(0, graph.height))
    # plot the start
    ax.scatter(style['start'][2], style['start'][1], graph.height-style['start'][0], c='r', marker='o')
    for g in style['goal']:
        ax.scatter(g[2], g[1], graph.height-g[0], c='g', marker='o')

    # plot the path
    goal_reached = tuple(set(style['goal']).intersection(set(style['came_from'].keys())))[0]
    current_state = goal_reached
    while style['came_from'][current_state]:
        x1, y1, t1 = current_state
        x2, y2, t2 = style['came_from'][current_state]
        current_state = x2, y2, t2
        x2, x1 = graph.height-x2, graph.height-x1
        a = Arrow3D([t2, t1], [y2, y1],
                    [x2, x1], mutation_scale=10,
                    lw=1, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    if 'title' in style.keys():
        plt.title(style['title'])
    return


class GridWithWeights_3D():
    def __init__(self, height, width, time_length=0, wall_wind=15, cost=1):
        self.height = height
        self.width = width
        self.time_length = time_length
        self.wall_wind = wall_wind
        if self.time_length:
            self.weights = np.ones(shape=(height, width, time_length)) * cost
        else:
            self.weights = np.ones(shape=(height, width)) * cost

    def in_bounds(self, id):
        if self.time_length:
            (x, y, t) = id
            return 0 <= x < self.height and 0 <= y < self.width and 0 <= t < self.time_length
        else:
            (x, y) = id
            return 0 <= x < self.height and 0 <= y < self.width

    def in_wind(self, id):
        (x, y, t) = id
        return self.weights[x, y, t] < self.wall_wind

    def neighbors(self, id):
        if self.time_length:
            (x, y, t) = id
            # Voila, time only goes forward, but we can stay in the same position
            results = [(x + 1, y, t + 1), (x, y - 1, t + 1), (x - 1, y, t + 1), (x, y + 1, t + 1), (x, y, t + 1)]
            # if (x + y) % 2 == 0 : results.reverse()  # aesthetics
            results = filter(self.in_bounds, results)
            # we also need within the wall wind limit
            # results = filter(self.in_wind, results)  # However, with this condition, we might never find a route
        else:
            (x, y) = id
            # Voila, time only goes forward, but we can stay in the same position
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
            # if (x + y) % 2 == 0 : results.reverse()  # aesthetics
            results = filter(self.in_bounds, results)
            # we also need within the wall wind limit
            # results = filter(self.in_wind, results)  # However, with this condition, we might never find a route
        return results

    def cost(self, to_node):
        return self.weights[to_node]

    def heuristic(self, a, b):
        """
        https://en.wikipedia.org/wiki/A*_search_algorithm
         For the algorithm to find the actual shortest path, the heuristic function must be admissible,
         meaning that it never overestimates the actual cost to get to the nearest goal node.
         That's easy!
        :param a:
        :param b:
        :return:
        """
        if self.time_length:
            (x1, y1, t1) = a[0]
            (x2, y2, t2) = b
        else:
            (x1, y1) = a[0]
            (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def a_star_search_3D(graph, start, goals):
    """
    :param graph:
    :param start: 3D, (x,y,0)
    :param goal: 2D, spam 3D space
    :return:
    """
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    go_to = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        # we change to in because the goals are in time span now
        if current in goals:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(next)
            #print(str(current) + '   ->   ' + str(next) + ', cost: ' + '   ->  ' + str(new_cost))
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # we use the x,y for goals because if admissible criterion is met, we are A OK!
                priority = new_cost + graph.heuristic(goals, next)
                #print(str(current) + '   ->   ' + str(next) + ', cost: ' + '   ->  ' + str(priority))
                frontier.put(next, priority)
                came_from[next] = current
                if current in go_to.keys():
                    go_to[current].append(next)
                else:
                    go_to[current] = [next]

    return came_from, cost_so_far, go_to


def walk_final_grid_go_to(START_STATE, GOAL_STATE, came_from):
    """
    A helper function to walk the whole grid world
    :return:
    """
    go_to_all = {}
    currentState = GOAL_STATE
    steps = 0
    while not currentState == START_STATE:
        # track the steps
        steps += 1
        prev = came_from[currentState]
        go_to_all[prev] = currentState
        currentState = prev

    return go_to_all, steps