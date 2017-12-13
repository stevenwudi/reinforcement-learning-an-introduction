import heapq


class GridWithWeights_3D():
    def __init__(self, width, height, time_length, wall_wind):
        self.width = width
        self.height = height
        self.time_length = time_length
        self.wall_wind = wall_wind
        self.weights = []

    def in_bounds(self, id):
        (x, y, t) = id
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= t < self.time_length

    def in_wind(self, id):
        (x, y, t) = id
        return self.weights[x, y, t] < self.wall_wind

    def neighbors(self, id):
        (x, y, t) = id
        # Voila, time only goes forward, but we can stay in the same position
        results = [(x + 1, y, t+1), (x, y - 1, t+1), (x - 1, y, t+1), (x, y + 1, t+1), (x, y, t+1)]
        #if (x + y) % 2 == 0 : results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        # we also need within the wall wind limit
        # results = filter(self.in_wind, results)  # However, with this condition, we might never find a route
        return results

    def cost(self, to_node):
        return self.weights[to_node]


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def heuristic(a, b):
    """
    https://en.wikipedia.org/wiki/A*_search_algorithm
     For the algorithm to find the actual shortest path, the heuristic function must be admissible,
     meaning that it never overestimates the actual cost to get to the nearest goal node.
     That's easy!
    :param a:
    :param b:
    :return:
    """
    if len(a) == 3:
        (x1, y1, t1) = a
        (x2, y2, t2) = b
    elif len(a) == 2:
        (x1, y1) = a
        (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)


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
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        # we change to in because the goals are in time span now
        if current in goals:
            break

        for next in graph.neighbors(current):
            #print(str(current) + '   ->   ' + str(next))
            new_cost = cost_so_far[current] + graph.cost(next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                # we use the x,y for goals because if admissible criterion is met, we are A OK!
                priority = new_cost + heuristic(goals[0], next)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


