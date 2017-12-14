import numpy as np
import matplotlib.pyplot as plt
from chapter08.Maze import Maze, convert_maze_to_grid, draw_grid
from chapter08.Dyna import Dyna
from chapter08.Astar import a_star_search


def main():
    # set up an instance for DynaMaze
    dynaMaze = Maze(width=9,
                    height=6,
                    start_state=[2, 0],
                    goal_states=[[0, 8]],
                    return_to_start=True)
    dynaMaze.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
    runs = 10
    episodes = 50
    planningSteps = [0, 5, 50]
    steps = np.zeros((len(planningSteps), episodes))

    # this random seed is for sampling from model
    # we do need this separate random seed to make sure the first episodes for all planning steps are the same
    rand = np.random.RandomState(0)

    # ################### Now we do Sarsa #################################################
    for run in range(0, runs):
        for index, planningStep in zip(range(0, len(planningSteps)), planningSteps):
            # set same random seed for each planning step
            np.random.seed(run)
            # generate an instance of Dyna-Q model
            model_Dyna_Sarsa = Dyna(rand=rand, maze=dynaMaze, planningSteps=planningStep, qLearning=False, expected=False, alpha=0.1)
            for ep in range(0, episodes):
                print('run:', run, 'planning step:', planningStep, 'episode:', ep)
                steps[index, ep] += model_Dyna_Sarsa.play()
                print(steps[index, ep])

    # averaging over runs
    steps /= runs

    plt.figure(0)
    for i in range(0, len(planningSteps)):
        plt.plot(range(0, episodes), steps[i, :], label=str(planningSteps[i]) + ' planning steps')
    print(steps[i, :])
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.title(model_Dyna_Sarsa.name)
    plt.legend()

    came_from, cost_so_far = model_Dyna_Sarsa.walk_final_grid()
    diagram = convert_maze_to_grid(dynaMaze)
    draw_grid(diagram, width=3, point_to=came_from,
              start=tuple(dynaMaze.START_STATE), goal=tuple(dynaMaze.GOAL_STATES[0]))

    print('Finish')
    ################################# Now DynaQ ##############################
    for run in range(0, runs):
        for index, planningStep in zip(range(0, len(planningSteps)), planningSteps):
            # set same random seed for each planning step
            np.random.seed(run)
            # generate an instance of Dyna-Q model
            model_DynaQ = Dyna(rand=rand, maze=dynaMaze, planningSteps=planningStep, qLearning=True)
            for ep in range(0, episodes):
                print('run:', run, 'planning step:', planningStep, 'episode:', ep)
                steps[index, ep] += model_DynaQ.play()
                print(steps[index, ep])

    # averaging over runs
    steps /= runs

    plt.figure(0)
    for i in range(0, len(planningSteps)):
        plt.plot(range(0, episodes), steps[i, :], label=str(planningSteps[i]) + ' planning steps')
    print(steps[i, :])
    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.title(model_DynaQ.name)
    plt.legend()

    came_from, cost_so_far = model_DynaQ.walk_final_grid()
    diagram = convert_maze_to_grid(dynaMaze)
    draw_grid(diagram, width=3, point_to=came_from,
              start=tuple(dynaMaze.START_STATE), goal=tuple(dynaMaze.GOAL_STATES[0]))

    print('Finish')

    #came_from, cost_so_far = a_star_search(diagram, tuple(dynaMaze.START_STATE), tuple(dynaMaze.GOAL_STATES[0]))
    #draw_grid_a_star(diagram, width=3, point_to=came_from, start=tuple(dynaMaze.START_STATE), goal=tuple(dynaMaze.GOAL_STATES[0]))


def figure8_5():
    # set up a blocking maze instance
    blockingMaze = Maze(width=9,
                        height=6,
                        start_state=[5, 3],
                        goal_states=[[0, 8]],
                        return_to_start=False,
                        reward_goal=1.0,
                        reward_move=0.0,
                        reward_obstacle=0.0
                        )
    blockingMaze.oldObstacles = [[3, i] for i in range(0, 8)]
    # new obstalces will block the optimal path
    blockingMaze.newObstacles = [[3, i] for i in range(1, 9)]
    # step limit
    maxSteps = 3000
    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    blockingMaze.changingPoint = 1000
    blockingMaze.maxSteps = maxSteps

    rand = np.random.RandomState(0)
    # generate an instance of Dyna-Q model
    # track the cumulative rewards
    rewards = np.zeros((2, maxSteps))

    runs = 10
    for run in range(0, runs):
        # set up models
        model_Dyna = Dyna(rand=rand,
                          maze=blockingMaze,
                          planningSteps=5,
                          qLearning=True,
                          expected=False,
                          alpha=0.7)

        model_Dyna_plus = Dyna(rand=rand,
                               maze=blockingMaze,
                               planningSteps=5,
                               qLearning=True,
                               expected=False,
                               alpha=0.7,
                               plus=True,
                               kappa=1e-4)
        models = [model_Dyna, model_Dyna_plus]
        # track cumulative reward in current run
        rewards_ = np.zeros(shape=(2, maxSteps))
        for i, model in enumerate(models):
            print('run:', run, model.name)

            # set old obstacles for the maze
            model.maze.obstacles = blockingMaze.oldObstacles
            steps = 0
            lastSteps = steps
            while steps < maxSteps:
                # play for an episode
                steps += model.play()
                # update cumulateive rewards
                steps_ = min(steps, maxSteps-1)
                rewards_[i, lastSteps:steps_] = rewards_[i, lastSteps]
                rewards_[i, steps_] = rewards_[i, lastSteps] + 1
                lastSteps = steps
                if steps > blockingMaze.changingPoint:
                    # change the obstacles
                    model.maze.obstacles = blockingMaze.newObstacles

        rewards += rewards_
    rewards /= runs
    plt.figure(1)
    for i in range(0, len(models)):
        plt.plot(range(0, maxSteps), rewards[i, :], label=models[i].name)
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()
    plt.show()


def figure8_6():
    # set up a blocking maze instance
    blockingMaze = Maze(width=9,
                        height=6,
                        start_state=[5, 3],
                        goal_states=[[0, 8]],
                        return_to_start=False,
                        reward_goal=1.0,
                        reward_move=0.0,
                        reward_obstacle=0.0
                        )
    blockingMaze.oldObstacles = [[3, i] for i in range(1, 9)]
    # new obstalces will block the optimal path
    blockingMaze.newObstacles = [[3, i] for i in range(1, 8)]
    # step limit
    maxSteps = 6000
    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    blockingMaze.changingPoint = 3000
    blockingMaze.maxSteps = maxSteps

    rand = np.random.RandomState(0)
    # generate an instance of Dyna-Q model
    # track the cumulative rewards
    rewards = np.zeros((2, maxSteps))

    runs = 5
    planningSteps = 50
    alpha = 0.7
    for run in range(0, runs):
        # set up models
        model_Dyna = Dyna(rand=rand,
                          maze=blockingMaze,
                          planningSteps=planningSteps,
                          qLearning=True,
                          expected=False,
                          alpha=alpha)

        model_Dyna_plus = Dyna(rand=rand,
                               maze=blockingMaze,
                               planningSteps=planningSteps,
                               qLearning=True,
                               expected=False,
                               alpha=alpha,
                               plus=True,
                               kappa=1e-3)
        models = [model_Dyna, model_Dyna_plus]
        # track cumulative reward in current run
        rewards_ = np.zeros(shape=(2, maxSteps))
        for i, model in enumerate(models):
            print('run:', run, model.name)

            # set old obstacles for the maze
            model.maze.obstacles = blockingMaze.oldObstacles
            steps = 0
            lastSteps = steps
            while steps < maxSteps:
                # play for an episode
                steps += model.play()
                # update cumulateive rewards
                steps_ = min(steps, maxSteps-1)
                rewards_[i, lastSteps:steps_] = rewards_[i, lastSteps]
                rewards_[i, steps_] = rewards_[i, lastSteps] + 1
                lastSteps = steps
                if steps > blockingMaze.changingPoint:
                    # change the obstacles
                    model.maze.obstacles = blockingMaze.newObstacles

        rewards += rewards_
    rewards /= runs
    plt.figure(1)
    for i in range(0, len(models)):
        plt.plot(range(0, maxSteps), rewards[i, :], label=models[i].name)
    plt.xlabel('time steps')
    plt.ylabel('cumulative reward')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    figure8_6()