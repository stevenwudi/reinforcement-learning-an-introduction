import numpy as np
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from chapter08.Maze import Maze, convert_maze_to_grid, draw_grid
from chapter08.Dyna import Dyna
from timeit import default_timer as timer
from chapter08.Astar import a_star_search


def main():
    # set up an instance for DynaMaze
    dynaMaze = Maze(height=6,
                    width=9,
                    start_state=[2, 0],
                    goal_states=[[0, 8]],
                    reward_goal=1.0,
                    reward_move=0.0,
                    reward_obstacle=0.0,
                    return_to_start=False)
    dynaMaze.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
    runs = 2
    episodes = 50
    planningSteps = [5, 50]
    steps = np.zeros((len(planningSteps), 3, episodes))
    # model hyper-parameters
    alpha = 0.1
    gamma = 0.95
    theta = 1e-4
    # this random seed is for sampling from model
    # we do need this separate random seed to make sure the first episodes for all planning steps are the same
    rand = np.random.RandomState(0)
    # set same random seed for each planning step
    np.random.seed(0)

    # ################### Now we do Sarsa #################################################
    for run in range(0, runs):
        for index, planningStep in enumerate(planningSteps):

            # generate an instance of Dyna-Q model
            model_Dyna_Sarsa = Dyna(rand=rand,
                                    maze=dynaMaze,
                                    planningSteps=planningStep,
                                    qLearning=False,
                                    expected=False,
                                    alpha=alpha)

            model_Dyna_Q = Dyna(rand=rand,
                               maze=dynaMaze,
                               planningSteps=planningStep,
                               gamma=gamma,
                               alpha=alpha,
                               qLearning=True)

            model_Dyna_Sarsa_Expected = Dyna(rand=rand,
                                             maze=dynaMaze,
                                             gamma=gamma,
                                             planningSteps=planningStep,
                                             qLearning=False,
                                             expected=True,
                                             plus=False,
                                             alpha=alpha)

            model_Dyna_PS = Dyna(rand=rand,
                                 maze=dynaMaze,
                                 gamma=gamma,
                                 planningSteps=planningStep,
                                 qLearning=True,
                                 expected=False,
                                 alpha=alpha,
                                 priority=True,
                                 theta=theta)

            if planningStep == 0:
                models = [model_Dyna_Sarsa, model_Dyna_Sarsa_Expected, model_Dyna_Q]
            else:
                models = [model_Dyna_Sarsa, model_Dyna_Sarsa_Expected, model_Dyna_Q, model_Dyna_PS]
            models = [model_Dyna_Q, model_Dyna_Sarsa_Expected, model_Dyna_PS]

            for m, model in enumerate(models):
                for ep in range(0, episodes):
                    #print('run:', run, 'planning step:', planningStep, 'episode:', ep, 'model: ', model.name)
                    print('planning step:', planningStep, 'episode:', ep, 'model: ', model.name)
                    steps[index, m, ep] += model.play(environ_step=True)

    # averaging over runs
    linestyles = ['-', '--', '-.', ':']

    plt.figure(0)
    for i in range(0, len(planningSteps)):
        for m, model in enumerate(models):
            plt.plot(range(0, episodes), steps[i, m, :],
                     label=model.name + "_" + str(planningSteps[i]), linestyle=linestyles[m])

    plt.xlabel('episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    plt.show()
    print('Finish')


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
    planningSteps = 5
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
    plt.title('planningSteps: %d, alpha: %.2f' % (planningSteps, alpha))
    plt.legend()
    plt.show()


def figure8_7():
    # get the original 6*9 maze
    original_maze = Maze(width=9,
                        height=6,
                        start_state=[2, 0],
                        goal_states=[[0, 8]],
                        return_to_start=False,
                        reward_goal=1.0,
                        reward_move=0.0,
                        reward_obstacle=0.0
                        )
    #original_maze.obstacles = [[3, i] for i in range(0, 8)]
    original_maze.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # say 1st maze has w * h states, then k-th maze has w * h * k * k states
    numOfMazes = 5
    # build all the mazes
    resize_idx = 1
    mazes = [original_maze.extendMaze(i) for i in range(resize_idx, numOfMazes + 1)]
    # My machine cannot afford too many runs...
    runs = 3
    rand = np.random.RandomState(0)
    planningSteps = 5
    alpha = 0.5
    gamma = 0.95
    theta = 1e-4
    epsilon = 0.1
    # track the # of backups
    backups = np.zeros((2, numOfMazes))

    for run in range(0, runs):
        for mazeIndex, maze in enumerate(mazes):
            maze.GOAL_STATES = [maze.GOAL_STATES[0]]
            model_Dyna_PS = Dyna(rand=rand,
                                 maze=maze,
                                 epsilon=epsilon,
                                 gamma=gamma,
                                 planningSteps=planningSteps,
                                 qLearning=True,
                                 expected=False,
                                 alpha=alpha,
                                 priority=True,
                                 theta=theta)

            model_Dyna = Dyna(rand=rand,
                              maze=maze,
                              epsilon=epsilon,
                              gamma=gamma,
                              planningSteps=planningSteps,
                              qLearning=True,
                              expected=False,
                              plus=False,
                              alpha=alpha)

            models = [model_Dyna_PS, model_Dyna]
            for m, model in enumerate(models):
                print('run:', run, model.name, 'maze size:', maze.WORLD_HEIGHT * maze.WORLD_WIDTH)
                # track steps / backups for each episode
                steps = []
                # play for an episode
                while True:
                    steps.append(model.play())
                    # print best action w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)
                    # check whether the (relaxed) optimal path is found
                    if model.checkPath():
                        break

                # update the total steps / backups for this maze
                backups[m][mazeIndex] += np.sum(steps)

                came_from, cost_so_far = model.walk_final_grid()
                diagram = convert_maze_to_grid(maze)
                s = set()
                for item in maze.GOAL_STATES:
                    s.add(tuple(item))
                draw_grid(diagram, width=3, point_to=came_from,
                          start=tuple(maze.START_STATE), goal=list(s.intersection(set(came_from.keys())))[0])

    # Dyna-Q performs several backups per step
    backups[:, :] *= planningSteps
    # average over independent runs
    backups /= float(runs)

    plt.figure(3)
    for i in range(0, len(models)):
        plt.plot(np.arange(resize_idx, numOfMazes + 1), backups[i, :], label=models[i].name)
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()
    print('Finish')
    plt.show()


def model_play_worker(model):
    # track steps / backups for each episode
    steps = []
    # play for an episode
    while True:
        steps.append(model.play())
        # print best action w.r.t. current state-action values
        # printActions(currentStateActionValues, maze)
        # check whether the (relaxed) optimal path is found
        if model.checkPath():
            break

    came_from, cost_so_far = model.walk_final_grid()
    diagram = convert_maze_to_grid(model.maze)
    s = set()
    for item in model.maze.GOAL_STATES:
        s.add(tuple(item))
    draw_grid(diagram, width=3, point_to=came_from,
              start=tuple(model.maze.START_STATE), goal=list(s.intersection(set(came_from.keys())))[0])

    return steps


def figure8_7_multiprocessing():
    # get the original 6*9 maze
    original_maze = Maze(width=9,
                        height=6,
                        start_state=[2, 0],
                        goal_states=[[0, 8]],
                        return_to_start=False,
                        reward_goal=1.0,
                        reward_move=0.0,
                        reward_obstacle=0.0
                        )
    original_maze.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]

    # due to limitation of my machine, I can only perform experiments for 5 mazes
    # say 1st maze has w * h states, then k-th maze has w * h * k * k states
    numOfMazes = 5
    # build all the mazes
    resize_idx = 1
    mazes = [original_maze.extendMaze(i) for i in range(resize_idx, numOfMazes + 1)]
    # My machine cannot afford too many runs...
    runs = 1
    rand = np.random.RandomState(0)
    planningSteps = 5
    alpha = 0.5
    gamma = 0.95
    theta = 1e-4
    epsilon = 0.1
    # track the # of backups

    start_time = timer()
    pool = Pool(processes=5)
    backups = np.zeros((runs, 4, numOfMazes))

    for mazeIndex, maze in enumerate(mazes):
        maze.GOAL_STATES = [maze.GOAL_STATES[0]]
        model_Dyna_PS_Q = Dyna(rand=rand,
                             maze=maze,
                             epsilon=epsilon,
                             gamma=gamma,
                             planningSteps=planningSteps,
                             qLearning=True,
                             expected=False,
                             alpha=alpha,
                             priority=True,
                             theta=theta)

        model_Dyna_PS_Sarsa = Dyna(rand=rand,
                             maze=maze,
                             epsilon=epsilon,
                             gamma=gamma,
                             planningSteps=planningSteps,
                             qLearning=False,
                             expected=False,
                             alpha=alpha,
                             priority=True,
                             theta=theta)

        model_Dyna_PS_Sarsa_expected = Dyna(rand=rand,
                             maze=maze,
                             epsilon=epsilon,
                             gamma=gamma,
                             planningSteps=planningSteps,
                             qLearning=False,
                             expected=True,
                             alpha=alpha,
                             priority=True,
                             theta=theta)

        model_Dyna = Dyna(rand=rand,
                          maze=maze,
                          epsilon=epsilon,
                          gamma=gamma,
                          planningSteps=planningSteps,
                          qLearning=True,
                          expected=False,
                          plus=False,
                          alpha=alpha)

        models = [model_Dyna_PS_Q, model_Dyna_PS_Sarsa, model_Dyna_PS_Sarsa_expected, model_Dyna]
        for m, model in enumerate(models):
            for run in range(0, runs):
                print('run:', run, model.name, 'maze size:', maze.WORLD_HEIGHT * maze.WORLD_WIDTH)
                result = pool.apply_async(model_play_worker, [model])
                backups[run][m][mazeIndex] = np.sum(result.get())

    pool.close()
    print('Finish, using %.2f sec!' % (timer() - start_time))

    # Dyna-Q performs several backups per step
    backups = np.sum(backups, axis=0)
    # average over independent runs
    backups /= float(runs)

    plt.figure(3)
    for i in range(0, len(models)):
        plt.plot(np.arange(resize_idx, numOfMazes + 1), backups[i, :], label=models[i].name)
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()
    print('Finish')
    plt.show()


def figure8_7_A_star():
    # get the original 6*9 maze
    original_maze = Maze(width=9,
                        height=6,
                        start_state=[2, 0],
                        goal_states=[[0, 8]],
                        return_to_start=False,
                        reward_goal=1.0,
                        reward_move=0.0,
                        reward_obstacle=0.0
                        )
    original_maze.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]

    # say 1st maze has w * h states, then k-th maze has w * h * k * k states
    numOfMazes = 1
    # build all the mazes
    resize_idx = 5
    mazes = [original_maze.extendMaze(i) for i in range(resize_idx, numOfMazes + 1)]
    # Dyna model hyper
    rand = np.random.RandomState(0)
    planningSteps = 5
    alpha = 0.5
    gamma = 0.95
    theta = 1e-4
    epsilon = 0.1
    # track the # of backups

    start_time = timer()
    pool = Pool(processes=5)
    backups = np.zeros((2, numOfMazes))

    for mazeIndex, maze in enumerate(mazes):
        maze.GOAL_STATES = [maze.GOAL_STATES[0]]
        model_Dyna_PS_Q = Dyna(rand=rand,
                             maze=maze,
                             epsilon=epsilon,
                             gamma=gamma,
                             planningSteps=planningSteps,
                             qLearning=True,
                             expected=False,
                             alpha=alpha,
                             priority=True,
                             theta=theta)


        models = [model_Dyna_PS_Q]
        for m, model in enumerate(models):
            for run in range(0, runs):
                print('run:', run, model.name, 'maze size:', maze.WORLD_HEIGHT * maze.WORLD_WIDTH)
                result = pool.apply_async(model_play_worker, [model])
                backups[run][m][mazeIndex] = np.sum(result.get())

    pool.close()
    print('Finish, using %.2f sec!' % (timer() - start_time))

    # Dyna-Q performs several backups per step
    backups = np.sum(backups, axis=0)
    # average over independent runs
    backups /= float(runs)

    plt.figure(3)
    for i in range(0, len(models)):
        plt.plot(np.arange(resize_idx, numOfMazes + 1), backups[i, :], label=models[i].name)
    plt.xlabel('maze resolution factor')
    plt.ylabel('backups until optimal solution')
    plt.yscale('log')
    plt.legend()
    print('Finish')
    plt.show()


if __name__ == "__main__":
    figure8_7_multiprocessing()