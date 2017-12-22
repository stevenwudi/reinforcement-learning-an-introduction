import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from chapter08.Maze import Maze, Maze_3D
from chapter08.Dyna import Dyna
from chapter08.Dyna_3D import Dyna_3D
from timeit import default_timer as timer
from chapter08.Astar import a_star_search_3D, convert_maze_to_grid, draw_grid, \
    walk_final_grid_go_to, convert_3Dmaze_to_grid, draw_grid_3d


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
                steps += model.play(environ_step=True)
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
    maxSteps = 8000
    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    blockingMaze.changingPoint = 3000
    blockingMaze.maxSteps = maxSteps

    rand = np.random.RandomState(0)
    # generate an instance of Dyna-Q model
    # track the cumulative rewards

    runs = 3
    planningSteps = 5
    alpha = 0.5
    gamma = 0.95
    kappa = 1e-3
    theta = 1e-4
    ucb = 1e-2
    model_num = 4
    # track cumulative reward in current run
    rewards = np.zeros((model_num, maxSteps))

    for run in range(0, runs):
        # set up models
        rewards_ = np.zeros(shape=(model_num, maxSteps))
        model_Dyna = Dyna(rand=rand,
                          maze=blockingMaze,
                          planningSteps=planningSteps,
                          qLearning=True,
                          expected=False,
                          gamma=gamma,
                          alpha=alpha)

        model_Dyna_plus = Dyna(rand=rand,
                               maze=blockingMaze,
                               planningSteps=planningSteps,
                               qLearning=True,
                               expected=False,
                               alpha=alpha,
                               plus=True,
                               gamma=gamma,
                               kappa=kappa)

        model_Dyna_plus_PS = Dyna(rand=rand,
                               maze=blockingMaze,
                               planningSteps=planningSteps,
                               qLearning=True,
                               expected=False,
                               alpha=alpha,
                               plus=True,
                               gamma=gamma,
                               kappa=kappa*100,
                               priority=True,
                               theta=theta)

        model_Dyna_UCB = Dyna(rand=rand,
                               maze=blockingMaze,
                               planningSteps=planningSteps,
                               qLearning=True,
                               expected=False,
                               alpha=alpha,
                               ucb=ucb,
                               gamma=gamma,)

        models = [model_Dyna_UCB, model_Dyna, model_Dyna_plus, model_Dyna_plus_PS]
        for i, model in enumerate(models):
            print('run:', run, model.name)

            # set old obstacles for the maze
            model.maze.obstacles = blockingMaze.oldObstacles
            steps = 0
            lastSteps = steps
            while steps < maxSteps:
                # play for an episode
                step_eps = model.play(environ_step=True)
                print(step_eps)
                steps += step_eps
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
                        return_to_start=True,
                        reward_goal=1.0,
                        reward_move=0.0,
                        reward_obstacle=-1.0
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
    original_maze = Maze(height=6,
                         width=9,
                         start_state=[2, 0],
                         goal_states=[[0, 8]],
                         return_to_start=True,
                         reward_goal=0.0,
                         reward_move=-1.0,
                         reward_obstacle=-100.0)
    original_maze.obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
    resize_idx = 2
    # say 1st maze has w * h states, then k-th maze has w * h * k * k states
    # build all the mazes
    mazes = [original_maze.extendMaze(i) for i in range(resize_idx, resize_idx + 1)]
    # Dyna model hyper
    rand = np.random.RandomState(0)
    planningSteps = 5
    alpha = 0.5
    gamma = 0.95
    theta = 1e-4
    epsilon = 0.1
    # track the # of backups

    pool = Pool(processes=1)
    runs = 1
    numOfMazes = 1
    numOfModels = 4
    backups = np.zeros((runs, numOfModels, numOfMazes))

    for mazeIndex, maze in enumerate(mazes):
        maze.GOAL_STATES = [maze.GOAL_STATES[0]]
        ############## A star search algorithm  ######################################
        start_time = timer()
        diagram = convert_maze_to_grid(maze)
        came_from, cost_so_far, go_to = a_star_search_3D(diagram, tuple(maze.START_STATE),
                                                  [tuple(maze.GOAL_STATES[0])])
        go_to, step_optim = walk_final_grid_go_to(tuple(maze.START_STATE), tuple(maze.GOAL_STATES[0]), came_from)
        draw_grid(diagram, width=3, point_to=came_from, start=tuple(maze.START_STATE),
                  goal=tuple(maze.GOAL_STATES[0]))
        print('Finish, using %.2f sec!' % (timer() - start_time))
        ##############End  A star search algorithm  ######################################
        model_Dyna_Prioritized_Sweeping_ES = Dyna(rand=rand,
                                         maze=maze,
                                         epsilon=epsilon,
                                         gamma=gamma,
                                         planningSteps=planningSteps,
                                         qLearning=False,
                                         expected=True,
                                         alpha=alpha,
                                         priority=True,
                                         theta=theta)

        model_Dyna_Prioritized_Sweeping_Q = Dyna(rand=rand,
                                         maze=maze,
                                         epsilon=epsilon,
                                         gamma=gamma,
                                         planningSteps=planningSteps,
                                         qLearning=True,
                                         expected=False,
                                         alpha=alpha,
                                         priority=True,
                                         theta=theta)

        model_Dyna_Prioritized_Sweeping_A_star_ES = Dyna(rand=rand,
                                                     maze=maze,
                                                     epsilon=epsilon,
                                                     gamma=gamma,
                                                     planningSteps=planningSteps,
                                                     qLearning=False,
                                                     expected=True,
                                                     alpha=alpha,
                                                     priority=True,
                                                     theta=theta,
                                                     policy_init=go_to)

        model_Dyna_Prioritized_Sweeping_A_star_Q = Dyna(rand=rand,
                                                     maze=maze,
                                                     epsilon=epsilon,
                                                     gamma=gamma,
                                                     planningSteps=planningSteps,
                                                     qLearning=True,
                                                     expected=False,
                                                     alpha=alpha,
                                                     priority=True,
                                                     theta=theta,
                                                     policy_init=go_to)

        models = [model_Dyna_Prioritized_Sweeping_A_star_Q, model_Dyna_Prioritized_Sweeping_Q,
                  model_Dyna_Prioritized_Sweeping_A_star_ES, model_Dyna_Prioritized_Sweeping_ES]

        for m, model in enumerate(models):
            for run in range(0, runs):
                print('run:', run, model.name, 'maze size:', maze.WORLD_HEIGHT * maze.WORLD_WIDTH)
                start_time = timer()
                result = pool.apply_async(model_play_worker, [model])
                backups[run][m][mazeIndex] = np.sum(result.get())
                print('Finish, using %.2f sec!' % (timer() - start_time))

    pool.close()

    # Dyna-Q performs several backups per step
    backups = np.sum(backups, axis=0)
    # average over independent runs
    backups /= float(runs)

    print(backups)


def figure8_7_3D():
    # get the original 6*9 maze
    time_length = 20
    maze = Maze_3D(height=7,
                width=9,
                time_length=time_length,
                start_state=(2, 0, 0),
                goal_states=[[0, 8]],
                return_to_start=True,
                reward_goal=1.0,
                reward_move=0.0,
                reward_obstacle=0.0)
    # set up goal states
    gs = []
    for t in range(time_length):
        gs.append(tuple([0, 8, t]))
    maze.GOAL_STATES = gs
    # set up obstacles
    obstacles = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
    for t in range(time_length-5):
        for item in obstacles:
            maze.obstacles_weight[item[0], item[1], t] = 1
    obstacles = [[2, 4], [3, 4], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]
    for t in range(5, time_length):
        for item in obstacles:
            maze.obstacles_weight[item[0], item[1], t] = 1
    mazes = [maze]
    # Dyna model hyper
    rand = np.random.RandomState(0)
    planningSteps = 5
    alpha = 0.5
    gamma = 0.95
    theta = 1e-4
    epsilon = 0.1
    # track the # of backups
    runs = 1
    numOfMazes = 1
    numOfModels = 3
    backups = np.zeros((runs, numOfModels, numOfMazes))

    for mazeIndex, maze in enumerate(mazes):
        ############## A star search algorithm  ######################################
        start_time = timer()
        diagram = convert_3Dmaze_to_grid(maze)
        came_from, cost_so_far, go_to = a_star_search_3D(diagram, tuple(maze.START_STATE), maze.GOAL_STATES)
        if True:
            draw_grid_3d(diagram, came_from=came_from, start=tuple(maze.START_STATE),
                         goal=tuple(maze.GOAL_STATES), title='A star')
        print('Finish, using %.2f sec!' % (timer() - start_time))
        ##############End  A star search algorithm  ######################################
        model_Dyna_Q = Dyna_3D(rand=rand,
                             maze=maze,
                             epsilon=epsilon,
                             gamma=gamma,
                             planningSteps=planningSteps,
                             qLearning=True,
                             expected=False,
                             alpha=alpha,
                             priority=False)

        model_Dyna_Prioritized_Sweeping_ES = Dyna_3D(rand=rand,
                                         maze=maze,
                                         epsilon=epsilon,
                                         gamma=gamma,
                                         planningSteps=planningSteps,
                                         qLearning=False,
                                         expected=True,
                                         alpha=alpha,
                                         priority=True,
                                         theta=theta)

        model_Dyna_Prioritized_Sweeping_Q = Dyna_3D(rand=rand,
                                         maze=maze,
                                         epsilon=epsilon,
                                         gamma=gamma,
                                         planningSteps=planningSteps,
                                         qLearning=True,
                                         expected=False,
                                         alpha=alpha,
                                         priority=True,
                                         theta=theta)

        models = [model_Dyna_Q, model_Dyna_Prioritized_Sweeping_Q, model_Dyna_Prioritized_Sweeping_ES]

        for m, model in enumerate(models):
            for run in range(0, runs):
                print('run:', run, model.name, 'maze size:', maze.WORLD_HEIGHT * maze.WORLD_WIDTH)
                start_time = timer()
                # track steps / backups for each episode
                steps = []
                # play for an episode
                while True:
                    steps.append(model.play())
                    # print best action w.r.t. current state-action values
                    # printActions(currentStateActionValues, maze)
                    # check whether the (relaxed) optimal path is found
                    came_from = model.checkPath()
                    if came_from:
                        draw_grid_3d(diagram, came_from=came_from, start=tuple(maze.START_STATE),
                                     goal=tuple(maze.GOAL_STATES), title=model.name)

                        break

                backups[run][m][mazeIndex] = np.sum(steps)
                print('Finish, using %.2f sec!' % (timer() - start_time))

    # Dyna-Q performs several backups per step
    backups = np.sum(backups, axis=0)
    # average over independent runs
    backups /= float(runs)
    print(backups)
    plt.show()


if __name__ == "__main__":
    figure8_7_3D()