import numpy as np
import matplotlib.pyplot as plt
from chapter08.Maze import Maze, convert_maze_to_grid, draw_grid
from chapter08.Dyna import Dyna
from chapter08.Astar import a_star_search


def main():
    # set up an instance for DynaMaze

    dynaMaze = Maze()
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




if __name__ == "__main__":
    main()