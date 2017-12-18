from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from chapter08.Maze import Maze, convert_maze_to_grid, draw_grid
from chapter08.Dyna import Dyna


def main():
    # hyper-parameters for Dyna models
    rand = np.random.RandomState(0)
    epsilon = 0.1
    planningSteps = [0, 5, 10]
    #planningSteps = [0, 5]
    linestyles = ['-', '--', '-.', ':']
    gamma = 0.95
    alpha = 0.5
    theta = 1e-4

    plt.figure()

    # wind strength for each column
    stochastic_wind_direction = 0   # 'UP'
    stochastic_wind = np.asarray([[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]] * 7)

    # set up an instance for DynaMaze
    maze = Maze(height=7,
                width=10,
                start_state=[3, 0],
                goal_states=[[3, 7]],
                stochastic_wind=stochastic_wind,
                stochastic_wind_direction=stochastic_wind_direction)

    for p, ps in enumerate(planningSteps):
        model_Dyna_PS = Dyna(rand=rand,
                             maze=maze,
                             epsilon=epsilon,
                             gamma=gamma,
                             planningSteps=ps,
                             qLearning=True,
                             expected=False,
                             alpha=alpha,
                             priority=True,
                             theta=theta)

        model_Dyna_Q = Dyna(rand=rand,
                          maze=maze,
                          epsilon=epsilon,
                          gamma=gamma,
                          planningSteps=ps,
                          qLearning=True,
                          expected=False,
                          plus=False,
                          alpha=alpha)

        model_Dyna_Sarsa = Dyna(rand=rand,
                          maze=maze,
                          epsilon=epsilon,
                          gamma=gamma,
                          planningSteps=ps,
                          qLearning=False,
                          expected=False,
                          plus=False,
                          alpha=alpha)

        model_Dyna_Sarsa_Expected = Dyna(rand=rand,
                          maze=maze,
                          epsilon=epsilon,
                          gamma=gamma,
                          planningSteps=ps,
                          qLearning=False,
                          expected=True,
                          plus=False,
                          alpha=alpha)

        if ps == 0:
            models = [model_Dyna_Sarsa, model_Dyna_Sarsa_Expected, model_Dyna_Q]
        else:
            models = [model_Dyna_Sarsa, model_Dyna_Sarsa_Expected, model_Dyna_Q]

        for m, model in enumerate(models):
            # track steps / backups for each episode
            print('planning step:', ps, 'model: ', model.name)

            episodes = []
            ep = 0
            episodeLimit = 100
            # play for an episode
            while ep < episodeLimit:
                time = model.play(environ_step=True)
                episodes.extend([ep] * time)
                ep += 1

            plt.plot(episodes, label=model.full_name,  linestyle=linestyles[p])
            plt.xlabel('Time steps')
            plt.ylabel('Episodes')
            plt.legend()
    plt.show()


if __name__ == "__main__":
    main()