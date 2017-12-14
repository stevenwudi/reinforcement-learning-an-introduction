import numpy as np


# model for planning in Dyna-Q
class Dyna:

    # @rand: an instance of np.random.RandomState for sampling
    def __init__(self,
                 rand,
                 maze,
                 gamma=0.95,
                 planningSteps=5,
                 expected=False,
                 qLearning=False,
                 alpha=0.1,
                 plus=False,
                 kappa=1e-4):
        """

        :param rand:
        :param maze:
        :param planningSteps:
        :param expected:
        :param qLearning:
        :param alpha:
        :param kappa:
        """
        self.model = dict()
        self.rand = rand
        self.maze = maze
        self.stateActionValues = np.zeros((maze.WORLD_HEIGHT, maze.WORLD_WIDTH, len(maze.actions)))

        # expected sarsa
        self.expected = expected

        # Q-learning
        self.qlearning = qLearning
        if self.qlearning:
            self.name = 'Dyna-Q'
        elif self.expected:
            self.name = 'DynaExpectedSarsa'
        else:
            self.name = 'Dyna-Sarsa'

        # discount
        self.gamma = gamma
        # probability for exploration
        self.epsilon = 0.1
        # step size
        self.alpha = alpha
        # weight for elapsed time
        self.timeWeight = 0
        # n-step planning
        self.planningSteps = planningSteps
        # average over several independent runs
        self.runs = 10
        # algorithm names
        self.methods = ['Dyna-Q', 'Dyna-Q+']
        # threshold for priority queue
        self.theta = 0

        # Dyna Plus algorithm
        self.plus = plus  # flag for Dyna+ algorithm
        if self.plus:
            self.name += '_plus'
        self.kappa = kappa  # Time weight
        self.time = 0  # track the total time

    #
    def feed(self, currentState, action, newState, reward):
        """"
        feed the model with previous experience
        """
        if self.plus:
            self.time += 1
            if tuple(currentState) not in self.model.keys():
                self.model[tuple(currentState)] = dict()
                # Actions that had never been tried before from a state were allowed to be
                # considered in the planning step
                for action_ in self.maze.actions:
                    if action_ != action:
                        # Such actions would lead back to the same state with a reward of zero
                        # Notice that the minimum time stamp is 1 omstead pf 0
                        self.model[tuple(currentState)][action_] = [list(currentState), 0, 1]

            self.model[tuple(currentState)][action] = [list(newState), reward, self.time]

        else:
            if tuple(currentState) not in self.model.keys():
                self.model[tuple(currentState)] = dict()
            self.model[tuple(currentState)][action] = [list(newState), reward]

    # randomly sample from previous experience
    def sample(self):
        stateIndex = self.rand.choice(range(0, len(self.model.keys())))
        state = list(self.model)[stateIndex]
        actionIndex = self.rand.choice(range(0, len(self.model[state].keys())))
        action = list(self.model[state])[actionIndex]
        if self.plus:
            newState, reward, time = self.model[state][action]
            # adjust reward with elapsed time since last visit
            reward += self.kappa * np.sqrt(self.time - time)
        else:
            newState, reward = self.model[state][action]

        return list(state), action, list(newState), reward

    # choose an action based on epsilon-greedy algorithm
    def chooseAction(self, state):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(self.maze.actions)
        else:
            values = self.stateActionValues[state[0], state[1], :]
            return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])


    # play for an episode for Dyna-Q algorithm
    # @stateActionValues: state action pair values, will be updated
    # @model: model instance for planning
    # @planningSteps: steps for planning
    # @maze: a maze instance containing all information about the environment
    # @dynaParams: several params for the algorithm
    def play(self):

        currentState = self.maze.START_STATE
        steps = 0
        while currentState not in self.maze.GOAL_STATES:
            # track the steps
            steps += 1
            # get action
            if steps == 1 or self.qlearning:
                currentAction = self.chooseAction(currentState)
            # take action
            newState, reward = self.maze.takeAction(currentState, currentAction)

            if self.qlearning:
                # Q-Learning update
                self.stateActionValues[currentState[0], currentState[1], currentAction] += \
                    self.alpha * (reward + self.gamma * np.max(self.stateActionValues[newState[0], newState[1], :]) -
                                        self.stateActionValues[currentState[0], currentState[1], currentAction])
            else:
                # sarsa or expected sarsa update
                if not self.expected:
                    newAction = self.chooseAction(newState)
                    valueTarget = self.stateActionValues[newState[0], newState[1], newAction]
                elif self.expected:
                    # calculate the expected value of new state
                    valueTarget = 0.0
                    actionValues = self.stateActionValues[newState[0], newState[1], :]
                    bestActions = np.argwhere(actionValues == np.max(actionValues))
                    for action in self.maze.actions:
                        if action in bestActions:
                            valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(self.maze.actions)) \
                                           * self.stateActionValues[newState[0], newState[1], action]
                        else:
                            valueTarget += self.epsilon / len(self.maze.actions) * self.stateActionValues[newState[0], newState[1], action]
                # Sarsa update
                self.stateActionValues[currentState[0], currentState[1], currentAction] += \
                    self.alpha * (reward + self.gamma * valueTarget - self.stateActionValues[currentState[0], currentState[1], currentAction])
            # feed the model with experience
            self.feed(currentState, currentAction, newState, reward)
            if not self.qlearning:
                currentAction = newAction
            currentState = newState
            # sample experience from the model
            for t in range(0, self.planningSteps):
                stateSample, actionSample, newStateSample, rewardSample = self.sample()
                if self.qlearning:
                    self.stateActionValues[stateSample[0], stateSample[1], actionSample] += \
                        self.alpha * (rewardSample + self.gamma * np.max(
                            self.stateActionValues[newStateSample[0], newStateSample[1], :]) -
                                      self.stateActionValues[stateSample[0], stateSample[1], actionSample])
                else:
                    # sarsa or expected sarsa update
                    if not self.expected:
                        newAction_sample = self.chooseAction(newStateSample)
                        valueTarget = self.stateActionValues[newStateSample[0], newStateSample[1], newAction_sample]
                    elif self.expected:
                        # calculate the expected value of new state
                        valueTarget = 0.0
                        actionValues = self.stateActionValues[newStateSample[0], newStateSample[1], :]
                        bestActions = np.argwhere(actionValues == np.max(actionValues))
                        for action in self.maze.actions:
                            if action in bestActions:
                                valueTarget += ((1.0 - self.epsilon) / len(bestActions) + self.epsilon / len(
                                    self.maze.actions)) \
                                               * self.stateActionValues[newStateSample[0], newStateSample[1], action]
                            else:
                                valueTarget += self.epsilon / len(self.maze.actions) * self.stateActionValues[
                                    newStateSample[0], newStateSample[1], action]
                                # Sarsa update
                    self.stateActionValues[stateSample[0], stateSample[1], actionSample] += \
                        self.alpha * (rewardSample + self.gamma * valueTarget -
                                      self.stateActionValues[stateSample[0], stateSample[1], actionSample])

            # check whether it has exceeded the step limit
            if steps > self.maze.maxSteps:
                print(currentState)
                break

        return steps

    def walk_final_grid(self):

        came_from = {}
        cost_so_far = {}
        # came_from[tuple([maze.START_STATE[1], maze.START_STATE[0]])] = None
        # cost_so_far[tuple([maze.START_STATE[1], maze.START_STATE[0]])] = 0
        came_from[tuple(self.maze.START_STATE)] = None
        cost_so_far[tuple(self.maze.START_STATE)] = 0
        # we use the maximum here directly
        self.epsilon = 0

        currentState = self.maze.START_STATE
        steps = 0
        while currentState not in self.maze.GOAL_STATES:
            # track the steps
            steps += 1

            # get action
            action = self.chooseAction(currentState)
            # take action
            newState, reward = self.maze.takeAction(currentState, action)
            cost_so_far[tuple(newState)] = reward
            came_from[tuple(newState)] = tuple(currentState)
            currentState = newState

            # check whether it has exceeded the step limit
            if steps > self.maze.maxSteps:
                print(currentState)
                break

        return came_from, cost_so_far
