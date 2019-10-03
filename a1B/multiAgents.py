from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # return successorGameState.getScore()
        # 获取当前游戏状态 其中G表示为Ghost %表示为墙 角标表示pacman 角标方向代表上一次选择的方向
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print 'successorGameState\n',successorGameState

        # 获取这样移动后新的位置
        newPos = successorGameState.getPacmanPosition()
        # print 'newPos',newPos

        # 获取食物在图中的分布（二维数组，有失误为T没食物为F)
        newFood = successorGameState.getFood()
        curFood = currentGameState.getFood()
        # print 'newFood',newFood

        # 获取Ghost的位置
        newGhostStates = successorGameState.getGhostStates()
        # print 'ghostState',newGhostStates[0].getPosition()

        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # 对这个选择评估的分数
        currscore = 0

        if action == "Stop":
            return -100

        # 如果当前状态能够使ghost害怕，将所有的时间加入进来
        for st in newScaredTimes:
            currscore += st

        # 根据Ghost所在的位置，获取与当前位置的距离
        ghost_distances = []
        for gs in newGhostStates:
            ghost_distances += [manhattanDistance(gs.getPosition(), newPos)]

        # 获取food所在的所有pos
        foodList = newFood.asList()
        curfoodList = curFood.asList()

        # 获取food所在的所有wall
        wallList = currentGameState.getWalls().asList()

        # 保存food的距离
        food_distences = []

        # 获取所有食物到达当前位置的距离
        for foodpos in foodList:
            food_distences += [manhattanDistance(newPos, foodpos)]

        # 对食物的距离取反
        inverse_food_distences = 0;
        if len(food_distences) > 0 and min(food_distences) > 0:
            inverse_food_distences = 1.0 / min(food_distences)
        # 考虑了ghost与当前的距离，其权值更大
        currscore += min(ghost_distances) * (inverse_food_distences ** 4)
        # 获取当前系统判定的分数 又可能当前吃到了豆子 分数更高些
        currscore += successorGameState.getScore()
        if newPos in curfoodList:
            currscore = currscore * 1.1
        return currscore


'''-------------------------use BFS to find maze path between pacman and food-------------------'''
from game import Actions


# position search problem
class PositionSearchProblem():
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.
    The state space consists of (x,y) positions in a pacman game.
    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.
        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


# BFS GSA
def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    startState = problem.getStartState()
    explored = set()
    Frontier = util.Queue()
    Frontier.push([[startState, None, 0]])
    while not Frontier.isEmpty():
        StateTriples = Frontier.pop()
        node = StateTriples[-1][0]
        if problem.isGoalState(node):
            solution = []
            for i in StateTriples[1:]:
                solution = solution + [i[1]]
            return solution
        if node not in explored:
            explored.add(node)
            for i in problem.getSuccessors(node):
                Frontier.push(StateTriples + [list(i)])
    print(Frontier.isEmpty())
    util.raiseNotDefined()


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.
    Example usage: mazeDistance( (2,4), (5,6), gameState)
    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = int(point1[0]), int(point1[1])
    x2, y2 = int(point2[0]), int(point2[1])
    walls = gameState.getWalls()

    assert not walls[x1][y1], 'point1 is a wall: ' + point1
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))


'''------------------------------------------end of using BFS---------------------------------'''
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # the return value of "result" function is (action, score) pair
        a = self.result(gameState, 0)[0]
        return a
        util.raiseNotDefined()

    def result(self, gameState, currentDepth):
        # for every layer we need to do mini-max
        # calculate the number of agents
        agentNum = gameState.getNumAgents()
        # a list used to find biggest or smallest score
        scores = []
        # when it is a terminal state
        if gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        # when it reaches the biggest depth
        if currentDepth == self.depth * agentNum:
            return (None, self.evaluationFunction(gameState))
        elif currentDepth < self.depth * agentNum:
            # find whose turn to move
            agentIndex = currentDepth % agentNum
            # Collect legal moves and successor states
            legalMoves = gameState.getLegalActions(agentIndex)
            # Choose one of the best actions
            for action in legalMoves:
                new_GameState = gameState.generateSuccessor(agentIndex, action)
                result = self.result(new_GameState, currentDepth + 1)[1]
                scores.append(result)
            if agentIndex == 0:
                bestScore = max(scores)
            else:
                bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
            return (legalMoves[chosenIndex], bestScore)
        else:
            print("depth is out of self.depth")
        # util.raiseNotDefined()
        # def max_value(state, currentDepth):
        #     # 当前深度加一
        #     currentDepth = currentDepth + 1
        #     # 若当前状态已经赢了或输了 或者 已经到达了规定的深度
        #     if state.isWin() or state.isLose() or currentDepth == self.depth:
        #         return self.evaluationFunction(state)
        #     # 初始化v
        #     v = float('-Inf')
        #     # 对每个min分支求max
        #     for pAction in state.getLegalActions(0):
        #         v = max(v, min_value(state.generateSuccessor(0, pAction), currentDepth, 1))
        #     return v
        #
        # def min_value(state, currentDepth, ghostNum):
        #     # 若当前状态已经赢了或输了
        #     if state.isWin() or state.isLose():
        #         return self.evaluationFunction(state)
        #     # 初始化v
        #     v = float('Inf')
        #     # 对每个max分支求min 其中有多个Ghost 所有多个Ghost分支
        #     for pAction in state.getLegalActions(ghostNum):
        #         if ghostNum == gameState.getNumAgents() - 1:
        #             # 所有Ghost的min找完了 开始找下一个max
        #             v = min(v, max_value(state.generateSuccessor(ghostNum, pAction), currentDepth))
        #         else:
        #             # 继续下一个Ghost
        #             v = min(v, min_value(state.generateSuccessor(ghostNum, pAction), currentDepth, ghostNum + 1))
        #     return v
        #
        # # pacman下一个状态可能的行动
        # Pacman_Actions = gameState.getLegalActions(0)
        #
        # maximum = float('-Inf')
        # result = ''
        #
        # # 针对下一个状态 寻找获胜概率最高的move
        # for action in Pacman_Actions:
        #     if (action != "Stop"):
        #         currentDepth = 0
        #         # 而所有的Ghost希望胜利概率最低的选择
        #         currentMax = min_value(gameState.generateSuccessor(0, action), currentDepth, 1)
        #         if currentMax > maximum:
        #             maximum = currentMax
        #             result = action
        # return result

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a = self.result(gameState, 0)[0]
        return a
        util.raiseNotDefined()

    def result(self, gameState, currentDepth=0, alpha=-10000, beta=10000):
        # for every layer we need to do mini-max
        agentNum = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose() or \
                (currentDepth == self.depth * agentNum):
            evaluation = self.evaluationFunction(gameState)
            # print((gameState.isWin() or gameState.isLose()),currentDepth)
            return (None, evaluation)
        elif currentDepth < self.depth * agentNum:
            # find whose turn to move
            agentIndex = currentDepth % agentNum
            # Collect legal moves and successor states
            legalMoves = gameState.getLegalActions(agentIndex)
            # Choose one of the best actions
            # print(legalMoves, currentDepth,self.depth * agentNum)
            maxEvaluation = -10000
            minEvaluation = 10000
            for action in legalMoves:
                new_GameState = gameState.generateSuccessor(agentIndex, action)
                result = self.result(new_GameState, currentDepth + 1, alpha, beta)[1]
                # print(result,currentDepth,alpha,beta)
                if agentIndex == 0 and result >= maxEvaluation:
                    maxEvaluation = result
                    bestAction = action
                    if result > beta:
                        return (bestAction, maxEvaluation)
                    alpha = max(alpha, maxEvaluation)
                elif agentIndex > 0 and result <= minEvaluation:
                    minEvaluation = result
                    bestAction = action
                    if result < alpha:
                        return (bestAction, minEvaluation)
                    beta = min(result, minEvaluation)
            # print(currentDepth, bestAction)
            if agentIndex == 0:
                return (bestAction, maxEvaluation)
            else:
                return (bestAction, minEvaluation)
        else:
            print("depth is out of self.depth")
        # util.raiseNotDefined()
        # def max_value(state, alpha, beta, currentDepth):
        #     # 当前深度加一
        #     currentDepth = currentDepth + 1
        #     # 若当前状态已经赢了或输了 或者 已经到达了规定的深度
        #     if state.isWin() or state.isLose() or currentDepth == self.depth:
        #         return self.evaluationFunction(state)
        #     v = float('-Inf')
        #     # 对每个min分支求max
        #     for pAction in state.getLegalActions(0):
        #         if pAction != "Stop":
        #             v = max(v, min_value(state.generateSuccessor(0, pAction), alpha, beta, currentDepth, 1))
        #             # 若已经比beta要大了 就没有搜索下去的必要了
        #             if v >= beta:
        #                 return v
        #             # 更新alpha的值
        #             alpha = max(alpha, v)
        #     return v
        #
        # def min_value(state, alpha, beta, currentDepth, ghostNum):
        #     # 若当前状态已经赢了或输了
        #     if state.isWin() or state.isLose():
        #         return self.evaluationFunction(state)
        #     # 初始化v
        #     v = float('Inf')
        #     # 对每个max分支求min 其中有多个Ghost 所有多个Ghost分支
        #     for pAction in state.getLegalActions(ghostNum):
        #         if ghostNum == gameState.getNumAgents() - 1:
        #             # 所有Ghost的min找完了 开始找下一个max
        #             v = min(v, max_value(state.generateSuccessor(ghostNum, pAction), alpha, beta, currentDepth))
        #         else:
        #             # 继续下一个Ghost
        #             v = min(v,
        #                     min_value(state.generateSuccessor(ghostNum, pAction), alpha, beta, currentDepth,
        #                               ghostNum + 1))
        #         # 若比alpha还要小了 就没搜索的必要了
        #         if v <= alpha:
        #             return v
        #         # 更新beta的值
        #         beta = min(beta, v)
        #     return v
        #
        # # pacman下一个状态可能的行动
        # pacmanActions = gameState.getLegalActions(0)
        # maximum = float('-Inf')
        # # 初始化alpha bate
        # alpha = float('-Inf')
        # beta = float('Inf')
        # maxAction = ''
        #
        # # 针对下一个状态 寻找获胜概率最高的move
        # for action in pacmanActions:
        #     if action != "Stop":
        #         currentDepth = 0
        #         # 而所有的Ghost希望胜利概率最低的选择
        #         currentMax = min_value(gameState.generateSuccessor(0, action), alpha, beta, currentDepth, 1)
        #         if currentMax > maximum:
        #             maximum = currentMax
        #             maxAction = action
        # # print
        # maximum
        # return maxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        a = self.result(gameState, 0)[0]
        return a
        util.raiseNotDefined()

    def result(self, gameState, currentDepth):
        # for every layer we need to do mini-max
        agentNum = gameState.getNumAgents()
        scores = []
        if gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        if currentDepth == self.depth * agentNum:
            return (None, self.evaluationFunction(gameState))
        elif currentDepth < self.depth * agentNum:
            # find whose turn to move
            agentIndex = currentDepth % agentNum
            # Collect legal moves and successor states
            legalMoves = gameState.getLegalActions(agentIndex)
            # Choose one of the best actions
            for action in legalMoves:
                new_GameState = gameState.generateSuccessor(agentIndex, action)
                result = self.result(new_GameState, currentDepth + 1)[1]
                scores.append(result)
            if agentIndex == 0:
                bestScore = max(scores)
                bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
                chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
                return (legalMoves[chosenIndex], bestScore)
            else:
                bestScore = sum(scores) / len(scores)
                return (None, bestScore)
        else:
            print("depth is out of self.depth")

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # util.raiseNotDefined()
    # the number of food in successorGameState
    numFood = newFood.count()
    # print([newGhostStates[i].configuration.pos for i in range(len(newGhostStates))])
    # the effect of the distance between pacman and normal ghost, scared ghost
    minPacGhost = newFood.height + newFood.width
    eatScaredChance_Max = 0
    eatableGhost = None
    for i in range(len(newGhostStates)):
        pacGhostDis = manhattanDistance(newPos, newGhostStates[i].configuration.pos)
        eatScaredChance = max([0, newGhostStates[i].scaredTimer - pacGhostDis])
        if pacGhostDis < minPacGhost and newGhostStates[i].scaredTimer == 0:
            minPacGhost = pacGhostDis
        if eatScaredChance > eatScaredChance_Max:
            eatScaredChance_Max = eatScaredChance
            eatableGhost = newGhostStates[i].configuration.pos

    # print((eatScaredChance_Max,newPos,eatableGhost))
    # the secure distance is 3, after both pacman and ghost make move, pacman is still safe
    # find the most dangerous distance
    pac_Ghost_Distance = min([minPacGhost, 3])
    # find the closet food to pacman
    minDistance = newFood.height * newFood.width
    manhattan_PriorityQueue = util.PriorityQueue()
    for y in range(newFood.height):
        for x in range(newFood.width):
            if newFood[x][y] == True:
                manhattan_PriorityQueue.push((x, y), manhattanDistance(newPos, (x, y)))
    for i in range(5):
        if manhattan_PriorityQueue.isEmpty():
            break
        else:
            maze = mazeDistance(newPos, manhattan_PriorityQueue.pop(), currentGameState)
            if maze < minDistance:
                minDistance = maze
    # calculate the evaluation value
    evaluation = currentGameState.getScore() + 200 * eatScaredChance_Max + 500 * pac_Ghost_Distance + (
                1 / (minDistance)) \
                 / max([numFood, 1])
    # print(evaluation)
    return evaluation

# Abbreviation
better = betterEvaluationFunction

