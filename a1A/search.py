"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))


    from util import Stack
    from game import Directions
    states = Stack()
    # 标记已经搜索过的状态集合
    havesearched = []
    states.push((problem.getStartState(), []))
    # # 循环终止条件：遍历完毕/目标测试成功
    while not states.isEmpty():
        cur_node, actions = states.pop()
        if problem.isGoalState(cur_node):
            return actions
        if cur_node not in havesearched:
            expand = problem.getSuccessors(cur_node)
            havesearched.append(cur_node)
            for location, direction, cost in expand:
                # 判断状态是否重复
                if (location not in havesearched):
                    states.push((location, actions + [direction]))
    util.raiseNotDefined()





def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"

    from util import Queue
    from game import Directions
    states = Queue()
    havesearched = []
    states.push((problem.getStartState(), []))
    while not states.isEmpty():
        cur_node, actions = states.pop()
        if problem.isGoalState(cur_node):
            return actions
        if cur_node not in havesearched:
            expand = problem.getSuccessors(cur_node)
            havesearched.append(cur_node)
            for location, direction, cost in expand:
                if (location not in havesearched):
                    states.push((location, actions + [direction]))
    util.raiseNotDefined()




def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

    start_point = problem.getStartState()
    queue = util.PriorityQueueWithFunction(lambda x: x[2])
    queue.push((start_point, None, 0))
    cost = 0
    visited = []
    path = []
    parentSeq = {}
    parentSeq[(start_point, None, 0)] = None
    while queue.isEmpty() == False:
        current_fullstate = queue.pop()
        # print current_fullstate
        if (problem.isGoalState(current_fullstate[0])):
            break
        else:
            current_state = current_fullstate[0]
            if current_state not in visited:
                visited.append(current_state)
            else:
                continue
            successors = problem.getSuccessors(current_state)
            for state in successors:
                cost = current_fullstate[2] + state[2];
                # print state,cost
                if state[0] not in visited:
                    queue.push((state[0], state[1], cost))
                    # parentSeq[state] = current_fullstate
                    parentSeq[(state[0], state[1])] = current_fullstate

    child = current_fullstate

    while (child != None):
        path.append(child[1])
        if child[0] != start_point:
            child = parentSeq[(child[0], child[1])]
        else:
            child = None
    path.reverse()

    return path[1:]

    # util.raiseNotDefined()



def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    # from sets import Set

    states = util.PriorityQueue()
    actions = []
    states.push((problem.getStartState(), actions), 0)
    visited = []
    tmpActions = []

    while states:
        currState, actions = states.pop()
        if problem.isGoalState(currState):
            break
        if currState not in visited:
            visited.append(currState)
            successors = problem.getSuccessors(currState)
            for successor, action, cost in successors:
                tempActions = actions + [action]
                nextCost = problem.getCostOfActions(tempActions) + heuristic(successor, problem)
                if successor not in visited:
                    states.push((successor, tempActions), nextCost)

    return actions

    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
