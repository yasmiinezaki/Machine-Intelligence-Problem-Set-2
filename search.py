from typing import Tuple
from game import HeuristicFunction, Game, S, A
from helpers.utils import NotImplemented

#TODO: Import any modules you want to use
import math

# All search functions take a problem, a state, a heuristic function and the maximum search depth.
# If the maximum search depth is -1, then there should be no depth cutoff (The expansion should not stop before reaching a terminal state) 

# All the search functions should return the expected tree value and the best action to take based on the search results

# This is a simple search function that looks 1-step ahead and returns the action that lead to highest heuristic value.
# This algorithm is bad if the heuristic function is weak. That is why we use minimax search to look ahead for many steps.
def greedy(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)
    
    terminal, values = game.is_terminal(state)
    if terminal: return values[agent], None

    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    value, _, action = max((heuristic(game, state, agent), -index, action) for index, (action , state) in enumerate(actions_states))
    return value, action

# Apply Minimax search and return the game tree value and the best action
# Hint: There may be more than one player, and in all the testcases, it is guaranteed that 
# game.get_turn(state) will return 0 (which means it is the turn of the player). All the other players
# (turn > 0) will be enemies. So for any state "s", if the game.get_turn(s) == 0, it should a max node,
# and if it is > 0, it should be a min node. Also remember that game.is_terminal(s), returns the values
# for all the agents. So to get the value for the player (which acts at the max nodes), you need to
# get values[0].
def minimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state) # get turn of the current player

    terminal, values = game.is_terminal(state) 
    if terminal: return (values[agent],None) # check if terminal node and return terminal value of agent
    
    if (max_depth==0): # depth end reached
        return  (heuristic(game,state,0),None) # if depth limited and reached max depth return heuristic value
    
    if (agent == 0): # in max node -> max players turn
        best_value = -math.inf # initialize best value and best action which will be our retun values later
        best_action = None
        for action in game.get_actions(state): # loop through list of possible actions from left of tree to right of tree
            value = minimax(game,game.get_successor(state,action),heuristic,max_depth - 1) # recursive call with a deeper depth and the new state after applying current action to current state
            if best_value<value[0]: # check for a bigger maximum
                best_value = value[0]
                best_action = action # if there is a new maximum then the action that leads to this new maximum is the new best value
        return (best_value,best_action) # when gone through all actions return the max value and the coresponding best action
    else: # min node
        best_value = math.inf # initialize best value and best action which will be our retun values later
        best_action = None
        for action in game.get_actions(state): # loop through list of possible actions from left of tree to right of tree
            value = minimax(game,game.get_successor(state,action),heuristic,max_depth - 1)# recursive call with a deeper depth and the new state after applying current action to current state
            if best_value>value[0]: # check for a smaller minimum
                best_value = value[0] # update best value
                best_action = action # update coresponding best action
        return (best_value,best_action) # return the min value after considering all actions

    
# Apply Alpha Beta pruning and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta(game: Game[S, A], state: S, heuristic: HeuristicFunction,max_depth: int = -1,alpha:int = -math.inf,beta: int = math.inf) -> Tuple[float, A]:

    agent = game.get_turn(state) # get turn

    terminal, values = game.is_terminal(state) # terminal node check
    if terminal: return (values[agent],None)
    
    if (max_depth==0): # depth end reached is applied if search is depth limited
        return  (heuristic(game,state,0),None) # return heuristic value is limit reached
    
    if (agent == 0): # max node
        best_value = -math.inf
        best_action = None
        for action in game.get_actions(state): 
            value= alphabeta(game,game.get_successor(state,action),heuristic,max_depth-1,alpha,beta) # recursive call with alpha and beta -> initially called with default values
            if best_value<value[0]: # check for a bigger maximum
                best_value = value[0]
                best_action = action
                alpha = max(alpha,best_value) # update alpha value to travel back up the tree
                if alpha >= beta: # check if we could prune
                    break # don't continue with actions if decided to prune
        return (best_value,best_action) # return max value and best coresponding action
    else: # min node
        best_value = math.inf
        best_action = None
        for action in game.get_actions(state): 
            value = alphabeta(game,game.get_successor(state,action),heuristic,max_depth - 1,alpha,beta) # recursive call with default values of alpha and beta
            if best_value>value[0]: # check for a smalller value
                best_value = value[0]
                best_action = action
                beta = min(beta,best_value) # update beta value to travel back up the tree
                if beta <= alpha: # check if we could prune
                    break # dont travese the rest of the tree if pruned
        return (best_value,best_action) # return min value and best coresponding action




# Apply Alpha Beta pruning with move ordering and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta_with_move_ordering(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1,alpha:int = -math.inf,beta: int = math.inf) -> Tuple[float, A]:
    agent = game.get_turn(state)

    terminal, values = game.is_terminal(state)
    if terminal: return (values[0],None)
    
    if (max_depth==0): # depth end reached
        return  (heuristic(game,state,0),None)
    
    if (agent == 0):
        descend_list= sorted([(heuristic(game,game.get_successor(state,action),0),action) for action in game.get_actions(state)] ,key=lambda x: x[0],reverse=True)
        # this is a list of all expected values of the available actions at the current state the expected values are calculated using the heuristic
        # these values are sorted in descending order to increase chance of finding the max value quicker thus lead to more pruning
        best_value = -math.inf # set up initial values like normal
        best_action = None

        for _,new_action in descend_list: # loop with the ordered actions based on the expected value they return
            value = alphabeta_with_move_ordering(game,game.get_successor(state,new_action),heuristic,max_depth - 1,alpha,beta) # calculate the actual value with recursive call
            if best_value<value[0]: # checks for max value
                best_value = value[0]
                best_action = new_action
                alpha = max(alpha,best_value) # update alpha
                if alpha >= beta: # check for pruning
                    break
        return (best_value,best_action) # return best value
    else:
        accend_list= sorted([(heuristic(game,game.get_successor(state,action),0),action) for action in game.get_actions(state)] ,key=lambda x: x[0])
        # these values are sorted in ascending order to increase chance of finding the min value quicker thus lead to more pruning
        best_value = math.inf
        best_action = None

        for _,new_action in accend_list: # loop with the ordered actions based on the expected value they return
            value = alphabeta_with_move_ordering(game,game.get_successor(state,new_action),heuristic,max_depth - 1,alpha,beta)
            if best_value>value[0]: # checks for min value
                best_value = value[0]
                best_action = new_action
                beta = min(beta,best_value) # update beta
                if beta <= alpha: # check if we could prune
                    break
        return (best_value,best_action)

# Apply Expectimax search and return the tree value and the best action
# Hint: Read the hint for minimax, but note that the monsters (turn > 0) do not act as min nodes anymore,
# they now act as chance nodes (they act randomly).
def expectimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)

    terminal, values = game.is_terminal(state)
    if terminal: return (values[0],None) # check if terminal node and return value if true
    
    if (max_depth==0): # depth end reached
        return  (heuristic(game,state,0),None) # applied if there is a depth limit
    
    if (agent == 0): # max node acts like a typical minimax tree max node
        best_value = -math.inf 
        best_action = None
        for action in game.get_actions(state): 
            value = expectimax(game,game.get_successor(state,action),heuristic,max_depth - 1)
            if best_value<value[0]:
                best_value = value[0]
                best_action = action
        return (best_value,best_action)
    else: # chance node assuming all probabilities are equal
        value_list=[]
        for action in game.get_actions(state): 
            value = expectimax(game,game.get_successor(state,action),heuristic,max_depth - 1)
            value_list.append(value[0]) # calculate all values of deeper levels
        best_value = sum(value_list) / len(value_list) # equal probabilities == average so best value is calculated by the sum of values over the number of values
        return (best_value,None) # there is no best action in this scenario as its done by chance