from typing import Any, Dict, List, Optional
from CSP import Assignment, BinaryConstraint, Problem, UnaryConstraint
from helpers.utils import NotImplemented
from copy import deepcopy

# This function should apply 1-Consistency to the problem.
# In other words, it should modify the domains to only include values that satisfy their variables' unary constraints.
# Then all unary constraints should be removed from the problem (they are no longer needed).
# The function should return False if any domain becomes empty. Otherwise, it should return True.
def one_consistency(problem: Problem) -> bool:
    constraints_to_remove=[]
    for con in problem.constraints: # loop on all problem constraints
        if con.__class__.__name__== "UnaryConstraint": # specifiy only unary constraints
            to_remove=[] # a list to store domain values to remove
            for val in problem.domains[con.variable]: # loop on all values of the domain of the variable associated with the constraint
                if con.condition(val) == False: # is the value does not satisfy the condition
                    to_remove.append(val) # append it to the to be removed list
            for val in to_remove: # after looping on all values start removing the values that do not satisfy the constraint
                problem.domains[con.variable].remove(val)
            if not problem.domains[con.variable]: # if the domain became empty after the removal then the problem cannot be solved so return false
                return False
            constraints_to_remove.append(con) # when done with constraint append it to the constraints to be removed list
    for con in constraints_to_remove: # remove all unary constraints from the problem
        problem.constraints.remove(con)
    return True # return true meaning there exists a solution that satisfies the unary constraints

# This function should implement forward checking
# The function is given the problem, the variable that has been assigned and its assigned value and the domains of the unassigned values
# The function should return False if it is impossible to solve the problem after the given assignment, and True otherwise.
# In general, the function should do the following:
#   - For each binary constraints that involve the assigned variable:
#       - Get the other involved variable.
#       - If the other variable has no domain (in other words, it is already assigned), skip this constraint.
#       - Update the other variable's domain to only include the values that satisfy the binary constraint with the assigned variable.
#   - If any variable's domain becomes empty, return False. Otherwise, return True.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def forward_checking(problem: Problem, assigned_variable: str, assigned_value: Any, domains: Dict[str, set]) -> bool:
    related_constraints = []
    for con in problem.constraints: # loop used to find all constraints related to the assigned variable and append them to related_constraints
        if con.variables[0] == assigned_variable or con.variables[1] == assigned_variable:
            related_constraints.append(con)
    for con in related_constraints: # loop on related_constraints for the assigned variable
        other = con.get_other(assigned_variable) # get the other variable in the constriants
        if other in domains.keys(): # if other variable exists in the current domain of unassigned variables 
            to_remove = [] # list to be filled with values to remove from domain of other variable
            for val in domains[other]: # loop on domain of other variable
                if con.is_satisfied({assigned_variable:assigned_value,other:val}) == False: # if other value breaks constraint append it to the to be removed
                    to_remove.append(val)
            for val in to_remove: # remove all values that dont satisfy the constraint
                domains[other].remove(val) 
            if not domains[other]: # if domain is empty that means that the current assignment cannot be the soultion thus return false
                return False
    return True # else the current assignment still could work


# This function should return the domain of the given variable order based on the "least restraining value" heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# Generally, this function is very similar to the forward checking function, but it differs as follows:
#   - You are not given a value for the given variable, since you should do the process for every value in the variable's
#     domain to see how much it will restrain the neigbors domain
#   - Here, you do not modify the given domains. But you can create and modify a copy.
# IMPORTANT: If multiple values have the same priority given the "least restraining value" heuristic, 
#            order them in ascending order (from the lowest to the highest value).
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def least_restraining_values(problem: Problem, variable_to_assign: str, domains: Dict[str, set]) -> List[Any]:
    related_constraints = []
    for con in problem.constraints: # get related constraints for variable to be assigned
        if con.variables[0] == variable_to_assign or con.variables[1] == variable_to_assign:
            related_constraints.append(con)

    restrain ={}
    for val in problem.domains[variable_to_assign]: # loop through all values of variable to be assigned
        count = 0
        for con in related_constraints: # loop through related constraints
            other = con.get_other(variable_to_assign) # get other variable in constraint
            if other in domains.keys(): # if other is in variables to be assigned
                for other_val in domains[other]:  # loop through all values of domain of other 
                    if con.is_satisfied({variable_to_assign:val,other:other_val}) == False: # if constraint is not satisfied increase count of restarints caused by this future assignment
                        count += 1
        restrain[val] = count # add the total count of restraints associated with this value of the variable to be assigned

    ans = []
    for w in sorted(restrain.items(), key=lambda x: x[1]): # sort ascendingly based on the number of restraints
        ans.append(w[0])  # return the list of values of the variable in the order of the least to most restraining
    return ans

# This function should return the variable that should be picked based on the MRV heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
# IMPORTANT: If multiple variables have the same priority given the MRV heuristic, 
#            order them in the same order in which they appear in "problem.variables".
def minimum_remaining_values(problem: Problem, domains: Dict[str, set]) -> str:
    var_and_count = {}
    for var in problem.variables:
        if var in domains.keys():
            var_and_count[var] = len(domains[var]) # count the number the domain values of each variable in the problem
    minimum = min(var_and_count.values()) # find the lowest number 
    for var in var_and_count.keys():
        if var_and_count[var] == minimum:
            return var # return the coresponding variable 

# check if the variable to be assigned and value to assign satisfies all constraints with variables and values already assigned
def check_constraints(assignment:Assignment,problem:Problem,variable_to_assign: str,value_to_assign:Any):
    if assignment == {}: return True
    related_constraints = []
    for con in problem.constraints: # get related constraints to variable to be assigned
        if con.variables[0] == variable_to_assign or con.variables[1] == variable_to_assign:
            related_constraints.append(con)
    for con in related_constraints:
        other = con.get_other(variable_to_assign)
        if other in assignment.keys():
                if con.is_satisfied({variable_to_assign:value_to_assign,other:assignment[other]}) == False: # if value to be assigned does not satisfy a constraint with variables already assigned
                    return False # return false indicating that this assignment will not work
    return True # else return true

# backtracking funtion that uses minimum_remaining_values and least_restraining_values and forward_checking
def backtrack(assignment:Assignment,problem:Problem,domains: Dict[str, set]):
    if problem.is_complete(assignment): # base case all variables are assigned return assignment
        return assignment
    var = minimum_remaining_values(problem,domains) # choose a variable with the least remaining values
    for val in least_restraining_values(problem,var,domains): # sort list of values according to the least restaining and loop
            assignment[var] = val # test assign variable with values
            current_domains_state = deepcopy(domains) # save current domain to undo deletion, deepcopy is used to ensure a complete separate copy of dict domains
            del domains[var] # delete domain of variable to only keep the domain of the unassigned variables
            if check_constraints(assignment,problem,var,val): # check assignment of the variable goes with the values of previously assigned vars
                if forward_checking(problem,var,val, domains):# apply forward checking and if there is no empty domain after continue
                    result = backtrack(assignment,problem,domains) # apply this assignment permenantly and recusrive call to assign rest of variables
                    if result != None : return result # if there is a possible assignment return assignment
            domains= current_domains_state # else undo changes in domain
            del assignment[var] # unassign the variable and loop on another value
    return None # if ther is no possible assignment return false
        

# This function should solve CSP problems using backtracking search with forward checking.
# The variable ordering should be decided by the MRV heuristic.
# The value ordering should be decided by the "least restraining value" heurisitc.
# Unary constraints should be handled using 1-Consistency before starting the backtracking search.
# This function should return the first solution it finds (a complete assignment that satisfies the problem constraints).
# If no solution was found, it should return None.
# IMPORTANT: To get the correct result for the explored nodes, you should check if the assignment is complete only once using "problem.is_complete"
#            for every assignment including the initial empty assignment, EXCEPT for the assignments pruned by the forward checking.
#           XX Also, if 1-Consistency deems the whole problem unsolvable, you shouldn't call "problem.is_complete" at all.
def solve(problem: Problem) -> Optional[Assignment]:
    if one_consistency(problem) == False: # calling one_consistency once intially
        return None # if one of the domains became empty return None i.e. no possible assignment
    domains = deepcopy(problem.domains) # used a temporary domain instead of problem.domains, used deep copy again to ensure a complete separate copy of dict domains
    return backtrack({},problem,domains) # initial call to backtrack with empty assignment