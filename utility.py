import os
import time
import tracemalloc
import numpy as np

# This file contains functions that are used by multiple other files.

def reconstruct_path(parents, start, goal):
    """
    Reconstructs the path to start given a goal and a dictionary of node
    to parent links that were built up while looking for the given goal.
    """
    current = goal
    path = [current]
    while current != start:
        current = parents[current]
        path.append(current)
    path.reverse() # Reverse the path to go from from start to goal.
    return path

def values_to_mat(v, shape):
    """ 
    Converts given numpy array with MDP state values 
    of valid states only into a matrix wherein invalid
    states have value -infinity. This is so that walls
    are clearly visible against valid states when
    visualized using matplotlib.
    """
    mat = np.full(shape, -1*float('inf'))
    for key, val in v.items():
        mat[key] = val
    return mat

def values_to_mat_str(v, shape, start, goals):
    """
    Converts given numpy array with MDP state values 
    into a string representation of the maze wherein
    walls are represented using the '#' symbol and
    'S' and 'G' represent the start and goal respectively.  
    """
    mat = np.full(shape, '#').tolist()
    for key, val in v.items():
        if key == start:
            mat[key[0]][key[1]] = "S: "+str(val)
            continue
        if key in goals:
            mat[key[0]][key[1]] = "G: "+str(val)
            continue
        mat[key[0]][key[1]] = str(val)
    mat = [str(row) for row in mat]
    return "\n".join(mat)

def policy_to_mat(policy, shape, start, goals):
    """
    Converts a given policy into a string representation
    wherein walls are represented using the '#' symbol and
    'S' and 'G' represent the start and goal respectively. 
    """
    mat = np.full(shape, "#").tolist()
    for key, val in policy.items():
        if (key == start): 
            mat[key[0]][key[1]] = f"S: {val}"
        elif (key in goals): 
           mat[key[0]][key[1]] = f"G: {val}"
        else:
            mat[key[0]][key[1]] = val
    mat = [str(row) for row in mat]
    return "\n".join(mat)

def output_result(result, out_dir, out_file):
    """ Given a run result, this function writes it into the given file. """
    with open(f'{out_dir}/{out_file}.txt', 'a', encoding='utf-8') as f:
        f.write(f'\n\nMETRICS:')
        if ("nano_seconds" in result):
            f.write(f'\nTime taken = {result["nano_seconds"]} nano seconds')
        if ("mem_usage" in result):
            f.write(f"\nMemory usage (peak) = {result['mem_usage']} bytes")
        if ("solution" in result):
            f.write(f"\nSolution path size = {len(result['solution'])} positions")
        if ("num_nodes_traversed" in result):
            f.write(f'\nNo. of positions traversed = {result["num_nodes_traversed"]}')
        if ("num_iterations" in result):
            f.write(f"\nNo. of iterations = {result["num_iterations"]}")

def extract_solution_mdp(maze, policy, out_dir, out_file):
    """ 
    Given a policy for a given maze, returns path from start
    to a goal if possible, as per that policy.
    """
    s =  maze.start # Begin at the start state.
    solution = [s]
    while (not s in maze.goals): # Until the goal state is reached ...
        if not s in policy:
            print('No solution found.')
            with open(f'{out_dir}/{out_file}.txt', 'a', encoding='utf-8') as f:
                f.write("\n\nNo solution found.")
            return solution
            break
        actions = policy[s]
        a = actions[0] # Get best action for this state as per policy.
        s_prime = maze.states[s][a] # Get next state as per policy.
        if (s_prime in solution): # s' in the solution already => loop
            print('Loop')
            with open(f'{out_dir}/{out_file}.txt', 'a', encoding='utf-8') as f:
                f.write("\nLoop")
            break
        solution.append(s_prime) # Append state to the solution.
        s = s_prime # s' is s in the next iteration
    return solution

def solve_maze(solver_type, solver, maze, out_dir, out_file, gamma=None, epsilon=None, max_iters=None):
    """ 
    Solves given maze using given solver and 
    returns path from start to a goal and other metrics. 
    """
    # If output location does not exist, create it.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Output maze.
    with open(f'{out_dir}/{out_file}.txt', 'w', encoding='utf-8') as f:
        f.write(f'MAZE:\n{str(maze)}')

    # Call solver with appropriate arguments as per whether
    # it is a 'search' or 'mdp' solver.
    solution_type = '' 
    if solver_type in ['dfs', 'bfs', 'a-star']:
        res = solver(maze)
        solution_type = 'search'
    elif solver_type == 'value-iteration':
        res = solver(
            maze=maze, out_dir=out_dir, out_file=out_file, 
            gamma=gamma, epsilon=epsilon, max_iters=max_iters
        )
        solution_type = 'mdp'
    elif solver_type == 'policy-iteration':
        res = solver(maze=maze, out_dir=out_dir, out_file=out_file, gamma=gamma)
        solution_type = 'mdp'
    else: # Throw an exception if the solver is invalid.
        raise Exception(f'Invalid solver type "{solver_type}".')
    
    # Extract solution using the appropriate method
    # for each kind of solver.
    if solution_type == 'mdp':
        print('Extracting solution from policy ...')
        res['solution'] = extract_solution_mdp(
            maze=maze, policy=res['policy'], 
            out_dir=out_dir, out_file=out_file
        )

    else: # solution_type == 'search'
        res['solution'] = [] # Path reconstruction.
        if res['goal'] is None: # No goal found => no solution.
            with open(f"{out_dir}/{out_file}.txt", 'a') as f:
                f.write("\nNo solution found.")
                print("No solution found.")
        res['solution'] = reconstruct_path(
            parents=res['parents'], 
            start=maze.start, 
            goal=res['goal']
        )

    # Output performance metrics.
    output_result(result=res, out_dir=out_dir, out_file=out_file)

    # Return result.
    return res

def track_mem_time(solver):
    ''' This function will return a wrapper function
        that computes and returns both execution time
        as well as memory usage of given solver.
        Any maze solver function that returns a path that
        solves the corresponding maze may be decorated with
        this function.
    '''
    def wrapper(*args, **kwargs):
        time_start = time.time_ns() # keep track of time
        tracemalloc.start() # keep track of memory usage
        res = solver(*args, **kwargs)
        # Add executing time in seconds to the result 
        # that is to be returned.
        mem_usage_peak = tracemalloc.get_traced_memory()[1]
        res['mem_usage'] = mem_usage_peak
        res['nano_seconds'] = time.time_ns() - time_start 
        return res
    return wrapper

def get_complete_maze_dim_limit(epsilon, gamma, step=1):
    """ 
    Given an epsilon = value iteration change threshold,
    and a gamma = discount factor (0, 1), this function
    returns the maximum input dimension of a maze <= which, 
    value iteration will be complete.
    """
    dim = 2 # Min dimension possible as per current implementation = 2.
    reward = 0 # Rewards
    try:
        while(reward < float('inf')): # as long as reward != positive overflow
            wh = ((dim*2)+1)-2 
            max_steps = wh**2 # max no. of valid states in the maze
            # Compute a reward that won't be zero even 
            # if discounted for max_steps no. of steps.
            reward = epsilon/(gamma**max_steps) 
            dim += step # Increase dimension by 1.
        return dim - step - 1 # Return last valid dimension.
    except ZeroDivisionError: # reward = negative overflow
        return dim - step - 1 # Return last valid dimension.
    
def get_best_reward(dim, epsilon, gamma):
    """
    Given an input maze dimension dim,
    an epsilon = value iteration change threshold, 
    and a gamma = discount factor (0, 1), this function returns 
    a reward such that value iteration is complete on 
    a maze of size (dim*2)+1 even if the optimal path is
    as long as the no. of valid states in the maze.
    """
    wh = (dim*2)+1 # width = height of the maze.
    max_steps = wh**2 # Size of maze = no. of states in the maze.
    reward = epsilon/(gamma**max_steps)
    return reward