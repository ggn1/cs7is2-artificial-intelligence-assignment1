import time
import tracemalloc
import numpy as np
from maze import draw_maze

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

def solve_maze(solver, maze, is_print, out_file, out_dir):
    """ 
    Solves given maze using given solver and 
    returns path from start to a goal and other metrics. 
    """
    res = solver(maze)
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

    # Optionally print maze and metrics.
    if is_print:
        with open(f'{out_dir}/{out_file}.txt', 'w', encoding='utf-8') as f:
            f.write(f'MAZE:\n{str(maze)}')
        output_result(result=res, dir=out_dir, filename=out_file)

    return res

def handle_result(res, maze, save_dir, save_filename, save_animation=False):
    ''' Displays path animation and prints 
        given run result for a maze of given shape. 
    '''
    draw_maze(
        maze=maze, path=res['path'], exploration=res['exploration'],
        save_dir=save_dir, save_filename=save_filename, save_animation=save_animation
    )
    print(f"\nMAZE ({maze.shape[0]} x {maze.shape[1]}):")
    print(f"Execution time = {res['seconds']} seconds")
    print(f"No. of dead ends = {res['num_dead_ends']}")
    print(f"No. of nodes traversed = {res['num_nodes_traversed']}/{maze.shape[0]*maze.shape[1]}")

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
            mat[key[0]][key[1]] = "S: "+str(np.round(val, 2))
            continue
        if key in goals:
            mat[key[0]][key[1]] = "G: "+str(np.round(val, 2))
            continue
        mat[key[0]][key[1]] = str(np.round(val, 2))
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

def output_result(result, dir, filename):
    """ Given a run result, this function writes it into the given file. """
    with open(f'{dir}/{filename}.txt', 'a', encoding='utf-8') as f:
        f.write(f'\n\nMETRICS:')
        if ("nano_seconds" in result):
            f.write(f'\nTime taken = {result["nano_seconds"]} nano seconds')
        if ("mem_usage" in result):
            f.write(f"\nMemory usage (peak) = {result['mem_usage']} bytes")
        if ("solution" in result):
            f.write(f"\nSolution path size = {len(result['solution'])} positions")
        if ("num_nodes_traversed" in result):
            f.write(f'\nNo. of positions traversed = {result["num_nodes_traversed"]}')
        if ("num_dead_ends" in result):
            f.write(f'\nNo. of dead ends = {result["num_dead_ends"]}')
        if ("num_iterations" in result):
            f.write(f"\nNo. of iterations = {result["num_iterations"]}")

def get_solution(maze, policy, dir_out, file_out):
    """ 
    Given returns the solution (path from start 
    to goal) to the given maze as per given policy.
    """
    print('Getting solution ...')
    s =  maze.start # Begin at the start state.
    solution = [s]
    while (not s in maze.goals): # Until the goal state is reached ...
        if not s in policy:
            print('No solution found.')
            with open(f'{dir_out}/{file_out}.txt', 'a', encoding='utf-8') as f:
                f.write("\n\nNo solution found.")
            break
        actions = policy[s]
        a = actions[0] # Get best action for this state as per policy.
        s_prime = maze.states[s][a] # Get next state as per policy.
        if (s_prime in solution): # I step loop.
            print('Loop')
            with open(f'{dir_out}/{file_out}.txt', 'a', encoding='utf-8') as f:
                f.write("\nLoop")
            break
        solution.append(s_prime) # Append state to the solution.
        s = s_prime
    return solution

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