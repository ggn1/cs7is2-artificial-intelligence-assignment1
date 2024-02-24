import numpy as np
from maze import draw_maze

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
    mat = np.full(shape, -1*float('inf'))
    for key, val in v.items():
        mat[key] = val
    return mat

def values_to_mat_str(v, shape, start, goal):
    mat = np.full(shape, '#').tolist()
    for key, val in v.items():
        if key == start:
            mat[key[0]][key[1]] = "S: "+str(np.round(val, 2))
            continue
        if key == goal:
            mat[key[0]][key[1]] = "G: "+str(np.round(val, 2))
            continue
        mat[key[0]][key[1]] = str(np.round(val, 2))
    mat = [str(row) for row in mat]
    return "\n".join(mat)

def policy_to_mat(policy, shape, start, goal):
    mat = np.full(shape, "#").tolist()
    for key, val in policy.items():
        if (key == start): 
            mat[key[0]][key[1]] = f"S: {val}"
        elif (key == goal): 
           mat[key[0]][key[1]] = f"G: {val}"
        else:
            mat[key[0]][key[1]] = val
    mat = [str(row) for row in mat]
    return "\n".join(mat)

def get_solution(maze, policy):
    """ 
    Given returns the solution (path from start 
    to goal) to the given maze as per given policy.
    """
    print('Getting solution ...')
    s =  maze.start # Begin at the start state.
    solution = [s]
    while (s != maze.goal): # Until the goal state is reached ...
        if not s in policy:
            print('No solution found.')
            break
        actions = policy[s]
        a = actions[0] # Get best action for this state as per policy.
        s_prime = maze.states[s][a] # Get next state as per policy.
        if (s_prime in solution): # I step loop.
            print('Loop')
            break
        solution.append(s_prime) # Append state to the solution.
        s = s_prime
    return solution