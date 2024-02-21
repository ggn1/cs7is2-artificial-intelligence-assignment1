# Imports 
import numpy as np
from queue import PriorityQueue
from track_time import track_time
from utility import handle_result
from maze import draw_maze, load_maze, MazeMDP

# def get_possible_s_primes(s, actions, states):
#     return [
#         s_prime 
#         for s_prime in [(s[0] + a[0], s[1] + a[1]) for a in actions] 
#         if s_prime in states
#     ]
#     return [(s[0] + a[0], s[1] + a[1]) for a in actions]

def print_2d_array(arr):
    for row in arr:
        print(row.tolist())

def initialize_v(maze):
    V = np.zeros(maze.maze.shape) # Value at all maze positions = 0.
    # V = (V == maze.maze).astype(int).astype(float) # Wall = 1, other = 0.
    # # Set all positions with a wall to have a big negative value
    # # since these are not valid states in the maze.
    # V[V == 1] = -1
    # Every actual state in the maze (position with no walls) 
    # is consequently initialized to have value 0.
    return V

def value_iteration(maze, epsilon, print_values):
    """ 
    Performs value iteration and returns values for each state
    and no. of iterations before convergence. 
    """
    # Initially, value of every valid state to be 0.
    # Positions corresponding to invalid states (walls) = -(maze dim)^2.
    Vold = np.zeros(maze.maze.shape)
    Vnew = np.zeros(maze.maze.shape)
    converged = False # Keep track of convergence.
    k = 0 # Keep track of kth iteration.
    print('\nPerforming value iteration ...')
    while(not converged): # Repeat until convergence.
        # For each state ...
        for s in maze.states:
            val_max_a = float('-inf') # Max value possible.
            # For each action ...
            for a in maze.actions:
                # Compute sum of immediate and future discounted rewards across all possible states from s.
                val_sum = 0 
                # For each possible new state ...
                for s_prime in [(s[0] + a[0], s[1] + a[1]) for a in maze.actions]:
                    transitionProb = maze.T(s, a, s_prime)
                    if (transitionProb > 0):
                        reward = (
                            maze.R(s) + # Immediate reward + 
                            (maze.gamma * Vold[s_prime]) # Discounted future reward.
                        )
                        val = transitionProb * reward
                    else:
                        val = 0
                    val_sum += val # Sum values.
                if val_sum > val_max_a: # Max val over actions.
                    val_max_a = val_sum
            Vnew[s] = val_max_a # keep track of computed value.
        # Value iteration is considered to have converged 
        # if maximum change of all state values from state 
        # k to k+1 <= some small epsilon change.
        converged = np.max(np.abs(Vnew - Vold)) <= epsilon # Check convergence.
        Vold = Vnew.copy() # Set latest values as old values for next iteration.
        k += 1 # Update iteration counts.
    if print_values:
        print(f'Iteration {k}:')
        print_2d_array(Vold)
    return Vold, k

def policy_extraction(maze, V):
    """ 
    Given values of each state and the maze,
    extracts policy as being the action at each
    state which maximizes expected reward. Returns path.
    """
    print('\nExtracting policy ...')
    policy = {state: (0, 0) for state in maze.states}
    for s in list(policy.keys()):
        val_max_a = float('-inf') # Max value possible.
        best_a = (0, 0) # Keep track of best action for this state.
        # For each action ...
        for a in maze.actions:
            # Compute sum of immediate and future discounted rewards across all possible states from s.
            val_sum = 0 
            # For each possible new state ...
            for s_prime in [(s[0] + a[0], s[1] + a[1]) for a in maze.actions]:
                transitionProb = maze.T(s, a, s_prime)
                if (transitionProb > 0):
                    reward = (
                        maze.R(s) + # Immediate reward + 
                        (maze.gamma * V[s_prime]) # Discounted future reward.
                    )
                    val = transitionProb * reward
                else:
                    val = 0
                val_sum += val # Sum values.
            if val_sum > val_max_a: # Max val over actions.
                val_max_a = val_sum
                best_a = a
        if best_a == (0, 0):
            print('No solution found')
            break
        policy[s] = best_a
    return policy

@track_time
def solver(maze, epsilon=1e-2, print_values=False):
    if print_values: 
        print('Maze:')
        print_2d_array(maze.maze)

    V, num_iters = value_iteration(maze, epsilon, print_values)
    
    policy = policy_extraction(maze, V)
    
    s = (1,1)
    path = []
    while (s != maze.goal):
        if (s in path):
            print('Loop')
            break
        path.append(s)
        a = policy[s]
        s = (s[0]+a[0], s[1]+a[1])
    path.append(s)

    return {
        'path': path,
        'v': V,
        'policy': policy,
        'num_iterations': num_iters
    }

if __name__ == '__main__':
    # TEST MAZE
    maze = MazeMDP(dim=50)
    res = solver(maze)
    # print('Result =', res)
    draw_maze(
        maze=maze.maze, 
        path=res['path'], 
        v=res['v']
    )

    # # TINY MAZE
    # maze = MazeMDP(maze=load_maze(path='./mazes/t_dim5.json'))
    # res = solver(maze, print_values=True)
    # print('Result =', res)
    # draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'])

    # # SMALL MAZE
    # maze = MazeMDP(maze=load_maze(path='./mazes/s_dim21.json'))
    # res = solver(maze)
    # # print('Result =', res)
    # draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'])

    # # MEDIUM MAZE
    # maze = MazeMDP(maze=load_maze(path='./mazes/m_dim41.json'))
    # res = solver(maze)
    # draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'])

    # # LARGE MAZE
    # maze = MazeMDP(maze=load_maze(path='./mazes/l_dim101.json'))
    # res = solver(maze)
    # # print('Result =', res)
    # draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'])