# Imports 
import numpy as np
from track_time import track_time
from utility import print_mat_2d, v_to_mat
from maze import draw_maze, load_maze, Maze

def value_iteration(maze, gamma, epsilon, print_values):
    """ 
    Performs value iteration and returns values for each state
    and no. of iterations before convergence. 
    """
    # Initially, value of every valid state to be 0.
    # Positions corresponding to invalid states (walls) = -(maze dim)^2.
    V = {state:0 for state in maze.states}
    converged = False # Keep track of convergence.
    k = 0 # Keep track of kth iteration.
    print('\nPerforming value iteration ...')
    while(not converged): # Repeat until convergence.
        # For each state ...
        state_max_diff = 0
        for s in maze.states:
            V_old = V.copy()
            Q = {} # Max value possible.
            # For each action ...
            for a in maze.actions:
                # Compute sum of immediate and future discounted rewards across all possible states from s.
                u = 0 
                # For each possible new state ...
                for s_prime in [(s[0]+a[0], s[1]+a[1]) for a in maze.actions]:
                    transitionProb = maze.T(s, a, s_prime)
                    if (transitionProb > 0):
                        reward = (
                            maze.R(s, a, s_prime) + # Immediate reward + 
                            (gamma * V_old[s_prime]) # Discounted future reward.
                        )
                        u += transitionProb * reward
                Q[a] = u
            V[s] = max(Q.values()) # keep track of computed value.
            state_diff = abs(V[s] - V_old[s])
            if state_diff > state_max_diff: 
                state_max_diff = state_diff
        # Value iteration is considered to have converged 
        # if maximum change of all state values from state 
        # k to k+1 <= some small epsilon change.
        converged = state_max_diff <= epsilon # Check convergence.
        k += 1 # Update iteration counts.
    if print_values:
        print(f'Iteration {k}:')
        print_mat_2d(v_to_mat(v=V, shape=maze.maze.shape))
    return V, k

def policy_extraction(maze, V):
    """ 
    Given values of each state and the maze,
    extracts policy as being the action at each
    state which maximizes expected reward. Returns path.
    """
    print('\nExtracting policy ...')
    policy = {state: (0, 0) for state in maze.states}
    for s in list(policy.keys()):
        Q = {}
        # For each action ...
        for a in maze.actions:
            u = 0 # expected utility of taking action a from state s
            # For each possible state ...
            for s_prime in [(s[0]+a[0], s[1]+a[1]) for a in maze.actions]:
                transition_prob = maze.T(s, a, s_prime)
                if (transition_prob > 0):
                    reward = (
                        maze.R(s, a, s_prime) + # Immediate reward + 
                        (gamma * V[s_prime]) # Discounted future reward.
                    )
                    u += transition_prob * reward
            Q[a] = u
        policy[s] = max(Q, key=Q.get)
    return policy

@track_time
def solver(maze, epsilon=1e-3, print_values=False):
    if print_values: 
        print('Maze:')
        print_mat_2d(maze.maze)

    V, num_iters = value_iteration(maze, epsilon, print_values)
    
    policy = policy_extraction(maze, V)
    
    s = (1,1)
    path = []
    while (s != maze.goal):
        try:
            if (s in path):
                print('Loop')
                break
            path.append(s)
            a = policy[s]
            s = maze.succ(s, a)
        except:
            break
    path.append(s)

    return {
        'path': path,
        'v': v_to_mat(V, maze.maze.shape),
        'policy': policy,
        'num_iterations': num_iters
    }

if __name__ == '__main__':
    # TEST MAZE
    maze = Maze(dim=60)
    res = solver(maze)
    draw_maze(
        maze=maze.maze, 
        path=res['path'], 
        v=res['v']
    )

    # # TINY MAZE
    # maze = Maze(maze=load_maze(path='./mazes/t_dim5.json'))
    # res = solver(maze, print_values=True)
    # draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # # SMALL MAZE
    # maze = Maze(maze=load_maze(path='./mazes/s_dim21.json'))
    # res = solver(maze)
    # draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # # MEDIUM MAZE
    # maze = Maze(maze=load_maze(path='./mazes/m_dim41.json'))
    # res = solver(maze)
    # draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # # LARGE MAZE
    # maze = Maze(maze=load_maze(path='./mazes/l_dim101.json'))
    # res = solver(maze)
    # draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'], v=res['v'])