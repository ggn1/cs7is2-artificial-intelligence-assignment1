# Imports 
import random
import numpy as np
from track_time import track_time
from utility import print_mat_2d, v_to_mat
from maze import draw_maze, load_maze, MazeMDP

def policy_evaluation(maze, V, policy):
    """ Evaluates given policy and returns value of each state based on it. """
    # For each state ...
    Vold = V.copy()
    for s in maze.states:
        a = policy[s] # Take action as per given policy.
        # Compute sum of immediate and future discounted rewards 
        # across all possible next states from s.
        u = 0 # expected utility under given policy
        # For each possible next state ...
        for s_prime in [(s[0]+a[0], s[1]+a[1]) for a in maze.actions]:
            transitionProb = maze.T(s, a, s_prime) # Transition probability.
            # Proceed only if not 0, else index out of bounds error.
            # This shall not affect end result as 0 * anything = 0.
            if (transitionProb > 0): 
                reward = (
                    maze.R(s, a, s_prime) + # Immediate reward + 
                    (maze.gamma * Vold[s_prime]) # Discounted future reward.
                )
                u += transitionProb * reward
        V[s] = u # keep track of computed value.
    return V

def policy_improvement(maze, V):
    """ Updates given policy to be optimized as per latest values of each state. """
    policy = {}
    for s in maze.states:
        Q = {}
        # For each action ...
        for a in maze.actions:
            u = 0 # Expected utility of taking action a at state s as per last evaluated policy.
            # For each possible next state ...
            for s_prime in [(s[0]+a[0], s[1]+a[1]) for a in maze.actions]:
                transition_prob = maze.T(s, a, s_prime)
                if (transition_prob > 0):
                    reward = (
                        maze.R(s, a, s_prime) + # Immediate reward + 
                        (maze.gamma * V[s_prime]) # Discounted future reward.
                    )
                    u += transition_prob * reward
            Q[a] = u
        policy[s] = max(Q, key=Q.get) # Update the policy.
    return policy

def get_path(maze, policy):
    print('Determining path ...')
    s = maze.start # Start state.
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
    return path

@track_time
def policy_iteration(maze, print_values=False):
    if print_values: 
        print('Maze:')
        print_mat_2d(maze.maze)

    # Initially,
    v_init = 0.0
    V = {}
    policy = {}
    for state in maze.states: # for all states in the maze,
        V[state] = v_init # value = 0 and 
        policy[state] = random.choice(maze.actions) # policy is random.

    k = 0 # Keep count of no. of iterations.
    converged = False # Keep track of whether policy has converged.
    # Policy is considered to have converged if it no longer changes
    # for all states between one iteration and the next.
    print('Policy iteration ...')
    while(not converged): 
        V = policy_evaluation(maze, V, policy)
        policy_improved = policy_improvement(maze, V)
        # If no change in policy after improvement,
        # then, it has converged.
        converged = all(
            policy_improved[state] == policy[state] 
            for state in maze.states
        )
        policy = policy_improved # Else, prep for next round of evaluation and improvement.
        k += 1
    
    # Once policy has converged, get the best path to goal based on it.
    path = get_path(maze, policy)

    return {
        'path': path,
        'v': v_to_mat(V, maze.maze.shape),
        'policy': policy,
        'num_iterations': k
    }

if __name__ == '__main__':
    # # TEST MAZE
    # maze = MazeMDP(dim=20, gamma=0.9)
    # res = policy_iteration(maze)
    # print('No. of iterations =', res['num_iterations'])
    # draw_maze(
    #     maze=maze.maze, 
    #     path=res['path'], 
    #     v=res['v']
    # )

    # TINY MAZE
    maze = MazeMDP(maze=load_maze(path='./mazes/t_dim5.json'), gamma=0.9)
    res = policy_iteration(maze, print_values=True)
    draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # SMALL MAZE
    maze = MazeMDP(maze=load_maze(path='./mazes/s_dim21.json'), gamma=0.9)
    res = policy_iteration(maze)
    draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # MEDIUM MAZE
    maze = MazeMDP(maze=load_maze(path='./mazes/m_dim41.json'), gamma=0.9)
    res = policy_iteration(maze)
    draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # LARGE MAZE
    maze = MazeMDP(maze=load_maze(path='./mazes/l_dim101.json'), gamma=0.9)
    res = policy_iteration(maze)
    draw_maze(maze=maze.maze, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.maze.shape[0]}', save_animation=False, path=res['path'], v=res['v'])