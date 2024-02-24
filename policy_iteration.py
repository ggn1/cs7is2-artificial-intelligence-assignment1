# Imports 
import random
import numpy as np
from track_time import track_time
from maze import draw_maze, load_maze, Maze
from utility import policy_to_mat, values_to_mat, values_to_mat_str, get_solution

output = 'output_pol_iter'

def policy_evaluation(maze, V, policy, gamma):
    """ Evaluates given policy and returns value of each state based on it. """
    # For each state ...
    Vold = V.copy()
    for s in maze.state_positions:
        a = policy[s][0] # Take action as per given policy.
        # Compute sum of immediate and future discounted rewards 
        # across all possible next states from s.
        u = 0 # expected utility under given policy
        # For each possible next state ...
        for s_prime in maze.states[s].surroundings.values():
            transitionProb = maze.T(s, a, s_prime) # Transition probability.
            # Proceed only if not 0, else index out of bounds error.
            # This shall not affect end result as 0 * anything = 0.
            if (transitionProb > 0): 
                reward = (
                    maze.R(s, a, s_prime) + # Immediate reward + 
                    (gamma * Vold[s_prime]) # Discounted future reward.
                )
                u += transitionProb * reward
        V[s] = u # keep track of computed value.
    return V

def policy_improvement(maze, V, gamma):
    """ Updates given policy to be optimized as per latest values of each state. """
    policy = {}
    for s in maze.state_positions:
        Q = {}
        # For each action ...
        for a in maze.actions:
            u = 0 # Expected utility of taking action a at state s as per last evaluated policy.
            # For each possible next state ...
            for s_prime in maze.states[s].surroundings.values():
                transition_prob = maze.T(s, a, s_prime)
                if (transition_prob > 0):
                    reward = (
                        maze.R(s, a, s_prime) + # Immediate reward + 
                        (gamma * V[s_prime]) # Discounted future reward.
                    )
                    u += transition_prob * reward
            Q[a] = u
        u_max = max(Q.values())
        policy[s] = [a for a, u in Q.items() if u == u_max] # Update policy.
    return policy

@track_time
def policy_iteration(maze, gamma=0.99, is_print=False):
    if is_print: 
        with open(f'{output}.txt', 'w', encoding="utf-8") as f:
            f.write('Maze:\n')
            f.write(str(maze))

    # Initially,
    V = {}
    policy = {}
    v_init = 0.0
    for state in maze.state_positions: # for all states in the maze,
        V[state] = v_init # value = 0 and 
        policy[state] = [random.choice(maze.actions)] # policy is random.

    k = 0 # Keep count of no. of iterations.
    converged = False # Keep track of whether policy has converged.
    # Policy is considered to have converged if it no longer changes
    # for all states between one iteration and the next.
    print('Policy iteration ...')
    while(not converged): 
        V = policy_evaluation(maze, V, policy, gamma)
        policy_improved = policy_improvement(maze, V, gamma)
        # If no change in policy after improvement,
        # then, policy has converged.
        converged = all(policy_improved[s] == policy[s] for s in maze.state_positions)
        policy = policy_improved # Else, prep for next round of evaluation and improvement.
        k += 1
        if is_print:
            with open(f'__output/{output}.txt', 'a', encoding="utf-8") as f:
                f.write(f'\n\nIteration {k}:\n')
                f.write(values_to_mat_str(
                    v=V, shape=maze.matrix.shape, 
                    start=maze.start, goal=maze.goal
                ))
            with open(f'__output/{output}.txt', 'a', encoding="utf-8") as f:
                f.write(f'\nPolicy:\n')
                f.write(str(policy_to_mat(
                    policy=policy, shape=maze.matrix.shape, 
                    start=maze.start, goal=maze.goal
                )))
    
    # Once policy has converged, get the best path to goal based on it.
    solution = get_solution(maze, policy)

    return {
        'solution': solution, 
        'state_values': values_to_mat(V, maze.matrix.shape),
        'policy': policy,
        'num_iterations': k
    }

if __name__ == '__main__':
    # TEST MAZE
    # maze = Maze(dim=70)
    # save_maze(maze, dir='.', filename='maze_latest')
    dim = 21
    maze = Maze(matrix=load_maze(f'./maze_latest_dim{dim}.json'))
    res = policy_iteration(maze, is_print=True, gamma=0.99)
    output = f'output_pol_iter_dim{dim}'
    draw_maze(
        maze=maze.matrix, 
        solution=res['solution'], 
        state_values=res['state_values'],
        save={'dir':'__output', 'filename':output, 'animation':False}
    )