# Imports 
import numpy as np
from track_time import track_time
from maze import draw_maze, save_maze, load_maze, Maze
from utility import policy_to_mat, values_to_mat, values_to_mat_str, get_solution

output = 'output_val_iter'

def update_values(maze, gamma, epsilon, is_print):
    """ 
    Performs value iteration and returns values for each state
    and no. of iterations before convergence. 
    """
    converged = False # Keep track of convergence.
    k = 0 # Keep track of kth iteration.
    print('\nUpdating utility values ...')
    # Initially, value associated with every valid position is 0.
    # v_init = float(maze.matrix.shape[0] ** 2)
    v_init = 0.0
    V = {state: v_init for state in maze.state_positions}
    while(not converged): # Repeat until convergence.
        state_diff = [] # Absolute difference between state values now v/s before.
        V_old = V.copy() # Since values will change in this iteration, keep a copy of old ones.
        for s in maze.state_positions: # For each valid state (not wall).
            Q = {} # Expected utility of each action.
            for a in maze.actions: # For each possible action ...
                u = 0 # Expected utility of this action.
                # For every possible next state ...
                for s_prime in maze.states[s].surroundings.values(): 
                    # Get probability of ending up in state s_prime 
                    # upon executing action a in state s.
                    transitionProb = maze.T(s, a, s_prime) 
                    if (transitionProb > 0):
                        # Get reward.
                        reward = (
                            maze.R(s, a, s_prime) + # Immediate reward + 
                            (gamma * V_old[s_prime]) # discounted future reward.
                        )
                        # Compute utility.
                        u += transitionProb * reward
                Q[a] = u # Keep track of this action's expected utility.
            V[s] = max(Q.values()) # Keep track of expected value of this state.
            state_diff.append(abs(V[s] - V_old[s])) # Measure change in state value.
        # Value iteration is considered to have converged 
        # if maximum change of all state values from state 
        # k to k+1 <= some small epsilon change.
        converged = np.max(state_diff) <= epsilon # Check convergence.
        k += 1 # Update iteration counts.
        if is_print:
            with open(f'./__output/{output}_dim{maze.matrix.shape[0]}.txt', 'a', encoding="utf-8") as f:
                f.write(f'\n\nIteration {k}:\n')
                f.write(values_to_mat_str(
                    v=V, shape=maze.matrix.shape,
                    start=maze.start, goal=maze.goal
                ))
    return V, k

def extract_policy(maze, V, gamma, is_print):
    """ 
    Given values of each state and the maze,
    extracts policy as being the action at each
    state which maximizes expected reward. Returns path.
    """
    print('\nExtracting policy ...')
    policy = {}
    for s in maze.state_positions: # For each valid state (not wall).
        Q = {} # Expected utility of each action.
        for a in maze.actions: # For each action ...
            u = 0 # Expected utility of this action.
            for s_prime in maze.states[s].surroundings.values(): # For every possible next state ...
                # Get probability of ending up in state s_prime 
                # upon executing action a in state s.
                transitionProb = maze.T(s, a, s_prime) 
                if (transitionProb > 0):
                    # Get reward.
                    reward = (
                        maze.R(s, a, s_prime) + # Immediate reward + 
                        (gamma * V[s_prime]) # discounted future reward.
                    )
                    # Compute utility.
                    u += transitionProb * reward
            Q[a] = u # Keep track of this action's expected utility.
        # Set action that resulted in maximum utility as policy for this state.
        u_max = max(Q.values())
        policy[s] = [a for a, u in Q.items() if u == u_max]
    if is_print:
        with open(f'__output/{output}_dim{maze.matrix.shape[0]}.txt', 'a', encoding="utf-8") as f:
            f.write(f'\n\nPolicy:\n')
            f.write(str(policy_to_mat(
                policy=policy, shape=maze.matrix.shape, 
                start=maze.start, goal=maze.goal
            )))
    return policy

@track_time
def value_iteration(maze, gamma=0.99, epsilon=1e-6, is_print=False, loop_resistent=False):
    if is_print: 
        with open(f'__output/{output}_dim{maze.matrix.shape[0]}.txt', 'w', encoding="utf-8") as f:
            f.write('Maze:\n')
            f.write(str(maze))

    V, num_iters = update_values(maze, gamma, epsilon, is_print)
    
    policy = extract_policy(maze, V, gamma, is_print=is_print)
    
    solution = get_solution(maze, policy)

    return {
        'solution': solution, 
        'state_values': values_to_mat(V, maze.matrix.shape),
        'policy': policy,
        'num_iterations': num_iters
    }

if __name__ == '__main__':
    # TEST MAZE
    # maze = Maze(dim=10)
    # save_maze(maze, dir='__mazes', filename=output)
    maze = Maze(matrix=load_maze('./__mazes/maze_latest_dim21.json'))
    res = value_iteration(maze, is_print=True, gamma=0.99)
    output = f'output_val_iter_dim{maze.matrix.shape[0]}'
    draw_maze(
        maze=maze.matrix, 
        solution=res['solution'], 
        state_values=res['state_values'],
        save={'dir':'__output/', 'filename':output, 'animation':False}
    )