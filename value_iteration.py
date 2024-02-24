# Imports 
import numpy as np
from track_time import track_time
from utility import print_mat_2d, v_to_mat
from maze import draw_maze, load_maze, Maze

def update_values(maze, gamma, epsilon, print_values):
    """ 
    Performs value iteration and returns values for each state
    and no. of iterations before convergence. 
    """
    converged = False # Keep track of convergence.
    k = 0 # Keep track of kth iteration.
    print('\nUpdating utility values ...')
    # Initially, value associated with every valid position is 0.
    V = {state:0 for state in maze.state_positions}
    while(not converged): # Repeat until convergence.
        state_max_diff = 0 # Max difference between value of any state now v/s before.
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
            state_diff = abs(V[s] - V_old[s]) # Measure change in state value.
            if state_diff > state_max_diff: # Track max change observed in any state.
                state_max_diff = state_diff
        # Value iteration is considered to have converged 
        # if maximum change of all state values from state 
        # k to k+1 <= some small epsilon change.
        converged = state_max_diff <= epsilon # Check convergence.
        k += 1 # Update iteration counts.
    if print_values:
        print(f'Iteration {k}:')
        print_mat_2d(v_to_mat(v=V, shape=maze.matrix.shape))
    return V, k

def extract_policy(maze, V, gamma):
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
        policy[s] = max(Q, key=Q.get)
    return policy

def get_solution(maze, policy):
    """ 
    Given returns the solution (path from start 
    to goal) to the given maze as per given policy.
    """
    s =  maze.start # Begin at the start state.
    solution = []
    while (s != maze.goal): # Until the goal state is reached ...
        if not s in policy:
            print('No solution found.')
            break
        a = policy[s] # Get best action for this state as per policy.
        s = maze.states[s][a] # Get next state as per policy.
        if (s in solution): # Break and report if a loop was detected.
            print('Loop')
            break
        solution.append(s) # Append state to the solution.
    return solution

@track_time
def value_iteration(maze, gamma=0.99, epsilon=1e-2, print_values=False):
    if print_values: 
        print('Maze:')
        print_mat_2d(maze.matrix)

    V, num_iters = update_values(maze, gamma, epsilon, print_values)
    
    policy = extract_policy(maze, V, gamma)
    
    solution = get_solution(maze, policy)

    return {
        'solution': solution,
        'state_values': v_to_mat(V, maze.matrix.shape),
        'policy': policy,
        'num_iterations': num_iters
    }

if __name__ == '__main__':
    # TEST MAZE
    maze = Maze(dim=50)
    res = value_iteration(maze, print_values=False)
    draw_maze(
        maze=maze.matrix, 
        solution=res['solution'], 
        state_values=res['state_values']
    )

    # # TINY MAZE
    # maze = Maze(maze=load_maze(path='./mazes/t_dim5.json'))
    # res = solver(maze, print_values=True)
    # draw_maze(maze=maze.matrix, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.matrix.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # # SMALL MAZE
    # maze = Maze(maze=load_maze(path='./mazes/s_dim21.json'))
    # res = solver(maze)
    # draw_maze(maze=maze.matrix, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.matrix.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # # MEDIUM MAZE
    # maze = Maze(maze=load_maze(path='./mazes/m_dim41.json'))
    # res = solver(maze)
    # draw_maze(maze=maze.matrix, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.matrix.shape[0]}', save_animation=False, path=res['path'], v=res['v'])

    # # LARGE MAZE
    # maze = Maze(maze=load_maze(path='./mazes/l_dim101.json'))
    # res = solver(maze)
    # draw_maze(maze=maze.matrix, save_dir='./solutions', save_filename=f'mdpvi_dim{maze.matrix.shape[0]}', save_animation=False, path=res['path'], v=res['v'])