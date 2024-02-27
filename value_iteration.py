import numpy as np
from maze import draw_maze, load_maze, Maze
from utility import track_mem_time, policy_to_mat, values_to_mat, values_to_mat_str, solve_maze

def update_values(maze, gamma, epsilon, out_file, out_dir):
    """ 
    Performs value iteration and returns values for each state
    and no. of iterations before convergence. 
    """
    converged = False # Keep track of convergence.
    k = 0 # Keep track of kth iteration.
    print('Updating utility values ...')
    # Initially, value associated with every valid position is 0.
    # v_init = float(maze.matrix.shape[0] ** 2)
    v_init = 0.0
    state_values = {state: v_init for state in maze.state_positions}
    while(not converged): # Repeat until convergence.
        state_diff = [] # Absolute difference between state values now v/s before.
        V_old = state_values.copy() # Since values will change in this iteration, keep a copy of old ones.
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
            state_values[s] = max(Q.values()) # Keep track of expected value of this state.
            state_diff.append(abs(state_values[s] - V_old[s])) # Measure change in state value.
        # Value iteration is considered to have converged 
        # if maximum change of all state values from state 
        # k to k+1 <= some small epsilon change.
        converged = np.max(state_diff) <= epsilon # Check convergence.
        k += 1 # Update iteration counts.
        
        # Output iteration result.
        with open(f'{out_dir}/{out_file}.txt', 'a', encoding="utf-8") as f:
            f.write(f'\n\nIteration {k}:\n')
            f.write(values_to_mat_str(
                v=state_values, shape=maze.matrix.shape,
                start=maze.start, goals=maze.goals
            ))
    return state_values, k

def extract_policy(maze, state_values, gamma, out_file, out_dir):
    """ 
    Given values of each state and the maze,
    extracts policy as being the action at each
    state which maximizes expected reward. Returns path.
    """
    print('Extracting policy ...')
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
                        (gamma * state_values[s_prime]) # discounted future reward.
                    )
                    # Compute utility.
                    u += transitionProb * reward
            Q[a] = u # Keep track of this action's expected utility.
        # Set action that resulted in maximum utility as policy for this state.
        u_max = max(Q.values())
        policy[s] = [a for a, u in Q.items() if u == u_max]

    # Output results.
    with open(f'{out_dir}/{out_file}.txt', 'a', encoding="utf-8") as f:
        f.write(f'\n\nPolicy:\n')
        f.write(str(policy_to_mat(
            policy=policy, shape=maze.matrix.shape, 
            start=maze.start, goals=maze.goals
        )))
    return policy

@track_mem_time
def value_iteration(maze, out_file, out_dir, gamma=0.99, epsilon=1e-6):
    with open(f'{out_dir}/{out_file}.txt', 'w', encoding="utf-8") as f:
        f.write('Maze:\n')
        f.write(str(maze))

    state_values, num_iters = update_values(
        maze=maze, gamma=gamma, epsilon=epsilon, 
        out_file=out_file, out_dir=out_dir
    )
    
    policy = extract_policy(
        maze=maze, state_values=state_values, gamma=gamma,
        out_file=out_file, out_dir=out_dir
    )
    
    return {
        'state_values': values_to_mat(state_values, maze.matrix.shape),
        'policy': policy,
        'num_iterations': num_iters
    }

def __conduct_experiments(sizes, id_nums):
    for maze_size in sizes:
        for i in id_nums:
            print(f'Solving {maze_size} x {maze_size} maze {i} ...')
            out_file = f'{i}_valiter'
            out_dir = f'__mazes/size{maze_size}'
            maze = Maze(matrix=load_maze(path=f"{out_dir}/{i}.json"))
            res = solve_maze(
                solver_type='value-iteration',
                solver=value_iteration, maze=maze, 
                out_dir=out_dir, out_file=out_file, 
                gamma=0.99, epsilon=1e-6
            )
            draw_maze(
                maze=maze.matrix, 
                solution=res['solution'], 
                state_values=res['state_values'],
                save={'dir': out_dir, 'filename': out_file, 'animation': True},
            )
            print('Done!\n')

if __name__ == '__main__':
    # Solve 1 maze each of varying sizes with 1 goal.
    __conduct_experiments(sizes=[7, 15, 21, 61, 101], id_nums=[1])
    
    # Solve 5 31x31 mazes with 2 goals.
    __conduct_experiments(sizes=[31], id_nums=list(range(1, 6)))

    # Trying different values of epsilon.
    maze_size = 15
    for epsilon in [1e5, 150, 0]:
        print(f'Solving {maze_size} x {maze_size} maze 1 epsilon = {epsilon}...')
        out_file = f'1_valiter_epsilon{epsilon}'
        out_dir = f'__mazes/size{maze_size}'
        maze = Maze(matrix=load_maze(path=f"{out_dir}/1.json"))
        res = solve_maze(
            solver_type='value-iteration',
            solver=value_iteration, maze=maze, 
            out_dir=out_dir, out_file=out_file, 
            gamma=0.99, epsilon=epsilon
        )
        draw_maze(
            maze=maze.matrix, 
            solution=res['solution'], 
            state_values=res['state_values'],
            save={'dir':out_dir, 'filename':out_file, 'animation':True},
        )
        print('Done!\n')