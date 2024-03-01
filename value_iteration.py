import numpy as np
from maze import draw_maze, load_maze, Maze
from utility import track_mem_time, policy_to_mat, values_to_mat, values_to_mat_str, solve_maze, extract_solution_mdp

def update_values(maze, gamma, epsilon, out_file, out_dir, max_iters=None):
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
    V = {state: v_init for state in maze.state_positions}
    while(not converged): # Repeat until convergence.
        if (not max_iters is None and k >= max_iters):
            with open(f'{out_dir}/{out_file}.txt', 'a', encoding="utf-8") as f:
                f.write(f'\n\nMax no. of iterations met.')
            break
        delta = 0 # Absolute difference between state values now v/s before.
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
                            maze.R(s) + # Immediate reward + 
                            (gamma * V_old[s_prime]) # discounted future reward.
                        )
                        # Compute utility.
                        u += transitionProb * reward
                Q[a] = u # Keep track of this action's expected utility.
            V[s] = max(Q.values()) # Keep track of expected value of this state.
            delta = max(delta, np.abs(V[s] - V_old[s])) # Max change in state values.
        # Value iteration is considered to have converged 
        # if maximum change of all state values from state 
        # k to k+1 <= some small epsilon change.
        # converged = np.max(state_diff) <= epsilon # Check convergence.
        converged = delta <= epsilon # Check convergence.
        k += 1 # Update iteration counts.
        
        # Output iteration result.
        with open(f'{out_dir}/{out_file}.txt', 'a', encoding="utf-8") as f:
            f.write(f'\n\nIteration {k}:\n')
            f.write(values_to_mat_str(
                v=V, shape=maze.matrix.shape,
                start=maze.start, goals=maze.goals
            ))

    return V, k

def extract_policy(maze, V, gamma, out_file, out_dir):
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
                        maze.R(s) + # Immediate reward + 
                        (gamma * V[s_prime]) # discounted future reward.
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
def value_iteration(maze, out_file, out_dir, gamma, epsilon, max_iters):
    with open(f'{out_dir}/{out_file}.txt', 'w', encoding="utf-8") as f:
        f.write('Maze:\n')
        f.write(str(maze))

    V, num_iters = update_values(
        maze=maze, gamma=gamma, epsilon=epsilon, 
        out_file=out_file, out_dir=out_dir, max_iters=max_iters
    )

    policy = extract_policy(
        maze=maze, V=V, gamma=gamma,
        out_file=out_file, out_dir=out_dir
    )
    
    return {
        'state_values': values_to_mat(V, maze.matrix.shape),
        'policy': policy,
        'num_iterations': num_iters
    }

def __conduct_experiments(sizes, id_nums, load_dir, save_dir, epsilon, gamma, max_iters):
    for maze_size in sizes:
        for i in id_nums:
            print(f'Solving {maze_size} x {maze_size} maze {i} ...')
            out_file = f'{i}_valiter'
            out_dir = f'{save_dir}/size{maze_size}'
            maze = Maze(
                matrix=load_maze(path=f"{load_dir}/size{maze_size}/{i}.json"),
                max_gamma=0.99, min_epsilon=1e-6
            )
            res = solve_maze(
                solver_type='value-iteration',
                solver=value_iteration, maze=maze, 
                out_dir=out_dir, out_file=out_file, 
                gamma=gamma, epsilon=epsilon, max_iters=max_iters
            )
            draw_maze(
                maze=maze.matrix, 
                solution=res['solution'], 
                state_values=res['state_values'],
                save={'dir': out_dir, 'filename': out_file, 'animation': True},
            )
            print('Done!\n')

if __name__ == '__main__':
    # out_dir = './__temp'
    # out_file = 'size101_1'
    # maze = Maze(
    #     matrix=load_maze(path='./__mazes/size101/1.json'), 
    #     max_gamma=0.99, min_epsilon=1e-6
    # )
    # res = value_iteration(
    #     maze=maze, out_dir=out_dir, out_file=out_file, 
    #     gamma=0.99, epsilon=1e-6
    # )
    # solution = extract_solution_mdp(
    #     maze=maze, policy=res['policy'], 
    #     out_dir=out_dir, out_file=out_file
    # )
    # draw_maze(
    #     maze=maze.matrix, solution=solution,
    #     save={'dir':out_dir, 'filename': out_file, 'animation':False}
    # )

    load_dir = '__mazes_old'
    save_dir = '__mazes'

    # # Solve 1 maze each of varying small sizes with 1 goal.
    # __conduct_experiments(
    #     sizes=[7, 15], id_nums=[1], 
    #     load_dir=load_dir, save_dir=save_dir,
    #     epsilon=1e-6, gamma=0.99, max_iters=None
    # )

    # # Solve 3 medium sized mazes with 1 goal.
    # __conduct_experiments(
    #     sizes=[21], id_nums=list(range(1, 4)), 
    #     load_dir=load_dir, save_dir=save_dir,
    #     epsilon=1e-6, gamma=0.99, max_iters=None
    # )

    # # Solve 5 medium sized mazes with 2 goals.
    # __conduct_experiments(
    #     sizes=[31], id_nums=list(range(1, 6)), 
    #     load_dir=load_dir, save_dir=save_dir,
    #     epsilon=1e-6, gamma=0.99, max_iters=None
    # )

    # # Solve 3 large mazes with 1 goal.
    # __conduct_experiments(
    #     sizes=[61, 101], id_nums=list(range(1, 4)), 
    #     load_dir=load_dir, save_dir=save_dir,
    #     epsilon=1e-3, gamma=0.98, max_iters=(101**2//2)
    # )