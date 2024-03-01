import random
from maze import draw_maze, load_maze, Maze
from utility import policy_to_mat, values_to_mat, values_to_mat_str, solve_maze, output_result, track_mem_time

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
                    maze.R(s) + # Immediate reward + 
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
                        maze.R(s) + # Immediate reward + 
                        (gamma * V[s_prime]) # Discounted future reward.
                    )
                    u += transition_prob * reward
            Q[a] = u
        u_max = max(Q.values())
        policy[s] = [a for a, u in Q.items() if u == u_max] # Update policy.
    return policy

@track_mem_time
def policy_iteration(maze, out_file, out_dir, gamma, max_iters=None):
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
        if (not max_iters is None and k >= max_iters):
            with open(f'{out_dir}/{out_file}.txt', 'a', encoding="utf-8") as f:
                f.write(f'\n\nMax no. of iterations met.')
            break
        V = policy_evaluation(maze, V, policy, gamma)
        policy_improved = policy_improvement(maze, V, gamma)
        # If no change in policy after improvement,
        # then, policy has converged.
        converged = all(policy_improved[s] == policy[s] for s in maze.state_positions)
        policy = policy_improved # Else, prep for next round of evaluation and improvement.
        k += 1
        with open(f'{out_dir}/{out_file}.txt', 'a', encoding="utf-8") as f:
            f.write(f'\n\nIteration {k}:\n')
            f.write(values_to_mat_str(
                v=V, shape=maze.matrix.shape, 
                start=maze.start, goals=maze.goals
            ))
        with open(f'{out_dir}/{out_file}.txt', 'a', encoding="utf-8") as f:
            f.write(f'\nPolicy:\n')
            f.write(str(policy_to_mat(
                policy=policy, shape=maze.matrix.shape, 
                start=maze.start, goals=maze.goals
            )))

    return {
        'state_values': values_to_mat(V, maze.matrix.shape),
        'policy': policy,
        'num_iterations': k
    }

def __conduct_experiments(sizes, id_nums, load_dir, save_dir, gamma, max_iters):
    for maze_size in sizes:
        for i in id_nums:
            print(f'Solving {maze_size} x {maze_size} maze {i} ...')
            out_file = f'{i}_politer'
            out_dir = f'{save_dir}/size{maze_size}'
            maze = Maze(
                matrix=load_maze(path=f"{load_dir}/size{maze_size}/{i}.json"),
                max_gamma=0.99, min_epsilon=1e-6
            )
            res = solve_maze(
                solver_type='policy-iteration',
                solver=policy_iteration, maze=maze, 
                out_dir=out_dir, out_file=out_file, 
                gamma=gamma, max_iters=max_iters
            )
            draw_maze(
                maze=maze.matrix, 
                solution=res['solution'], 
                state_values=res['state_values'],
                save={'dir':out_dir, 'filename':out_file, 'animation':True},
            )
            print('Done!\n')

if __name__ == '__main__':
    load_dir = '__mazes_old'
    save_dir = '__mazes'

    # Solve 1 maze each of varying small sizes with 1 goal.
    __conduct_experiments(
        sizes=[7, 15], id_nums=[1], 
        load_dir=load_dir, save_dir=save_dir,
        gamma=0.99, max_iters=None
    )

    # Solve 3 medium sized mazes with 1 goal.
    __conduct_experiments(
        sizes=[21], id_nums=list(range(1, 4)), 
        load_dir=load_dir, save_dir=save_dir,
        gamma=0.99, max_iters=None
    )

    # Solve 5 medium sized mazes with 2 goals.
    __conduct_experiments(
        sizes=[31], id_nums=list(range(1, 6)), 
        load_dir=load_dir, save_dir=save_dir,
        gamma=0.99, max_iters=None
    )

    # Solve 3 large mazes with 1 goal.
    __conduct_experiments(
        sizes=[61, 101], id_nums=list(range(1, 4)), 
        load_dir=load_dir, save_dir=save_dir,
        gamma=0.98, max_iters=(101**2//2)
    )