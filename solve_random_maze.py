import os
import argparse
from datetime import datetime
from maze import Maze, draw_maze
from dfs import dfs
from bfs import bfs
from a_star import a_star
from value_iteration import value_iteration
from policy_iteration import policy_iteration
from utility import reconstruct_path, extract_solution_mdp, output_result

def solve_dfs(dim, num_goals):
    """ Solve a random maze using DFS. """
    out_dir = 'out_dfs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = datetime.now().strftime("%m%d%Y%H%M%S")
    maze = Maze(dim=dim, num_goals=num_goals) # Generate maze.
    with open(f"{out_dir}/{out_file}.txt", 'w') as f:
        f.write(f'MAZE:\n{str(maze)}')
    res = dfs(maze=maze)
    if res['goal'] is None: # No goal found => no solution.
        print("No solution found.")
    else:
        solution = reconstruct_path(res['parents'], start=maze.start, goal=res['goal'])
        res['solution'] = solution
        draw_maze(
            maze=maze.matrix, solution=solution, exploration=list(res['parents'].keys()),
            save={'dir':out_dir, 'filename':out_file, 'animation':False}
        )
    output_result(result=res, out_dir=out_dir, out_file=out_file)

def solve_bfs(dim, num_goals):
    """ Solve a random maze using BFS. """
    out_dir = 'out_bfs'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = datetime.now().strftime("%m%d%Y%H%M%S")
    maze = Maze(dim=dim, num_goals=num_goals) # Generate maze.
    with open(f"{out_dir}/{out_file}.txt", 'w') as f:
        f.write(f'MAZE:\n{str(maze)}')
    res = bfs(maze=maze)
    if res['goal'] is None: # No goal found => no solution.
        print("No solution found.")
    else:
        solution = reconstruct_path(res['parents'], start=maze.start, goal=res['goal'])
        res['solution'] = solution
        draw_maze(
            maze=maze.matrix, solution=solution, exploration=list(res['parents'].keys()),
            save={'dir':out_dir, 'filename':out_file, 'animation':False}
        )
    output_result(result=res, out_dir=out_dir, out_file=out_file)

def solve_astar(dim, num_goals):
    """ Solve a random maze using A* search. """
    out_dir = 'out_astar'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = datetime.now().strftime("%m%d%Y%H%M%S")
    maze = Maze(dim=dim, num_goals=num_goals) # Generate maze.
    with open(f"{out_dir}/{out_file}.txt", 'w') as f:
        f.write(f'MAZE:\n{str(maze)}')
    res = a_star(maze=maze)
    if res['goal'] is None: # No goal found => no solution.
        print("No solution found.")
    else:
        solution = reconstruct_path(res['parents'], start=maze.start, goal=res['goal'])
        res['solution'] = solution
        draw_maze(
            maze=maze.matrix, solution=solution, exploration=list(res['parents'].keys()),
            save={'dir':out_dir, 'filename':out_file, 'animation':False}
        )
    output_result(result=res, out_dir=out_dir, out_file=out_file)

def solve_valiter(dim, num_goals, gamma, epsilon, max_iters):
    """ Solve a random maze using value iteration. """
    out_dir = 'out_valiter'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = datetime.now().strftime("%m%d%Y%H%M%S")
    maze = Maze(dim=dim, num_goals=num_goals) # Generate maze.
    res = value_iteration(
        maze=maze, out_dir=out_dir, out_file=out_file, 
        gamma=gamma, epsilon=epsilon, max_iters=max_iters
    )
    solution = extract_solution_mdp(
        maze=maze, policy=res['policy'], 
        out_dir=out_dir, out_file=out_file
    )
    res['solution'] = solution
    draw_maze(
        maze=maze.matrix, solution=solution, state_values=res['state_values'],
        save={'dir':out_dir, 'filename':out_file, 'animation':False}
    )
    output_result(result=res, out_dir=out_dir, out_file=out_file)

def solve_politer(dim, num_goals, gamma, max_iters):
    """ Solve a random maze using policy iteration. """
    out_dir = 'out_politer'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = datetime.now().strftime("%m%d%Y%H%M%S")
    maze = Maze(dim=dim, num_goals=num_goals) # Generate maze.
    res = policy_iteration(
        maze=maze, out_dir=out_dir, out_file=out_file, 
        gamma=gamma, max_iters=max_iters
    )
    solution = extract_solution_mdp(
        maze=maze, policy=res['policy'], 
        out_dir=out_dir, out_file=out_file
    )
    res['solution'] = solution
    draw_maze(
        maze=maze.matrix, solution=solution, state_values=res['state_values'],
        save={'dir':out_dir, 'filename':out_file, 'animation':False}
    )
    output_result(result=res, out_dir=out_dir, out_file=out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Solve a random maze without saving it.')
    parser.add_argument('-dim', '--dimension', type=int, required=True, help="Maze dimension.")
    parser.add_argument('-ng', '--num-goals', type=int, required=True, help="No. of goals that the maze should have.")
    parser.add_argument('-s', '--solver', type=str, required=True, help="Name of solver. Can be dfs, bfs, astar, valiter, or polyiter")
    parser.add_argument('-df', '--discount-factor', type=float, help="Discount facotr = gamma.")
    parser.add_argument('-ct', '--change-threshold', type=float, help="Change threshold = epsilon.")
    parser.add_argument('-mi', '--max-iters', type=int, help="Maximum iterations limit.")
    args = parser.parse_args()

    if args.solver == 'dfs':
        solve_dfs(dim=args.dimension, num_goals=args.num_goals)

    if args.solver == 'bfs':
        solve_bfs(dim=args.dimension, num_goals=args.num_goals)

    if args.solver == 'astar':
        solve_astar(dim=args.dimension, num_goals=args.num_goals)

    if args.solver == 'valiter':
        gamma = args.discount_factor if args.discount_factor else 0.99
        epsilon = args.change_threshold if args.change_threshold else 1e-6
        max_iters = args.max_iters if args.max_iters else None
        solve_valiter(dim=args.dimension, num_goals=args.num_goals, gamma=gamma, epsilon=epsilon, max_iters=max_iters)

    if args.solver == 'politer':
        gamma = args.discount_factor if args.discount_factor else 0.99
        epsilon = args.change_threshold if args.change_threshold else 1e-6
        max_iters = args.max_iters if args.max_iters else None
        solve_politer(dim=args.dimension, num_goals=args.num_goals, gamma=gamma, max_iters=max_iters)
