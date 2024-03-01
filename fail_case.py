from value_iteration import value_iteration
from maze import Maze, load_maze, draw_maze
from utility import extract_solution_mdp, output_result

if __name__ == "__main__":
    # Demonstrating fail case when reward is a small constant,
    # and thus, need for reward as implemented now which
    # takes into consideration, values of epsilon and gamma.
    maze = Maze(matrix=load_maze(path='./__mazes/size101/1.json'))
    maze.reward = 100
    res = value_iteration(maze=maze, out_dir="__fail_case", out_file="101_1", epsilon=1e-6, gamma=0.99)
    output_result(result=res, out_dir='__fail_case', out_file='101_1')
    solution = extract_solution_mdp(maze=maze, policy=res['policy'], out_dir="__fail_case", out_file="101_1")
    draw_maze(maze=maze.matrix, save={'dir':"__fail_case", "filename": "101_1", "animation":False}, solution=solution, state_values=res['state_values'])