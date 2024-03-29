import os
import argparse
from maze import Maze, save_maze, draw_maze

def init_mazes(dims, num_mazes, num_goals, folder, min_epsilon, max_gamma):
    """ 
    Initializes num_mazes no. of mazes for each given 
    dimension with num_goals no. of goals and saves 
    them into given folder.
    """
    for dim in dims:
        size = dim*2+1
        print(f'Creating {size} x {size} mazes ...', end=' ')
        for i in range(num_mazes):
            maze = Maze(dim=dim, num_goals=num_goals, min_epsilon=min_epsilon, max_gamma=max_gamma)
            out_file = f'{i+1}'
            out_dir = f'{folder}/size{maze.matrix.shape[0]}'
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            save_maze(maze=maze, out_dir=out_dir, out_file=out_file)
            draw_maze(
                maze=maze.matrix,
                save={'dir':out_dir, 'filename':out_file, 'animation':False}
            )
        print('done!')

# This file was used to create and save random mazes 
# for assignment 1 experiments.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Initialize Mazes')
    parser.add_argument('-dst', '--dst-dir', type=str, help="Directory where mazes are to be saved.")
    args = parser.parse_args()

    # 1 maze each of varying sizes [7x7, 15x15] with 1 goal.
    init_mazes(
        dims=[3, 7], num_mazes=1, num_goals=1, folder=args.dst_dir,
        min_epsilon=1e-6, max_gamma=0.99
    )

    # 3 mazes of varying sizes [21x21, 61x61, 101x101] with 1 goal.
    init_mazes(
        dims=[10, 30, 50], num_mazes=3, num_goals=1, folder=args.dst_dir,
        min_epsilon=1e-6, max_gamma=0.99
    )

    # 20 31x31 mazes with 2 goals.
    # This is so that 5 mazes can manually be extracted 
    # from the randomly generated ones such that in each of them, 
    # one goal is farther away from the start than the other.
    init_mazes(
        dims=[15], num_mazes=20, num_goals=2, folder=args.dst_dir, 
        min_epsilon=1e-6, max_gamma=0.99
    )

    # # For video demo.
    # # One 41x41 maze with 1 goal.
    # init_mazes(
    #     dims=[20], num_mazes=1, num_goals=1, folder="./__demo", 
    #     min_epsilon=1e-6, max_gamma=0.99
    # )