import os
from maze import Maze, save_maze, draw_maze

# This file was used to create and save random mazes 
# for assignment 1 experiments.

# A very small maze that is easy to track manually.
dim = 3
size = dim*2+1
print(f'Creating {size} x {size} mazes ...', end=' ')
maze = Maze(dim=dim, num_goals=1)
out_file = f'1'
out_dir = f'./__mazes/size{maze.matrix.shape[0]}'
if not os.path.exists(out_dir): os.makedirs(out_dir)
save_maze(maze, out_dir=out_dir, out_file=out_file)
draw_maze(
    maze=maze.matrix,
    save={'dir':out_dir, 'filename':out_file, 'animation':False}
)
print('done!')

# 2 Mazes each, of varying sizes [21x21, 61x61, 101x101]
# such that for each size, there exists one maze with 
# a single goal close to the start and another maze with
# a single goal far away from the start.
for dim in [10, 30, 50]:
    size = dim*2+1
    print(f'Creating {size} x {size} mazes ...', end=' ')
    # Maze 1 = Goal close to start.
    maze = Maze(dim=dim, num_goals=1, goals=[(size//3, size//3)])
    out_file = f'1'
    out_dir = f'./__mazes/size{maze.matrix.shape[0]}'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    save_maze(maze, dir=out_dir, filename=out_file)
    draw_maze(
        maze=maze.matrix,
        save={'dir':out_dir, 'filename':out_file, 'animation':False}
    )
    # Maze 2 = Goal far from start.
    maze = Maze(dim=dim, num_goals=1, goals=[(size-2, size-2)])
    out_file = f'2'
    out_dir = f'./__mazes/size{maze.matrix.shape[0]}'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    save_maze(maze, dir=out_dir, filename=out_file)
    draw_maze(
        maze=maze.matrix,
        save={'dir':out_dir, 'filename':out_file, 'animation':False}
    )
    print('done!')

# 40 Relatively small and easy to view mazes that have 2 goals.
# This is so that 10 mazes can manually be extracted 
# from the randomly generated ones such that in each of them, 
# one goal is farther away from the start than the other.
dim = 8
size = dim*2+1
print(f'Creating {size} x {size} mazes ...', end=' ')
for i in range(40):
    maze = Maze(dim=dim, num_goals=2)
    out_file = f'{i+1}'
    out_dir = f'./__mazes/size{maze.matrix.shape[0]}'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    save_maze(maze, dir=out_dir, filename=out_file)
    draw_maze(
        maze=maze.matrix,
        save={'dir':out_dir, 'filename':out_file, 'animation':False}
    )
print('done!')