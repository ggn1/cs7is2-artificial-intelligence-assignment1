import os
from maze import Maze, save_maze, draw_maze

for dim in [6, 10, 30, 50]:
    size = dim*2+1
    print(f'Creating {size} x {size} mazes ...', end=' ')
    for i in range(2): # Create 42 (size x size) mazes. 1 with 2 goals and 1 with 1 goal.
        maze = Maze(dim=dim, num_goals=1 if i==1 else 2)
        output = f'{i+1}'
        dir_out = f'./__mazes/size{maze.matrix.shape[0]}'
        if not os.path.exists(dir_out): os.makedirs(dir_out)
        save_maze(maze, dir=dir_out, filename=output)
        draw_maze(
            maze=maze.matrix,
            save={'dir':dir_out, 'filename':output, 'animation':False}
        )
    print('done!')