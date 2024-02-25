import os
from maze import Maze, save_maze, draw_maze

# # A very small maze that is easy to track manually.
# dim = 3
# size = dim*2+1
# print(f'Creating {size} x {size} mazes ...', end=' ')
# maze = Maze(dim=dim, num_goals=1)
# output = f'1'
# dir_out = f'./__mazes/size{maze.matrix.shape[0]}'
# if not os.path.exists(dir_out): os.makedirs(dir_out)
# save_maze(maze, dir=dir_out, filename=output)
# draw_maze(
#     maze=maze.matrix,
#     save={'dir':dir_out, 'filename':output, 'animation':False}
# )
# print('done!')

# # 2 Mazes each, of varying sizes [21x21, 61x61, 101x101]
# # such that for each size, there exists one maze with 
# # a single goal close to the start and another maze with
# # a single goal far away from the start.
# for dim in [10, 30, 50]:
#     size = dim*2+1
#     print(f'Creating {size} x {size} mazes ...', end=' ')
#     # Maze 1 = Goal close to start.
#     maze = Maze(dim=dim, num_goals=1, goals=[(size//3, size//3)])
#     output = f'1'
#     dir_out = f'./__mazes/size{maze.matrix.shape[0]}'
#     if not os.path.exists(dir_out): os.makedirs(dir_out)
#     save_maze(maze, dir=dir_out, filename=output)
#     draw_maze(
#         maze=maze.matrix,
#         save={'dir':dir_out, 'filename':output, 'animation':False}
#     )
#     # Maze 2 = Goal far from start.
#     maze = Maze(dim=dim, num_goals=1, goals=[(size-2, size-2)])
#     output = f'2'
#     dir_out = f'./__mazes/size{maze.matrix.shape[0]}'
#     if not os.path.exists(dir_out): os.makedirs(dir_out)
#     save_maze(maze, dir=dir_out, filename=output)
#     draw_maze(
#         maze=maze.matrix,
#         save={'dir':dir_out, 'filename':output, 'animation':False}
#     )
#     print('done!')

# # 40 Relatively small and easy to view mazes that have 2 goals.
# # This is so that 10 mazes can manually be extracted 
# # from the randomly generated ones such that in each of them, 
# # one goal is farther away from the start than the other.
# dim = 8
# size = dim*2+1
# print(f'Creating {size} x {size} mazes ...', end=' ')
# for i in range(40):
#     maze = Maze(dim=dim, num_goals=2)
#     output = f'{i+1}'
#     dir_out = f'./__mazes/size{maze.matrix.shape[0]}'
#     if not os.path.exists(dir_out): os.makedirs(dir_out)
#     save_maze(maze, dir=dir_out, filename=output)
#     draw_maze(
#         maze=maze.matrix,
#         save={'dir':dir_out, 'filename':output, 'animation':False}
#     )
# print('done!')