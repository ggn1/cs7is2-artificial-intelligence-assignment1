# Running this python file creates and saves 3 mazes
# (small, medium, large) that shall be used to
# evaluate each solver.

# Imports.
from maze import create_maze, save_maze, draw_maze

# Create mazes.
maze_s = create_maze(10) # maze size = 11 * 11
maze_m = create_maze(20) # maze size = 31 * 31
maze_l = create_maze(50) # maze size = 91 * 91

# Save mazes.
save_maze(maze=maze_s, dir='./mazes', filename=f's')
save_maze(maze=maze_m, dir='./mazes', filename=f'm')
save_maze(maze=maze_l, dir='./mazes', filename=f'l')

# Draw mazes
draw_maze(maze=maze_s, save_dir='./mazes', save_filename=f's_dim{len(maze_s)}')
draw_maze(maze=maze_m, save_dir='./mazes', save_filename=f'm_dim{len(maze_m)}')
draw_maze(maze=maze_l, save_dir='./mazes', save_filename=f'l_dim{len(maze_l)}')