# Running this python file creates and saves 3 mazes
# (small, medium, large) that shall be used to
# evaluate each solver.

# Imports.
from maze import create_maze, save_maze, draw_maze

# Create mazes.
maze_t = create_maze(2) # maze size = 5 * 5
maze_xs = create_maze(3) # maze size = 7 * 7
maze_s = create_maze(5) # maze size = 11 * 11
maze_m = create_maze(15) # maze size = 31 * 31
maze_l = create_maze(45) # maze size = 91 * 91

# Save mazes.
save_maze(maze=maze_t, dir='./mazes', filename=f'maze_{len(maze_t)}')
save_maze(maze=maze_xs, dir='./mazes', filename=f'maze_{len(maze_xs)}')
save_maze(maze=maze_s, dir='./mazes', filename=f'maze_{len(maze_s)}')
save_maze(maze=maze_m, dir='./mazes', filename=f'maze_{len(maze_m)}')
save_maze(maze=maze_l, dir='./mazes', filename=f'maze_{len(maze_l)}')

# Draw mazes
draw_maze(maze=maze_t, save_dir='./mazes', save_filename=f'maze_{len(maze_t)}')
draw_maze(maze=maze_xs, save_dir='./mazes', save_filename=f'maze_{len(maze_xs)}')
draw_maze(maze=maze_s, save_dir='./mazes', save_filename=f'maze_{len(maze_s)}')
draw_maze(maze=maze_m, save_dir='./mazes', save_filename=f'maze_{len(maze_m)}')
draw_maze(maze=maze_l, save_dir='./mazes', save_filename=f'maze_{len(maze_l)}')