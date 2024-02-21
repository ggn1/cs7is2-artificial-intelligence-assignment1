# Running this python file creates and saves 3 mazes
# (small, medium, large) that shall be used to
# evaluate each solver.

# Imports.
from maze import Maze, save_maze, draw_maze

# Create mazes.
maze_t = Maze(dim=2) # maze size = 5 * 5
# maze_s = Maze(dim=10) # maze size = 21 * 21
# maze_m = Maze(dim=20) # maze size = 31 * 31
# maze_l = Maze(dim=50) # maze size = 91 * 91

# Save mazes.
save_maze(maze=maze_t, dir='./mazes', filename=f't')
# save_maze(maze=maze_s, dir='./mazes', filename=f's')
# save_maze(maze=maze_m, dir='./mazes', filename=f'm')
# save_maze(maze=maze_l, dir='./mazes', filename=f'l')

# Draw mazes
draw_maze(maze=maze_t.maze, save_dir='./mazes', save_filename=f't_dim{maze_t.maze.shape[0]}')
# draw_maze(maze=maze_s.maze, save_dir='./mazes', save_filename=f's_dim{maze_s.maze.shape[0]}')
# draw_maze(maze=maze_m.maze, save_dir='./mazes', save_filename=f'm_dim{maze_m.maze.shape[0]}')
# draw_maze(maze=maze_l.maze, save_dir='./mazes', save_filename=f'l_dim{maze_l.maze.shape[0]}')