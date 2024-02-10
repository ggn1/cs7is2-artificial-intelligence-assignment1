# Running this python file creates and saves 3 mazes
# (small, medium, large) that shall be used to
# evaluate each solver.

# Imports.
from maze import create_maze, save_maze

# Create mazes.
maze_s = create_maze(5) # maze size = 11 * 11
maze_m = create_maze(15) # maze size = 31 * 31
maze_l = create_maze(45) # maze size = 91 * 91

# Save mazes.
save_maze(maze=maze_s, dir='./mazes', filename='maze_11')
save_maze(maze=maze_m, dir='./mazes', filename='maze_31')
save_maze(maze=maze_l, dir='./mazes', filename='maze_91')