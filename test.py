from maze import create_maze, save_maze, load_maze, draw_maze
from dfs import dfs
from bfs import bfs

dim = 20
m, g = create_maze(dim)
res = bfs(m, g)
# print('PATH =', res['path'])
# print('EXPLORATION =', res['exploration'])
draw_maze(
    maze=m, path=res['path'], 
    exploration=res['exploration'],
    # save_dir='./solutions', save_filename=f'/maze_{len(m)}', save_animation=True
)