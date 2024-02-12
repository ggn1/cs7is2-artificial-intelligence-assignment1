from maze import create_maze, save_maze, load_maze, draw_maze
from dfs import dfs

dim = 20
m, g = create_maze(dim)
res = dfs(m, g)
draw_maze(
    maze=m, path=res['path'], 
    # save_dir='./solutions', save_filename=f'/maze_{len(m)}', save_animation=True
)