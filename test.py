from maze import create_maze, save_maze, load_maze, draw_maze
from dfs import dfs

dim = 20
m, g = create_maze(dim)
save_maze(m, g, './mazes', '0')
m, g = load_maze(f'./mazes/0_dim{dim*2+1}_goalx{g[0]}y{g[1]}.json')
res = dfs(m, g)
draw_maze(maze=m, path=res['path'])