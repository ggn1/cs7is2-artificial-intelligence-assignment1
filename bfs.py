# Imports
import numpy as np
from queue import Queue
from track_time import eval_perf
from maze import create_maze, draw_maze

@eval_perf
def find_path(maze):
    # BFS algorithm to find the shortest path
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (1, 1)
    end = (maze.shape[0]-2, maze.shape[1]-2)
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0]+dx, node[1]+dy)
            if (next_node == end):
                return path + [next_node], len(visited)
            if (next_node[0] >= 0 and next_node[1] >= 0 and 
                next_node[0] < maze.shape[0] and next_node[1] < maze.shape[1] and 
                maze[next_node] == 0 and not visited[next_node]):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))

dim = int(input("Enter the dimension of the maze: "))
maze = create_maze(dim)
performance = find_path(maze)
print(performance)
draw_maze(maze=maze, save_filename='bfs_trial1', save_animation=False, path=performance['path'])