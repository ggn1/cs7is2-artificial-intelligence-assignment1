import random
import numpy as np
from queue import LifoQueue
from track_time import track_time
from maze import create_maze, draw_maze, load_maze

def dfs(goal, maze, lifo_queue, visited=[], path=[]):
    ''' Recursive implementation of Depth First Search for maze solving. 
        @param goal: Goal position.
        @param maze: Maze to solve.
        @param lifo_queue: Stack (LIFO Queue) of nodes to visit.
        @param visited: List of visited nodes.
        @param path: Solution path so far.
    '''
    if lifo_queue.empty(): return path # If stack is empty, return path.
    node = lifo_queue.get() # Get last added node from the LIFO queue.
    visited.append(node) # This is the node we visit in this step of DFS.
    # path.append(node) # Since this nide has been visited now, add it to the path.
    if node == goal: 
        path.append(node)
        return path # If this node is the goal, return path.
    is_dead_end = True # Assume this is a dead end.
    for node_to_visit in [ # For every position adjacent to this node...
        (node[0] + d[0], node[1] + d[1]) 
        for d in [(0, 1), (1, 0), (0, -1), (-1, 0)] # up, right, down, left
    ]:
        if (
            # If the adjacent position in this direction has not yet been
            # visited, is within the maze (i.e. does not lie beyond the 
            # maze's dimensions) and is a path (not a wall), then and 
            # only then, add it to the LIFO queue (i.e. list of nodes to visit).
            not node_to_visit in visited and
            node_to_visit[0] > 0 and
            node_to_visit[0] < maze.shape[0] and
            node_to_visit[1] > 0 and
            node_to_visit[1] < maze.shape[1] and
            maze[node_to_visit] == 0
        ): 
            lifo_queue.put(node_to_visit)
            is_dead_end = False # This is not a dead end.
    if not is_dead_end: # If this was not a dead end ...
        path.append(node) # Add this node to the path.
    return dfs(goal, maze, lifo_queue, visited, path) # Recursively call the next DFS step.

# Load mazes.
maze_t = load_maze(path='./mazes/maze_7.json')
maze_s = load_maze(path='./mazes/maze_11.json')
maze_l = load_maze(path='./mazes/maze_31.json')
maze_m = load_maze(path='./mazes/maze_91.json')

# Perform Depth First Search (DFS).
maze = maze_m
start = (1, 1)
goal = (maze.shape[0] - 2, maze.shape[1] - 2)
stack = LifoQueue() # Create stack of nodes to visit.
stack.put(start) # Initialize stack with stating position.
path = dfs(goal=goal, maze=maze, lifo_queue=stack)
draw_maze(
    maze=maze, path=path, save_dir=f'./solutions', 
    save_filename=f'dfs_maze{len(maze)}', save_animation=False
)
# print(f'\nTiny Maze ({len(maze_t)} x {len(maze_t)}):')
# print('PATH =', path)