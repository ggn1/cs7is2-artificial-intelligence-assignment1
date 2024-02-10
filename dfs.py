import random
import numpy as np
from queue import LifoQueue
from track_time import track_time
from maze import create_maze, draw_maze, load_maze

def man_dist(node1, node2):
    ''' Computes and returns Euclidean distance 
        between given node 1 and node 2. 
    '''
    return np.abs(node1[0] - node2[0]) + np.abs(node1[1] - node2[1])


@track_time
def dfs(maze):
    # SET UP
    # ------
    # Define possible directions:
    # -> (0, 1) = UP
    # -> (1, 0) = RIGHT
    # -> (0, -1) = DOWN
    # -> (-1, 0) = LEFT
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    # Maze entrance (start) is always at the top leftmost cell in the maze and
    # the exit (goal) is always at the bottom rightmost cell in the implemented maze.
    start = (1, 1) 
    goal = (maze.shape[0]-2, maze.shape[1]-2)
    lifo_queue = LifoQueue() # LIFO queue = stack to use to keep track of nodes to visit.
    visited = [] # List = array to keep track of nodes that have already been visited.
    path = [] # List that keeps track of the solution path including backtracking.
    is_dead_end = False # To keep track of whether a dead end has been reached or not.
    
    # DFS ALGORITHM
    # -------------
    lifo_queue.put(start) # Initialize LIFO queue with the first node to visit.
    while(not lifo_queue.empty()): # Do until the stack is empty ...
        node = lifo_queue.get() # Pop the latest element from the queue.
        visited.append(node) # Add this node to the list of visited ones.
        path.append(node) # Add this node to the solution path.
        if node == goal: break # Perform goal check and stop searching if at the goal.
        is_dead_end = True # Assume that this is a dead end.
        for node_to_visit in [(node[0]+d[0], node[1]+d[1]) for d in directions]:
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
        if is_dead_end: # If this was indeed a dead end ...
            node_next = lifo_queue.get()
            lifo_queue.put(node_next) # put node back in to simulate a peak operation
            # Travel back through path taken to the node that is
            # closest (Manhattan distance) to the next node to pop.
            node_closest = None
            dist_least = float('inf')
            # For previous path nodes (excluding this node itself) ...
            for i in range(len(path)-2, 0): 
                node_prev = path[i] # Get previous node.
                man_dist = man_dist(node_prev, node_next) # Compute Manhattan distance.
                if man_dist <= dist_least: # If this is least distance so far ...
                    dist_least = man_dist # Update distance value.
                    node_closest = node_prev # Update node position.
                else: 
                    # If this prev_node resulted in a path that is 
                    # farther away from the last known node closest
                    # to the next node to pop, then exit the loop
                    # since this means that we've found the last point
                    # from which the algorithm branched out.
                    break
            path.append(node_closest)
    return path # Returns the path taken.

# Load mazes.
maze_s = load_maze(path='./mazes/maze_11.json')
maze_l = load_maze(path='./mazes/maze_31.json')
maze_m = load_maze(path='./mazes/maze_91.json')

# Perform Depth First Search (DFS).
path_s, time_s = dfs(maze_s)
print(f'Algorithm = DFS | Maze Size = {maze_s.shape} | Time Taken = {time_s} seconds.')
print('Saving maze animation ...')
draw_maze(
    maze=maze_s, save_dir='./solutions', 
    save_filename=f'maze{maze_s.shape[0]}_dfs', save_animation=True, 
    path=path_s
)
print('Done!\n')