from queue import Queue
from track_time import track_time
from maze import draw_maze, load_maze

@track_time
def bfs(maze):
    start = (1, 0)
    goal = (maze.shape[0] - 2, maze.shape[1] - 1)
    fifo_queue = Queue() # Create stack of nodes to visit.
    fifo_queue.put(start) # Initialize stack with stating position.
    visited = set()
    parents = {}
    num_dead_ends = 0
    while not fifo_queue.empty():
        node = fifo_queue.get()
        visited.add(node) # This is the node we visit in this step of DFS.
        if node == goal: break # If this is the goal node then end search.
        is_dead_end = True # Assume this node is a dead end.
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
                node_to_visit[0] >= 0 and
                node_to_visit[0] < maze.shape[0] and
                node_to_visit[1] >= 0 and
                node_to_visit[1] < maze.shape[1] and
                maze[node_to_visit] == 0
            ): 
                fifo_queue.put(node_to_visit)
                parents[node_to_visit] = node
                is_dead_end = False
        if is_dead_end: num_dead_ends += 1
    # Reconstruct continuous solution path from start to goal.
    path = [] # Keep track of the solution path.
    node_cur = goal # current node
    while node_cur != start: # Until we get to the start node ...
        path.append(node_cur) # Add current node to the solution path.
        node_cur = parents[node_cur] # Update current node to be its parent.
    path.append(start) # Add the starting node to the path to complete it.
    path.reverse()  # Reverse the path to get it from start to goal
    return {
        'path': path, # Return path.
        'num_nodes_traversed': len(visited), # Return no. of nodes traversed.
        'num_dead_ends': num_dead_ends # Return no. of dead ends encountered.
    } 

def handle_result(res, maze_shape):
    ''' Displays path animation and prints 
        given run result for a maze of given shape. 
    '''
    draw_maze(
        maze=maze, path=res['path'], save_dir=f'./solutions', 
        save_filename=f'bfs_maze{maze_shape[0]}', save_animation=False
    )
    print(f"\nMAZE ({maze_shape[0]} x {maze_shape[1]}):")
    print(f"Execution time = {res['seconds']} seconds")
    print(f"No. of dead ends = {res['num_dead_ends']}")
    print(f"No. of nodes traversed = {res['num_nodes_traversed']}/{maze_shape[0]*maze_shape[1]}")

# Perform Depth First Search (DFS).

# TINY MAZE (5 x 5)
maze_t = load_maze(path='./mazes/maze_5.json')
maze = maze_t
res = bfs(maze=maze)
handle_result(res, maze.shape)

# EXTRA SMALL MAZE (7 x 7)
maze_xs = load_maze(path='./mazes/maze_7.json')
maze = maze_xs
res = bfs(maze=maze)
handle_result(res, maze.shape)

# SMALL MAZE (11 x 11)
maze_s = load_maze(path='./mazes/maze_11.json')
maze = maze_s
res = bfs(maze=maze)
handle_result(res, maze.shape)

# MEDIUM MAZE (31 x 31)
maze_m = load_maze(path='./mazes/maze_31.json')
maze = maze_m
res = bfs(maze=maze)
handle_result(res, maze.shape)

# LARGE MAZE (91 x 91)
maze_l = load_maze(path='./mazes/maze_91.json')
maze = maze_l
res = bfs(maze=maze)
handle_result(res, maze.shape)