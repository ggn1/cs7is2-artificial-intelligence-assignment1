from queue import Queue
from track_time import track_time
from maze import draw_maze, load_maze

@track_time
def bfs(maze, goal):
    start = (1, 1)
    fifo_queue = Queue() # Create stack of nodes to visit.
    fifo_queue.put(start) # Initialize stack with stating position.
    visited = []
    parents = {}
    num_dead_ends = 0
    while not fifo_queue.empty():
        node = fifo_queue.get()
        visited.append(node) # This is the node we visit in this step of bfs.
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
                maze[node_to_visit] == 1 or 
                maze[node_to_visit] == 2
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
        if not node_cur in parents:
            print('No solution found.')
            break
        node_cur = parents[node_cur] # Update current node to be its parent.
    path.append(start) # Add the starting node to the path to complete it.
    path.reverse()  # Reverse the path to get it from start to goal
    return {
        'path': path, # Return path.
        'exploration': visited,
        'num_nodes_traversed': len(set(visited)), # Return no. of nodes traversed.
        'num_dead_ends': num_dead_ends # Return no. of dead ends encountered.
    } 

def handle_result(res, maze):
    ''' Displays path animation and prints 
        given run result for a maze of given shape. 
    '''
    draw_maze(
        maze=maze, path=res['path'], exploration=res['exploration'],
        save_dir=f'./solutions', save_filename=f'bfs_maze{maze.shape[0]}', 
        save_animation=False
    )
    print(f"\nMAZE ({maze.shape[0]} x {maze.shape[1]}):")
    print(f"Execution time = {res['seconds']} seconds")
    print(f"No. of dead ends = {res['num_dead_ends']}")
    print(f"No. of nodes traversed = {res['num_nodes_traversed']}/{maze.shape[0]*maze.shape[1]}")

# Perform Depth First Search (bfs).

# SMALL MAZE
maze_s, goal_s = load_maze(path='./mazes/s_dim21.json')
maze = maze_s
res = bfs(maze, goal_s)
handle_result(res, maze)

# MEDIUM MAZE
maze_m, goal_m = load_maze(path='./mazes/m_dim41.json')
maze = maze_m
res = bfs(maze, goal_m)
handle_result(res, maze)

# LARGE MAZE
maze_l, goal_l = load_maze(path='./mazes/l_dim101.json')
maze = maze_l
res = bfs(maze, goal_l)
handle_result(res, maze)