from queue import Queue
from maze import Maze, load_maze
from track_time import track_time
from utility import handle_result

@track_time
def bfs(maze):
    to_visit = Queue() # Create FIFO queue of nodes to visit.
    to_visit.put(maze.start) # Initialize stack with stating position.
    visited = []
    parents = {}
    num_dead_ends = 0
    while not to_visit.empty():
        node = to_visit.get()
        visited.append(node) # This is the node we visit in this step of bfs.
        if node == maze.goal: break # If this is the goal node then end search.
        neighbors = maze.get_walkable_neighbors(node, ignore=visited)
        if len(neighbors) == 0: num_dead_ends += 1
        for neighbor in neighbors:
            to_visit.put(neighbor)
            parents[neighbor] = node
    # Reconstruct continuous solution path from start to goal.
    path = [] # Keep track of the solution path.
    node_cur = maze.goal # current node 
    while node_cur != maze.start: # Until we get to the start node ...
        path.append(node_cur) # Add current node to the solution path.
        if not node_cur in parents:
            print('No solution found.')
            break
        node_cur = parents[node_cur] # Update current node to be its parent.
    path.append(maze.start) # Add the starting node to the path to complete it.
    path.reverse()  # Reverse the path to get it from start to goal
    return {
        'path': path, # Return path.
        'exploration': visited,
        'num_nodes_traversed': len(set(visited)), # Return no. of nodes traversed.
        'num_dead_ends': num_dead_ends # Return no. of dead ends encountered.
    }

if __name__ == '__main__':
    # Perform Breadth First Search (BFS).

    # SMALL MAZE
    maze = Maze(maze=load_maze(path='./mazes/s_dim21.json'))
    res = bfs(maze)
    handle_result(res, maze.maze, './solutions', f'bfs_dim{maze.maze.shape[0]}')

    # MEDIUM MAZE
    maze = Maze(maze=load_maze(path='./mazes/m_dim41.json'))
    res = bfs(maze)
    handle_result(res, maze.maze, './solutions', f'bfs_dim{maze.maze.shape[0]}')

    # LARGE MAZE
    maze = Maze(maze=load_maze(path='./mazes/l_dim101.json'))
    res = bfs(maze)
    handle_result(res, maze.maze, './solutions', f'bfs_dim{maze.maze.shape[0]}')