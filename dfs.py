from queue import LifoQueue
from track_time import track_time
from utility import handle_result
from maze import Maze, load_maze

@track_time
def dfs(maze):
    to_visit = LifoQueue() # Create stack of nodes to visit.
    to_visit.put(maze.start) # Initialize stack with stating position.
    visited = []
    parents = {}
    num_dead_ends = 0
    while not to_visit.empty():
        s = to_visit.get()
        visited.append(s) # This is the state we visit in this step of DFS.
        if maze.isEnd(s): break # If this is terminal state then end search.
        # What states do we end up in when upon taking 
        # each possible action from this state = s_prime.
        s_prime = [maze.succ(s, a) for a in maze.actions] 
        # Only states that have not yet been visited are walkable.
        s_prime_walkable = [sp for sp in s_prime if maze.is_walkable(sp, visited)]
        if len(s_prime_walkable) == 0: num_dead_ends += 1
        for sp in s_prime_walkable:
            to_visit.put(sp)
            parents[sp] = s
    # Reconstruct continuous solution path from start to goal.
    path = [] # Keep track of the solution path.
    state_cur = maze.goal # current node 
    while state_cur != maze.start: # Until we get to the start node ...
        path.append(state_cur) # Add current node to the solution path.
        if not state_cur in parents:
            print('No solution found.')
            break
        state_cur = parents[state_cur] # Update current node to be its parent.
    path.append(maze.start) # Add the starting node to the path to complete it.
    path.reverse()  # Reverse the path to get it from start to goal
    return {
        'path': path, # Return path.
        'exploration': visited, # All the nodes that were visited.
        'num_nodes_traversed': len(set(visited)), # Return no. of nodes traversed.
        'num_dead_ends': num_dead_ends # Return no. of dead ends encountered.
    }

if __name__ == '__main__':
    # Perform Depth First Search (DFS).

    # SMALL MAZE
    maze = Maze(maze=load_maze(path='./mazes/s_dim21.json'))
    res = dfs(maze)
    handle_result(res, maze.maze, './solutions', f'dfs_dim{maze.maze.shape[0]}')

    # # MEDIUM MAZE
    # maze = Maze(maze=load_maze(path='./mazes/m_dim41.json'))
    # res = dfs(maze)
    # handle_result(res, maze.maze, './solutions', f'dfs_dim{maze.maze.shape[0]}')

    # # LARGE MAZE
    # maze = Maze(maze=load_maze(path='./mazes/l_dim101.json'))
    # res = dfs(maze)
    # handle_result(res, maze.maze, './solutions', f'dfs_dim{maze.maze.shape[0]}')