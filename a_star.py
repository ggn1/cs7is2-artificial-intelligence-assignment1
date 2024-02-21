# Imports 
import numpy as np
from queue import PriorityQueue
from track_time import track_time
from utility import handle_result
from maze import draw_maze, load_maze, Maze

def man_dist(s1, s2):
    ''' Compute Manhattan distance between 2 given states. '''
    return np.abs(s1[0] - s2[0]) + np.abs(s1[1] - s2[1])

@track_time
def a_star(maze):
    to_visit = PriorityQueue()
    visited = []
    to_visit.put((
        0 + man_dist(maze.start, maze.goal), # Priority = f = g + h.
        maze.start, # position
        0 # g
    ))

    # Stopping condition 1 = no more nodes to be visited = queue is empty.
    parents = {} # Capture node parent-child links to backtrack best path from goal later.
    num_dead_ends = 0 # Keep track of no. of dead ends encountered.
    while not to_visit.empty(): # While there are nodes yet to be visited ...
        s3tuple = to_visit.get() # Pop state from the priority queue with least F value.
        s = s3tuple[1] # Get node position.
        visited.append(s) # Add this node to list of visited ones.
        if maze.isEnd(s): break # If goal is found, stop.
        # What states do we end up in when upon taking 
        # each possible action from this state = s_prime.
        s_prime = [maze.succ(s, a) for a in maze.actions]
        # Only states that have not yet been visited are walkable.
        s_prime_walkable = [sp for sp in s_prime if maze.is_walkable(sp, visited)]
        for sp in s_prime_walkable: # Add each neighbor to priority queue as per F value.
            g = s3tuple[2] + 1 # G = No. of hops from neighbor to start.
            h = man_dist(sp, maze.goal) # H = Manhattan distance from neighbor to goal.
            f = g + h # F = G + H
            to_visit.put((f, sp, g)) # Add this neighbor to list of nodes to visit.
            parents[sp] = s # Keep track of this node being the parent of this neighbor.
    
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
    # Perform A* Search.

    # SMALL MAZE
    maze = Maze(maze=load_maze(path='./mazes/s_dim21.json'))
    res = a_star(maze)
    handle_result(res, maze.maze, './solutions', f'a_star_dim{maze.maze.shape[0]}')

    # # MEDIUM MAZE
    # maze = Maze(maze=load_maze(path='./mazes/m_dim41.json'))
    # res = a_star(maze)
    # handle_result(res, maze.maze, './solutions', f'a_star_dim{maze.maze.shape[0]}')

    # # LARGE MAZE
    # maze = Maze(maze=load_maze(path='./mazes/l_dim101.json'))
    # res = a_star(maze)
    # handle_result(res, maze.maze, './solutions', f'a_star_dim{maze.maze.shape[0]}')