# Imports 
import numpy as np
from queue import PriorityQueue
from track_time import track_time
from utility import handle_result
from maze import draw_maze, load_maze, Maze

def man_dist(node1, node2):
    ''' Compute Manhattan distance between 2 given nodes. '''
    return np.abs(node1[0] - node2[0]) + np.abs(node1[1] - node2[1])

@track_time
def a_star(maze):
    to_visit = PriorityQueue()
    visited = []
    to_visit.put(
        (0 + man_dist(maze.start, maze.goal), # Priority = f = g + h.
        maze.start, # position
        0 # g
    ))

    # Stopping condition 1 = no more nodes to be visited = queue is empty.
    parents = {} # Capture node parent-child links to backtrack best path from goal later.
    num_dead_ends = 0 # Keep track of no. of dead ends encountered.
    while not to_visit.empty(): # While there are nodes yet to be visited ...
        node = to_visit.get() # Pop node from the priority queue with least F value.
        node_pos = node[1] # Get node position.
        visited.append(node_pos) # Add this node to list of visited ones.
        if node_pos == maze.goal: break # If goal is found, stop.
        neighbors = maze.get_walkable_neighbors(node_pos, ignore=visited) # Get valid neighbors.
        if len(neighbors) == 0: num_dead_ends += 1 # If no valid neighbors, then this is a dead end.
        for neighbor in neighbors: # Add each neighbor to priority queue as per F value.
            g = node[2] + 1 # G = No. of hops from neighbor to start.
            h = man_dist(neighbor, maze.goal) # H = Manhattan distance from neighbor to goal.
            f = g + h # F = G + H
            to_visit.put((f, neighbor, g)) # Add this neighbor to list of nodes to visit.
            parents[neighbor] = node_pos # Keep track of this node being the parent of this neighbor.
    
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

    # MEDIUM MAZE
    maze = Maze(maze=load_maze(path='./mazes/m_dim41.json'))
    res = a_star(maze)
    handle_result(res, maze.maze, './solutions', f'a_star_dim{maze.maze.shape[0]}')

    # LARGE MAZE
    maze = Maze(maze=load_maze(path='./mazes/l_dim101.json'))
    res = a_star(maze)
    handle_result(res, maze.maze, './solutions', f'a_star_dim{maze.maze.shape[0]}')