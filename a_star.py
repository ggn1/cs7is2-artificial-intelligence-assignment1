# Imports 
import numpy as np
from queue import PriorityQueue
from track_time import track_time
from maze import Maze, load_maze, draw_maze, save_maze

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
        if s == maze.goal: break # If goal is found, stop.
        # What states do we end up in when upon taking 
        # each possible action from this state = s_prime.
        # Only consider states that are valid (not walls) and
        # have not been visited yet.
        s_prime_list = [
            maze.states[s][a] 
            for a in maze.actions 
            if (
                not maze.states[maze.states[s][a]] is None 
                and not maze.states[s][a] in visited
            )
        ]
        if len(s_prime_list) == 0: 
            num_dead_ends += 1
        for sp in s_prime_list: # Add each neighbor to priority queue as per F value.
            g = s3tuple[2] + 1 # G = No. of hops from neighbor to start.
            h = man_dist(sp, maze.goal) # H = Manhattan distance from neighbor to goal.
            f = g + h # F = G + H
            to_visit.put((f, sp, g)) # Add this neighbor to list of nodes to visit.
            parents[sp] = s # Keep track of this node being the parent of this neighbor.
    
    # Reconstruct continuous solution path from start to goal.
    solution = [] # Keep track of the solution path.
    state_cur = maze.goal # current node 
    while state_cur != maze.start: # Until we get to the start node ...
        solution.append(state_cur) # Add current node to the solution path.
        if not state_cur in parents:
            print('No solution found.')
            break
        state_cur = parents[state_cur] # Update current node to be its parent.
    solution.append(maze.start) # Add the starting node to the path to complete it.
    solution.reverse()  # Reverse the path to get it from start to goal
    
    return {
        'solution': solution, # Return path.
        'exploration': visited, # All the nodes that were visited.
        'num_nodes_traversed': len(set(visited)), # Return no. of nodes traversed.
        'num_dead_ends': num_dead_ends # Return no. of dead ends encountered.
    }

output = 'output_astar'
if __name__ == '__main__':
    # TEST MAZE
    # maze = Maze(dim=10)
    # save_maze(maze, dir='__mazes', filename=output)
    maze = Maze(matrix=load_maze('./__mazes/maze_latest_dim101.json'))
    output = f'output_astar_dim{maze.matrix.shape[0]}'
    res = a_star(maze)
    with open(f'__output/{output}.txt', 'w', encoding='utf-8') as f:
        f.write(f'MAZE:\n{str(maze)}')
        f.write(f'\n\nMETRICS:')
        f.write(f'\nNo. of positions traversed = {res["num_nodes_traversed"]}')
        f.write(f'\nNo. of dead ends = {res["num_dead_ends"]}')
    draw_maze(
        maze=maze.matrix, 
        solution=res['solution'], 
        exploration=res['exploration'],
        save={'dir':'__output', 'filename':output, 'animation':False}
    )