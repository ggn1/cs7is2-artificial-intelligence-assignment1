# Imports 
import numpy as np
from queue import PriorityQueue
from utility import print_result, track_mem_time
from maze import Maze, load_maze, draw_maze, save_maze

def euc_dist(s1, s2):
    ''' Compute Euclidean distance between 2 given states. '''
    return ((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)**(1/2)

def get_least_f(fringe):
    """
    Returns state in the fringe with least f value.
    """
    f_min = None
    for s, v in fringe.items():
        if f_min is None or v['f'] < f_min[1]['f']:
            f_min = (s, v)
    return f_min[0]

@track_mem_time
def a_star(maze, is_print):
    global out_dir, out_file
    if is_print:
        with open(f'{out_dir}/{out_file}.txt', 'w', encoding='utf-8') as f:
            f.write(f'MAZE:\n{str(maze)}')

    to_visit = {} # Maze positions that are yet to be visited.
    visited = {} # Maze positions that have already been visited.

    # Add start node to positions to visit. Set 
    g = 0 # g(s_prime) = g(s) + 1
    h = min([euc_dist(maze.start, goal) for goal in maze.goals])
    f = g + h
    s_prime_fhg = {'f': f, 'h': h, 'g': g}
    
    to_visit[maze.start] = {'f': 0, 'h': 0, 'g': 0}

    goal_found = None
    parents = {} # Capture node parent-child links to backtrack best path from goal later.
    num_dead_ends = 0 # Keep track of no. of dead ends encountered.
    while len(to_visit) > 0: # While there are nodes yet to be visited ...
        s = get_least_f(fringe=to_visit) # Find state with least F value
        s_fhg = to_visit.pop(s)       # and remove it from the visited list.
        visited[s] = s_fhg # Add this state to list of visited ones.
        if s in maze.goals: # If this is a terminal state then end search.
            goal_found = s
            break 

        # For each position in the maze that can be visited from this state ...
        s_prime_list = [
            maze.states[s][a] for a in maze.actions 
            if not maze.states[maze.states[s][a]] is None
        ]
        is_dead_end = True # Assume this is a dead end.
        for s_prime in s_prime_list:
            if s_prime in visited: # If s_prime was already visited, skip it this time.
                continue
            # Else, compute g, h and f values of s_prime.
            g = s_fhg['g'] + 1 # g(s_prime) = g(s) + 1
            h = min([ # h(s_prime) = Manhattan distance from s_prime to nearest goal.
                euc_dist(s_prime, goal) 
                for goal in maze.goals
            ])
            f = g + h # f(s_prime) = g(s_prime) + h(s_prime)
            s_prime_fhg = {'f': f, 'h': h, 'g': g}
            # If the new state is yet to be visited, update
            # its f, g, h values if current g value is smaller 
            # than last recorded value.
            if s_prime in to_visit and s_prime_fhg['f'] >= to_visit[s_prime]['f']:
                continue
            to_visit[s_prime] = s_prime_fhg
            parents[s_prime] = s # Keep track of this node being the parent of this neighbor.
            is_dead_end = False
        if is_dead_end:
            num_dead_ends += 1
    
    # Reconstruct continuous solution path from start to goal.
    solution = [] # Keep track of the solution path.
    state_cur = goal_found # current node 
    while state_cur != maze.start: # Until we get to the start node ...
        solution.append(state_cur) # Add current node to the solution path.
        if not state_cur in parents:
            print("No solution found.")
            with open(f'{out_dir}/{out_file}.txt', 'a', encoding='utf-8') as f:
                f.write("\nNo solution found.")
                break
        state_cur = parents[state_cur] # Update current node to be its parent.
    solution.append(maze.start) # Add the starting node to the path to complete it.
    solution.reverse()  # Reverse the path to get it from start to goal
    
    return {
        'solution': solution, # Return path.
        'exploration': list(visited.keys()), # All the nodes that were visited.
        'num_nodes_traversed': len(set(visited)), # Return no. of nodes traversed.
        'num_dead_ends': num_dead_ends # Return no. of dead ends encountered.
    }

def __solve_mazes(sizes, id_nums):
    global out_file, out_dir
    for maze_size in sizes:
        for i in id_nums:
            print(f'Solving {maze_size} x {maze_size} maze {i} ...')
            out_file = f'{i}_astar'
            out_dir = f'__mazes/size{maze_size}'
            maze = Maze(matrix=load_maze(path=f"{out_dir}/{i}.json"))
            res = a_star(maze, is_print=True)
            print_result(result=res, dir=out_dir, filename=out_file)
            draw_maze(
                maze=maze.matrix, 
                solution=res['solution'], 
                exploration=res['exploration'],
                save={'dir':out_dir, 'filename':out_file, 'animation':True},
            )
            print('Done!\n')

out_file = ''
out_dir = ''
if __name__ == '__main__':
    # Solve 1 tiny maze with a single goal.
    __solve_mazes(sizes=[7], id_nums=[1])

    # Solve 10 small mazes with 2 goals.
    __solve_mazes(sizes=[17], id_nums=list(range(1, 11)))

    # Solve 2 medium, large and extra large mazes with 1 goal
    # such that for each size, there is one maze with a goal
    # close to the start point and another with it far away 
    # from the start point.
    __solve_mazes(sizes=[21, 61, 101], id_nums=list(range(1, 3)))