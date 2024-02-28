from queue import PriorityQueue
from maze import Maze, load_maze, draw_maze
from utility import track_mem_time, solve_maze

def man_dist(s1, s2):
    """ Compute Manhattan distance between 2 given states. """
    return abs(s1[0] - s2[0])+ abs(s1[1] - s2[1])

@track_mem_time
def a_star(maze):
    """ A* search. """
    fringe = PriorityQueue() # Keep track of nodes with lowest f(n) = g(n) + h(n).
    parents = {} # Keep track of parents of each node for path reconstruction from goal later.
    g = {} # For each position n in the maze, keep track of distance from start = g(n).
    goal = None # Goal found.

    # Initialize data structures.
    fringe.put((0, maze.start)) # Initially, fringe has only start node n with f(n) = 0.
    parents[maze.start] = None # Start position or node is root and hence has no parent.
    g[maze.start] = 0 # Cost of start node from start is 0.

    while(not fringe.empty()): # Until the fringe is empty ...
        # Get current position / node / state in the maze.
        _, s = fringe.get() # returns (f(s), s)
        
        # If s is a goal, then stop!
        if (s in maze.goals):
            goal = s
            break

        # Else, get all valid next states from s.
        # Valid next states are those states that are
        # reachable from s through any of the 4 actions
        # and are not walls or lie outside the maze.
        s_prime_list = [
            maze.states[s][a] for a in maze.actions 
            if not maze.states[maze.states[s][a]] is None
        ]

        # For every next state ...
        for s_prime in s_prime_list:
            s_prime_g_new = g[s] + 1 # (cost so far = g(s)) + (cost to go from s to s_prime = 1)
            if (
                not s_prime in g # If g(s') is unknown
                or s_prime_g_new < g[s_prime] # or if new g(s') < known g(s')
            ):  
                # Update g(s') to new value.
                g[s_prime] = s_prime_g_new 
                
                # Compute h(s') = min(Manhattan Distance between s' and each goal in the maze.)
                s_prime_h = min([man_dist(s_prime, goal) for goal in maze.goals])

                # Compute f(s') = g(s') + h(s')
                s_prime_f = s_prime_g_new + s_prime_h

                # Add new f(s') to the fringe.
                fringe.put((s_prime_f, s_prime))

                # s is parent of s'
                parents[s_prime] = s

    return {
        'parents': parents,
        'goal': goal,
        'num_nodes_traversed': len(parents)
    }

def __conduct_experiments(sizes, id_nums):
    """ Solves all mazes as required for assignment 1 experiments. """
    for maze_size in sizes:
        for i in id_nums:
            print(f'Solving {maze_size} x {maze_size} maze {i} ...')
            out_file = f'{i}_astar'
            out_dir = f'__mazes/size{maze_size}'
            maze = Maze(matrix=load_maze(path=f"{out_dir}/{i}.json"))
            res = solve_maze(
                solver_type='a-star', solver=a_star, maze=maze, 
                out_file=out_file, out_dir=out_dir
            )
            draw_maze(
                maze=maze.matrix, 
                solution=res['solution'], 
                exploration=list(res['parents'].keys()),
                save={'dir':out_dir, 'filename':out_file, 'animation':True},
            )
            print('Done!\n')

if __name__ == '__main__':
    """ Triggers solving of all mazes as required for assignment 1 experiments. """
    # Solve 1 maze each of varying sizes with 1 goal.
    __conduct_experiments(sizes=[7, 15], id_nums=[1])

    # Solve 3 mazes each of varying sizes with 1 goal.
    __conduct_experiments(sizes=[21, 61, 101], id_nums=list(range(1, 4)))
    
    # Solve 5 31x31 mazes with 2 goals.
    __conduct_experiments(sizes=[31], id_nums=list(range(1, 6)))