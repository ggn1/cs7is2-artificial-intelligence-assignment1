import argparse
from queue import Queue
from maze import Maze, load_maze, draw_maze
from utility import track_mem_time, solve_maze

@track_mem_time
def bfs(maze):
    """ Breadth First Search. """
    fringe = Queue() # Create FIFO queue of nodes to visit.
    parents = {} # Keep track of node relationships.
    goal = None # Goal found.
    
    # Initialize data structures.
    fringe.put(maze.start) # Initially, fringe has only start position or node.
    parents[maze.start] = None # Start position or node is root and hence has no parent.

    while not fringe.empty(): # Until the fringe is empty ...
        s = fringe.get() # Get position / node / state in the maze from FIFO fringe.
        if s in maze.goals: # If this is a terminal state then end search.
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
            if not s_prime in parents: # If this node has not yet been visited ...                
                fringe.put(s_prime) # Add it to the list of visited nodes.
                parents[s_prime] = s # s is parent of s'

    return {
        'parents': parents,
        'goal': goal,
        'num_nodes_traversed': len(parents)
    }

def __conduct_experiment(sizes, id_nums, load_dir, save_dir, save_anim=True):
    """ Solves all mazes as required for assignment 1 experiments. """
    for maze_size in sizes:
        for i in id_nums:
            print(f'Solving {maze_size} x {maze_size} maze {i} ...')
            out_file = f'{i}_bfs'
            out_dir = f'{save_dir}/size{maze_size}'
            maze = Maze(
                matrix=load_maze(path=f"{load_dir}/size{maze_size}/{i}.json"),
                max_gamma=0.99, min_epsilon=1e-6
            )
            res = solve_maze(
                solver_type='bfs', solver=bfs, maze=maze, 
                out_file=out_file, out_dir=out_dir
            )
            draw_maze(
                maze=maze.matrix, 
                solution=res['solution'], 
                exploration=list(res['parents'].keys()),
                save={'dir':out_dir, 'filename':out_file, 'animation':save_anim},
            )
            print('Done!\n')

if __name__ == '__main__':
    """ Triggers solving of all mazes as required for assignment 1 experiments. """
    parser = argparse.ArgumentParser(prog='Breadth First Search')
    parser.add_argument('-l', '--load-dir', type=str, help="Directory containing mazes of sizes defined in this file.")
    parser.add_argument('-s', '--save-dir', type=str, help="Directory in which to store mazes.")
    parser.add_argument('-a', '--save-anim', action='store_true', help='Save solution animation.')
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = args.save_dir
    save_anim = args.save_anim

    # Solve 1 maze each of varying sizes with 1 goal.
    __conduct_experiment(sizes=[7, 15], id_nums=[1], load_dir=load_dir, save_dir=save_dir, save_anim=save_anim)

    # Solve 3 mazes each of varying sizes with 1 goal.
    __conduct_experiment(sizes=[21, 61, 101], id_nums=list(range(1, 4)), load_dir=load_dir, save_dir=save_dir, save_anim=save_anim)
    
    # Solve 5 31x31 mazes with 2 goals.
    __conduct_experiment(sizes=[31], id_nums=list(range(1, 6)), load_dir=load_dir, save_dir=save_dir, save_anim=save_anim)

    # # For video demo.
    # # Solve 1 41 x 41 maze with 1 goal.
    # __conduct_experiment(sizes=[41], id_nums=[1], load_dir='__demo', save_dir='__demo', save_anim=save_anim)