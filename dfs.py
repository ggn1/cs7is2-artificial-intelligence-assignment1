from queue import LifoQueue
from utility import print_result, track_mem_time
from maze import Maze, load_maze, draw_maze, save_maze

@track_mem_time
def dfs(maze, is_print):
    global output, dir_out
    if is_print:
        with open(f'{dir_out}/{output}.txt', 'w', encoding='utf-8') as f:
            f.write(f'MAZE:\n{str(maze)}')
    to_visit = LifoQueue() # Create stack of nodes to visit.
    to_visit.put(maze.start) # Initialize stack with stating position.
    visited = []
    parents = {}
    goal_found = None
    num_dead_ends = 0
    while not to_visit.empty():
        s = to_visit.get()
        visited.append(s) # This is the state we visit in this step of DFS.
        if s in maze.goals: # If this is a terminal state then end search.
            goal_found = s
            break 
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
        for sp in s_prime_list:
            to_visit.put(sp)
            parents[sp] = s
    # Reconstruct continuous solution path from start to goal.
    solution = [] # Keep track of the solution path.
    state_cur = goal_found # current node 
    while state_cur != maze.start: # Until we get to the start node ...
        solution.append(state_cur) # Add current node to the solution path.
        if not state_cur in parents:
            print("No solution found.")
            with open(f'{dir_out}/{output}.txt', 'a', encoding='utf-8') as f:
                f.write("\nNo solution found.")
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

def __solve_mazes(sizes, id_nums):
    global output, dir_out
    for maze_size in sizes:
        for i in id_nums:
            print(f'Solving {maze_size} x {maze_size} maze {i} ...')
            output = f'{i}_dfs'
            dir_out = f'__mazes/size{maze_size}'
            maze = Maze(matrix=load_maze(path=f"{dir_out}/{i}.json"))
            res = dfs(maze, is_print=True)
            print_result(result=res, dir=dir_out, filename=output)
            draw_maze(
                maze=maze.matrix, 
                solution=res['solution'], 
                exploration=res['exploration'],
                save={'dir':dir_out, 'filename':output, 'animation':True},
            )
            print('Done!\n')

output = ''
dir_out = ''
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