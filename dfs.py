from queue import LifoQueue
from track_time import track_time
from maze import Maze, load_maze, draw_maze, save_maze

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
        if s == maze.goal: break # If this is terminal state then end search.
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

output = 'output_dfs'
if __name__ == '__main__':
    # TEST MAZE
    # maze = Maze(dim=10)
    # save_maze(maze, dir='__mazes', filename=output)
    maze = Maze(matrix=load_maze('./__mazes/maze_latest_dim101.json'))
    output = f'output_dfs_dim{maze.matrix.shape[0]}'
    res = dfs(maze)
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