import numpy as np
from maze import draw_maze

def handle_result(res, maze, save_dir, save_filename, save_animation=False):
    ''' Displays path animation and prints 
        given run result for a maze of given shape. 
    '''
    draw_maze(
        maze=maze, path=res['path'], exploration=res['exploration'],
        save_dir=save_dir, save_filename=save_filename, save_animation=save_animation
    )
    print(f"\nMAZE ({maze.shape[0]} x {maze.shape[1]}):")
    print(f"Execution time = {res['seconds']} seconds")
    print(f"No. of dead ends = {res['num_dead_ends']}")
    print(f"No. of nodes traversed = {res['num_nodes_traversed']}/{maze.shape[0]*maze.shape[1]}")

def print_mat_2d(mat):
    for row in mat:
        print(row.tolist())

def v_to_mat(v, shape):
    mat = np.zeros(shape) - 1
    for key, val in v.items():
        mat[key] = val
    return mat