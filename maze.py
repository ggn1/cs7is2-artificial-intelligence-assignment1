# Code for maze generation and display from: https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96.

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def create_maze(dim):
    # Create a grid filled with walls
    maze = np.ones((dim*2+1, dim*2+1))

    # Define the starting point
    x, y = (0, 0)
    maze[2*x+1, 2*y+1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2*nx+1, 2*ny+1] == 1:
                maze[2*nx+1, 2*ny+1] = 0
                maze[2*x+1+dx, 2*y+1+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
            
    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0

    return maze

#... animate the path through the maze ...
def draw_maze(maze, save_dir='', save_filename='', save_animation=False, path=None):
    fig, ax = plt.subplots(figsize=(10,10))
    
    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    ax.set_xticks([i for i in range(maze.shape[0])] if maze.shape[0] <= 21 else [])
    ax.set_yticks([i for i in range(maze.shape[1])] if maze.shape[0] <= 21 else [])
    
    # Prepare for path animation
    if path is not None:
        line, = ax.plot([], [], color='red', linewidth=2)
        
        def init():
            line.set_data([], [])
            return line,
        
        # update is called for each path point in the maze
        def update(frame):
            x, y = path[frame]
            line.set_data(*zip(*[(p[1], p[0]) for p in path[:frame+1]]))  # update the data
            return line,
        
        ani = animation.FuncAnimation(
            fig, update, frames=range(len(path)), init_func=init, 
            blit=True, repeat = False, interval=20
        )
    
    # Draw entry and exit arrows
    ax.arrow(0, 1, .4, 0, fc='green', ec='green', head_width=0.3, head_length=0.3)
    ax.arrow(maze.shape[1]-1, maze.shape[0]-2, 0.4, 0, fc='blue', ec='blue', head_width=0.3, head_length=0.3)

    plt.show()

    # Lines added by me to optionally save the animation and save the 
    # image of the maze with the path drawn through it. OWN CODE
    if save_dir != '':
        fig.savefig(f'{save_dir}/{save_filename}.png')
        if save_animation: ani.save(f'{save_dir}/{save_filename}.gif', writer="pillow")

# OWN CODE
def save_maze(maze, dir, filename):
    ''' Saves given maze in a json file with the 
        given name in the given directory. 
    '''
    if (
        len(filename) <= 5 or 
        filename[-5:] != '.json'
    ): filename += '.json'
    with open(f'{dir}/{filename}', 'w') as f:
        json.dump(maze.tolist(), f)

# OWN CODE
def load_maze(path):
    ''' Loads and returns a given maze from a json  
        file at the given path. 
    '''
    with open(path, 'r') as f:
       maze = np.array(json.load(f))
    return maze