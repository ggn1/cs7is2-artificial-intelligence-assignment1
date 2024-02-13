# Code for maze generation and display from: https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96.

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# wall = 0
# space = 1
# start = 3
# goal = 2

#... animate the path through the maze ...
def draw_maze(maze, save_dir='', save_filename='', save_animation=False, path=None, exploration=None):
    fig, ax = plt.subplots(figsize=(10,10))
    
    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    ax.imshow(maze, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    scatter_path = None 
    scatter_exp = None
    
    # Prepare for path animation
    if path is not None and exploration is not None:

        def init():
            nonlocal scatter_path
            nonlocal scatter_exp
            scatter_path = ax.scatter([], [], marker='o', color='w', s=2)
            scatter_exp = ax.scatter([], [], marker='o', color='r', s=2)
            return scatter_path, scatter_exp
        
        # update is called for each path point in the maze
        def update(frame):
            nonlocal scatter_path
            nonlocal scatter_exp
            exp_i = exploration.index(path[frame])
            path_xy = [p for p in path[:frame+1]]
            exp_xy = [e for e in exploration[:exp_i] if not e in path_xy]
            path_x = [p[1] for p in path_xy]
            path_y = [p[0] for p in path_xy]
            exp_x = [e[1] for e in exp_xy]
            exp_y = [e[0] for e in exp_xy]
            scatter_exp.set_offsets(np.column_stack([exp_x, exp_y]))
            scatter_path.set_offsets(np.column_stack([path_x, path_y]))
            return scatter_path, scatter_exp

        ani = animation.FuncAnimation(
            fig, update, frames = range(len(path)), init_func = init, 
            repeat = False, interval=1, cache_frame_data=False, blit=True
        )

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
    with open(f'{dir}/{filename}_dim{maze.maze.shape[0]}.json', 'w') as f:
        json.dump(maze.tolist(), f)

def load_maze(path):
    ''' Loads and returns a given maze from a json  
        file at the given path. 
    '''
    with open(path, 'r') as f:
       maze = np.array(json.load(f))
    return maze

class Maze():
    def __init__(self, maze=None, dim=20):
        self.start = (1, 1)
        self.goal = None
        self.dim = None
        self.maze = maze
        if type(self.maze) != type(None): # If the maze is given, and thus is not None ...
            self.goal = self.__find_goal()
            self.dim = (self.maze.shape[0]-1)//2
        else:
            self.dim = dim # Input dimension = 20 By default.
            self.maze = np.zeros((dim*2+1, dim*2+1)) # Create a grid filled with walls only.
            self.__create_maze()

    def __find_goal(self):
        ''' Return position of the goal in this maze. '''
        for i in range(0, self.maze.shape[0]):
            for j in range(0, self.maze.shape[1]):
                if self.maze[i, j] == 2: 
                    return (i, j)

    def __create_maze(self):
        # wall = 0
        # space = 1
        # Initialize the stack with the starting point
        stack = [self.start]
        while len(stack) > 0:
            x, y = stack[-1]
            # Define possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # up, right, down, left
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (
                    nx >= 0 and ny >= 0 
                    and nx < self.dim and ny < self.dim 
                    and self.maze[2*nx+1, 2*ny+1] == 0
                ):
                    self.maze[2*nx+1, 2*ny+1] = 1
                    self.maze[2*x+1+dx, 2*y+1+dy] = 1
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()
                
        # Create an entrance and an exit
        goal = (random.randint(2, 2*self.dim), random.randint(2, 2*self.dim))
        while(self.maze[goal] == 0):
            goal = (random.randint(2, 2*self.dim), random.randint(2, 2*self.dim))
        self.maze[self.start] = 3 # start = 3
        self.maze[goal] = 2 # goal = 2
        self.goal = goal

    def get_walkable_neighbors(self, node, ignore=[]):
        ''' Returns a list of nodes which can be 
            reached from the given node and are 
            not in the given list of nodes to ignore.
        '''
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] # up, right, down, left
        neighbors = []
        for d in directions:
            neighbor = (node[0] + d[0], node[1] + d[1])
            if (
                # If the adjacent position in this direction has not yet been
                # visited, is within the maze (i.e. does not lie beyond the 
                # maze's dimensions) and is a path (not a wall), then and 
                # only then, add it to the LIFO queue (i.e. list of nodes to visit).
                not neighbor in ignore and
                neighbor[0] >= 0 and
                neighbor[0] < self.maze.shape[0] and
                neighbor[1] >= 0 and
                neighbor[1] < self.maze.shape[1] and
                self.maze[neighbor] == 1 or # neighbor is empty
                self.maze[neighbor] == 2    # neighbor is goal
            ): neighbors.append(neighbor)
        return neighbors