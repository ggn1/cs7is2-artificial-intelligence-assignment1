# Code for maze generation and display from: https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96.

import json
import random
import numpy as np
from maze_state import MazeState
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# wall = 0
# space = 1
# start = 3
# goal = 2

def draw_maze(maze, save=None, solution=None, exploration=None, state_values=None):
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
    
    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    # Plot values (value iteration) if available.
    if state_values is not None:
        axes[0].matshow(state_values, cmap=plt.cm.Blues)

    axes[1].imshow(maze, interpolation='nearest')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    scatter_path = None 
    scatter_exp = None
    
    # Animate solution path.
    if solution is not None: # Id there is a solution, provided, draw it on the maze.
        def init():
            nonlocal scatter_path
            nonlocal scatter_exp
            scatter_path = axes[1].scatter([], [], marker='o', color='w', s=2)
            if exploration is not None: # Animate explored path as well.
                scatter_exp = axes[1].scatter([], [], marker='o', color='r', s=2)
                return scatter_path, scatter_exp
            else:
                return scatter_path,
        # Update is called for each path point in the maze.
        def update(frame):
            nonlocal scatter_path
            path_xy = [p for p in solution[:frame+1]]
            path_x = [p[1] for p in path_xy]
            path_y = [p[0] for p in path_xy]
            scatter_path.set_offsets(np.column_stack([path_x, path_y]))
            if exploration is not None: 
                nonlocal scatter_exp
                exp_i = exploration.index(solution[frame])
                exp_xy = [e for e in exploration[:exp_i] if not e in path_xy]
                exp_x = [e[1] for e in exp_xy]
                exp_y = [e[0] for e in exp_xy]
                scatter_exp.set_offsets(np.column_stack([exp_x, exp_y]))
                return scatter_path, scatter_exp
            return scatter_path,
        ani = animation.FuncAnimation(
            fig, update, frames = range(len(solution)), init_func = init, 
            repeat = False, interval=1, cache_frame_data=False, blit=True
        )

    # Show the figure or animation.
    plt.show()

    if not save is None: # Optionally save the figure.
        if (
            not type(save) == dict or
            not "dir" in save or
            not "filename" in save or
            not "animation" in save or
            not type(save["dir"]) == type("filename")  == str or
            not type(save["animation"]) == bool
        ): raise Exception( # Validate form of save parameter.
            'Bar argument. Parameter "save" must be of the form '
            + '{"dir": str, "filename":str, "animate":bool}.'
        )
        fig.savefig(f'{save['dir']}/{save['filename']}.png')
        if save['animation']: # Optionally save animation.
            ani.save(f"{save['dir']}/{save['filename']}.gif", writer="pillow")

def save_maze(maze, dir, filename):
    ''' Saves given maze in a json file with the 
        given name in the given directory. 
    '''
    with open(f'{dir}/{filename}_dim{maze.matrix.shape[0]}.json', 'w') as f:
        json.dump(maze.matrix.tolist(), f)

def load_maze(path):
    ''' Loads and returns a given maze from a json  
        file at the given path. 
    '''
    with open(path, 'r') as f:
       matrix = np.array(json.load(f))
    return matrix

class Maze():
    def __init__(self, start=(1, 1), matrix=None, dim=20):
        self.start = start
        if type(matrix) != type(None): # If the maze is given, and thus is not None ...
            self.matrix = matrix
            self.goal = self.__find_goal() # Find the goal.
            self.__dim = (self.matrix.shape[0]-1)//2
        else:
            self.__dim = dim # Input dimension = 20 By default.
            self.matrix = np.zeros((dim*2+1, dim*2+1)) # Create a grid filled with walls only.
            self.__create_maze() # Sets goal.
        self.actions = ["↑", "→", "↓", "←"]
        self.states, self.state_positions = self.__get_states()

    def __find_goal(self):
        ''' Return position of the goal in this maze. '''
        for i in range(0, self.matrix.shape[0]):
            for j in range(0, self.matrix.shape[1]):
                if self.matrix[i, j] == 2: 
                    return (i, j)

    def __create_maze(self):
        # wall = 0
        # space = 1
        # start = 3
        # goal = 2   
           
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
                    and nx < self.__dim and ny < self.__dim 
                    and self.matrix[2*nx+1, 2*ny+1] == 0
                ):
                    self.matrix[2*nx+1, 2*ny+1] = 1
                    self.matrix[2*x+1+dx, 2*y+1+dy] = 1
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()
                
        # Create an entrance and an exit
        goal = (random.randint(2, 2*self.__dim), random.randint(2, 2*self.__dim))
        while(self.matrix[goal] == 0): # If goal is wall, look again.
            goal = (random.randint(2, 2*self.__dim), random.randint(2, 2*self.__dim))
        self.matrix[self.start] = 3 # start = 3
        self.matrix[goal] = 2 # goal = 2
        self.goal = goal
          
    def __get_states(self):
        """
        Returns a matrix of MazeState objects for 
        each free space in the maze. Only empty spots
        (not walls) are considered part of state space.
        # wall = 0
        # space = 1
        # start = 3
        # goal = 2
        """
        states = np.array([[None] * self.matrix.shape[0] for _ in range(self.matrix.shape[0])])
        state_positions = [] # Only positions without a wall is considered to be a valid state.
        # For each position in the maze ...
        for i in range(1, len(self.matrix)-1):
            for j in range(1, len(self.matrix)-1):
                # If (i, j) is a wall, use 'None' to represent walls in the state.
                if self.matrix[i][j] == 0:
                    states[i][j] = None
                    continue
                state_positions.append((i, j)) # Since not wall, add to valid state positions.
                    
                # Else for each direction that the agent can
                # move in from (i, j), define the new position
                # it will end up in. If the agent moves in the
                # direction of a wall, it's position does not change.

                # Up
                up = (i-1, j)
                # if self.matrix[(i-1, j)] == 0: # wall
                #     up = (i, j)
                # else: # not wall
                #     up = (i-1, j)
                    
                # Down
                down = (i+1, j)
                # if self.matrix[(i+1, j)] == 0: # wall
                #     down = (i, j)
                # else: # not wall
                #     down = (i+1, j)
                    
                # Right
                right = (i, j+1)
                # if self.matrix[(i, j+1)] == 0: # wall
                #     right = (i, j)
                # else: # not wall
                #     right = (i, j+1)
                    
                # Left
                left = (i, j-1)
                # if self.matrix[(i, j-1)] == 0: # wall
                #     left = (i, j)
                # else: # not wall
                #     left = (i, j-1)

                # Update state to capture information about immediate surroundings.
                states[(i, j)] = MazeState(up=up, right=right, down=down, left=left)

        return states, state_positions

    def __str__(self):
        mat = np.full(self.matrix.shape, "#")
        for p in self.state_positions:
            mat[p] = " "
        mat[self.start] = "S"
        mat[self.goal] = "G"
        return str(mat)

    def T(self, s, a, s_prime):
        """ 
        Transition function which when given a state s and action a,
        returns the probability of ending up in given state s_prime.
        This scenario is deterministic. Thus, if s_prime is truly the 
        result of taking action a at state s, then probability of new 
        state being s_prime is 1. Otherwise, it is 0.
        @param s: Current state.
        @param a: Action taken.
        @param s_prime: New state.
        """
        if self.states[s] is None:
            raise Exception('Given state s is a wall.')
        # Determine what s_prime should be
        # if action a is executed at state s.
        s_prime_true = self.states[s][a]
        if self.states[s_prime_true] is None:
            s_prime_true = s
        # Return 1.0 if expected s_prime and given one is
        # same and 0.0 otherwise.
        return int(s_prime == s_prime_true)

    def R(self, s, a, s_prime):
        """ 
        Rewards function which when given a state s and action a,
        returns the reward of taking action a in state s to end
        up in state s_prime.
        @param s: Current state.
        @param a: Action taken.
        @param s_prime: Next state.
        """
        reward = 0
        max_reward = self.matrix.shape[0] * self.matrix.shape[1]
        if s == self.goal: # s is goal => big positive reward.
            reward += max_reward
        else:
            # count no. of walls and exits around s.
            num_walls = np.sum(int(self.matrix[self.states[s][a]] == 0) for a in self.actions)
            # num_exits = 4 - num_walls
            if num_walls == 3: # dead end => negative reward.
                reward += -1 * 0.8 * max_reward
            if s_prime == s: # action takes agent into the wall => negative reward.
                reward += -1 * 0.1 * max_reward
            else: # action takes agent out of the dead end => positive reward.
                reward += 0.1 * max_reward
        return reward
        # if s == self.goal:
        #     return max_reward
        # return 0.1*max_reward