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

def draw_maze(maze, save_dir='', save_filename='', save_animation=False, path=None, exploration=None, v=None):
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
    
    # Set the border color to white
    fig.patch.set_edgecolor('white')
    fig.patch.set_linewidth(0)

    # Plot values (value iteration) if available.
    if v is not None:
        axes[0].matshow(v, cmap=plt.cm.Blues)

    axes[1].imshow(maze, interpolation='nearest')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    scatter_path = None 
    scatter_exp = None
    
    # Prepare for path animation
    if path is not None:

        def init():
            nonlocal scatter_path
            nonlocal scatter_exp
            scatter_path = axes[1].scatter([], [], marker='o', color='w', s=2)
            if exploration is not None:
                scatter_exp = axes[1].scatter([], [], marker='o', color='r', s=2)
                return scatter_path, scatter_exp
            return scatter_path,
        
        # update is called for each path point in the maze
        def update(frame):
            nonlocal scatter_path
            path_xy = [p for p in path[:frame+1]]
            path_x = [p[1] for p in path_xy]
            path_y = [p[0] for p in path_xy]
            scatter_path.set_offsets(np.column_stack([path_x, path_y]))
            if exploration is not None: 
                nonlocal scatter_exp
                exp_i = exploration.index(path[frame])
                exp_xy = [e for e in exploration[:exp_i] if not e in path_xy]
                exp_x = [e[1] for e in exp_xy]
                exp_y = [e[0] for e in exp_xy]
                scatter_exp.set_offsets(np.column_stack([exp_x, exp_y]))
                return scatter_path, scatter_exp
            return scatter_path,

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

def save_maze(maze, dir, filename):
    ''' Saves given maze in a json file with the 
        given name in the given directory. 
    '''
    with open(f'{dir}/{filename}_dim{maze.maze.shape[0]}.json', 'w') as f:
        json.dump(maze.maze.tolist(), f)

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
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        if type(self.maze) != type(None): # If the maze is given, and thus is not None ...
            self.goal = self.__find_goal() # Find the goal.
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
        while(self.maze[goal]):
            goal = (random.randint(2, 2*self.dim), random.randint(2, 2*self.dim))
        self.maze[self.start] = 3 # start = 3
        self.maze[goal] = 2 # goal = 2
        self.goal = goal

    def isEnd(self, s):
        """ Returns true if this is the goal state. """
        return self.maze[s] == 2

    def is_walkable(self, s, visited):
        """ 
        Give a state s, returns if this state is reachable 
        in terms of a search algorithms like DFS, BFS and A*.
        @param s: Given state.
        @param visited: List of states that have been visited.
        @return bool: Whether or not this state is walkable.
        """
        # A state is walkable only if it has not yet been
        # visited, is within the maze (i.e. does not lie beyond the 
        # maze's dimensions) and is open space (not a wall).
        return (
            not s in visited and
            s[0] >= 0 and
            s[0] < self.maze.shape[0] and
            s[1] >= 0 and
            s[1] < self.maze.shape[1] and
            self.maze[s] != 0 # s is not a wall
        )

    def succ(self, s, a):
        """ Given a state and an action, returns new state. """
        s_prime = (s[0] + a[0], s[1] + a[1])
        if (
            not self.is_walkable(s_prime, visited=[]) or 
            self.maze[s_prime] == 0
        ): return s
        return s_prime
    
class MazeMDP(Maze):
    def __init__(self, gamma=0.99, maze=None, dim=20):
        self.gamma = gamma
        self.states = []
        super(MazeMDP, self).__init__(maze=maze, dim=dim)
        self.add_states()
        self.max_reward = len(self.states) * (10**2)
    
    def add_states(self):
        """
        Computes all possible states of this maze.
        Here, states of the maze = all non-walled positions in the maze.
        """
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                self.states.append((i, j))

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
        # Compute what s_prime should be.
        s_prime_true = (s[0] + a[0], s[1] + a[1])
        if not self.is_walkable(s_prime_true, visited=[]): 
            # If s_prime_true is a wall, then state remains s.
            s_prime_true = s
        # Compare that with what it is given to be.
        if s_prime_true == s_prime: return 1 # If same, then P(s, a, s') = 1.0.
        else: return 0  # Else, P(s, a, s') = 0.0.
    
    def R(self, s): # a, s' does not matter in this scenario
        """ 
        Rewards function which when given a state s and action a,
        returns the reward of taking action a in state s to end
        up in state s_prime.
        @param s: Current state.
        @param a: Action taken.
        @param s_prime: New state.
        """
        # Reward is 0 as long as s is not the goal and shall 
        # be a big positive number.
        # if (s[0]+a[0], s[1]+a[1]) != s_prime:
        #     return -100
        if s == self.goal: 
            return self.max_reward
        num_exits = 0
        for dir in self.actions:
            try:
                if self.maze[(s[0]+dir[0], s[1]+dir[1])] == 1:
                    num_exits += 1
            except:
                pass # could be invalid state that does not exist
        if num_exits >= 1: 
            if num_exits == 1: # dead end
                return -10 * self.maze.shape[0]
            return -1 * self.maze.shape[0]  # normal path
        return -100 * self.maze.shape[0] # invalid or closed path or is wall