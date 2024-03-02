import json
import random
import numpy as np
import matplotlib.pyplot as plt
from maze_state import MazeState
from utility import get_best_reward
import matplotlib.animation as animation

def draw_maze(maze, save=None, solution=None, exploration=None, state_values=None):
    """ Visualizes given maze. """

    if not state_values is None:
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10,5))
            # Set the border color to white
        fig.patch.set_edgecolor('white')
        fig.patch.set_linewidth(0)

        axes[0].imshow(maze, interpolation='nearest')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        scatter_path = None 
        scatter_exp = None
        
        # Animate solution path.
        if solution is not None: # Id there is a solution, provided, draw it on the maze.
            def init():
                nonlocal scatter_path
                nonlocal scatter_exp
                scatter_path = axes[0].scatter([], [], marker='o', color='w', s=2)
                if exploration is not None: # Animate explored path as well.
                    scatter_exp = axes[0].scatter([], [], marker='o', color='r', s=2)
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

        # Plot values (value iteration) if available.
        axes[1].matshow(state_values, cmap=plt.cm.Blues)
    else:
        fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
        
        # Set the border color to white
        fig.patch.set_edgecolor('white')
        fig.patch.set_linewidth(0)

        axis.imshow(maze, interpolation='nearest')
        axis.set_xticks([])
        axis.set_yticks([])

        scatter_path = None 
        scatter_exp = None
        
        # Animate solution path.
        if not solution is None: # Id there is a solution, provided, draw it on the maze.
            def init():
                nonlocal scatter_path
                nonlocal scatter_exp
                scatter_path = axis.scatter([], [], marker='o', color='w', s=2)
                if not exploration is None: # Animate explored path as well.
                    scatter_exp = axis.scatter([], [], marker='o', color='r', s=2)
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
                if not exploration is None: 
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
        if save['animation']: # Optionally save animation.
            print('Saving animation ...')
            ani.save(f"{save['dir']}/{save['filename']}.mp4", writer="ffmpeg", fps=30)

    # Render figure.
    plt.show()

    if not save == None:
        fig.savefig(f'{save['dir']}/{save['filename']}.png')

def save_maze(maze, out_dir, out_file):
    ''' Saves given maze in a json file with the 
        given name in the given directory. 
    '''
    with open(f'{out_dir}/{out_file}.json', 'w') as f:
        json.dump(maze.matrix.tolist(), f)

def load_maze(path):
    ''' Loads and returns a given maze from a json  
        file at the given path. 
    '''
    with open(path, 'r') as f:
       matrix = np.array(json.load(f))
    return matrix

class Maze():
    """ Maze. """
    def __init__(self, min_epsilon=1e-6, max_gamma=0.99, dim=None, start=(1, 1), matrix=None, num_goals=1):
        """
        Initializes a maze.
        @param min_epsilon: Minimum value of change factor for which value
                            iteration would be complete on this maze.
        @param max_gamma: Maximum value of discount factor for which value
                          iteration would be complete on this maze.
        @param dim: Input dimension. Size of maze output = (dim*2+1 x dim*2+1).
        @param matrix: If not None, means that a new maze need not be created and 
                       that this maze is to be initialized with the given 
                       matrix. This is to allow loading of previously saved mazes.
        @param num_goals: No. of goals that this maze should have.
        """
        self.start = start
        self.min_epsilon = min_epsilon
        self.max_gamma = max_gamma
        if type(matrix) != type(None): # If the maze is given, and thus is not None ...
            self.matrix = matrix # Set matrix to given one.
            self.goals = self.__find_goals() # Find the goal.
            self.dim = (self.matrix.shape[0]-1)//2 # Get dimension input to generate the maze.
        else:
            self.dim = dim # Input dimension = 20 by default => maze size (20*2+1 x 20*2+1).
            self.matrix = np.zeros((dim*2+1, dim*2+1)) # Create a grid filled with walls (0s) only.
            self.__create_maze(num_goals=num_goals) # Add spaces, start and goal(s).
        self.reward = get_best_reward(dim=self.dim, epsilon=self.min_epsilon, gamma=self.max_gamma)
        self.actions = ["↑", "→", "↓", "←"] # Up, right, down, left.
        self.states, self.state_positions = self.__get_states() # Get valid (not walls and inside maze) states. 

    def __find_goals(self):
        ''' Return position of goals in this maze. '''
        goals = []
        for i in range(0, self.matrix.shape[0]):
            for j in range(0, self.matrix.shape[1]):
                if self.matrix[i, j] == 2: 
                    goals.append((i, j))
        return goals

    def __create_maze(self, num_goals):
        """ Creates a new maze with given no. of goals.
            # wall = 0
            # space = 1
            # start = 3
            # goal = 2
        """
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
                    and self.matrix[2*nx+1, 2*ny+1] == 0
                ):
                    self.matrix[2*nx+1, 2*ny+1] = 1
                    self.matrix[2*x+1+dx, 2*y+1+dy] = 1
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()
                
        # Create start.
        self.matrix[self.start] = 3 # start = 3
        
        # Create goals.
        goals_new = []
        while(len(goals_new) < num_goals):
            goal = (random.randint(1, 2*self.dim-1), random.randint(2, 2*self.dim))
            while ( 
                self.matrix[goal] != 0 or # If goal is not a wall or 
                not ( # if goal is on the outer wall or
                    1 <= goal[0] < self.matrix.shape[0]-1 and
                    1 <= goal[1] < self.matrix.shape[1]-1
                ) or (( # if there is a wall/nothing to the top of the goal
                    goal[1]-1 < 0
                    or goal[1]-1 >= self.matrix.shape[1]
                    or self.matrix[goal[0], goal[1]-1] == 0
                ) and ( # and there is a wall/nothing to the right of the goal
                    goal[0]+1 < 0
                    or goal[0]+1 >= self.matrix.shape[0]
                    or self.matrix[goal[0]+1, goal[1]] == 0
                ) and ( # and there is a wall/nothing to the bottom of the goal
                    goal[1]-1 < 0
                    or goal[1]+1 >= self.matrix.shape[1]
                    or self.matrix[goal[0], goal[1]+1] == 0
                ) and ( # and there is a wall/nothing to the left of the goal
                    goal[0]-1 < 0
                    or goal[0]-1 >= self.matrix.shape[0]
                    or self.matrix[goal[0]-1, goal[1]] == 0
                ))
            ): # then, look again ...
                goal = (random.randint(1, 2*self.dim-1), random.randint(2, 2*self.dim))
            self.matrix[goal] = 2 # goal = 2
            goals_new.append(goal)
        self.goals = goals_new
          
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
                up = (i-1, j)
                down = (i+1, j)
                right = (i, j+1)
                left = (i, j-1)

                # Update state to capture information about immediate surroundings.
                states[(i, j)] = MazeState(up=up, right=right, down=down, left=left)

        return states, state_positions

    def __str__(self):
        """ 
        Prints the matrix of the maze as a string with walls 
        represented by '#', space by ' ', start position by 'S',
        and goal positions by 'G'.
        """
        mat = np.full(self.matrix.shape, "#")
        for p in self.state_positions:
            mat[p] = " "
        mat[self.start] = "S"
        for goal in self.goals: mat[goal] = "G"
        mat = mat.tolist()
        mat_str = [str(row) for row in mat]
        return "\n".join(mat_str)

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

    def R(self, s):
        """ 
        Reward function which when given a state s,
        returns the reward of being in that state.
        @param s: Current state.
        """
        if s in self.goals: # If this is the goal, then positive reward.
            return self.reward
        return 0 # Else, no reward.