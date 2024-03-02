Please run the following commands to run the experiments 
mentioned in the report involving each of the 5 algorithms.

########### TO CREATE A RANDOM MAZE AND SOLVE IT ###########
This is the fastest way to see the algorithms working on a
randomly generated maze of your chosen size. It is advised to
pick dim in the range [2, 50] as these are the sizes that
have been experimented with most. Smaller sizes are not
possible. While bigger sizes are possible, solvers will be slower,
especially Value Iteration and Policy Iteration.

A given input dim will generate mazes of size (dim*2+1 x dim*2+1).
That is, dim = 2 => maze size = (5 x 5) and dim = 50 => maze size (101 x 101).

To learn more about what command line arguments are possible, please refer to
file "solve_random_maze.py".

Once each of the following commands are run, output may 
be found in a folder "out_<name of solver>" that gets created in the
current working directory which will contain a file in the format
dayMonthYearHoursMinsSecs with the latest run result. The output 
here will be a .png file and a .txt file.

To solve a random maze using DFS:
    # Run > python solve_random_maze.py -dim 10 -ng 1 -s "dfs"

To solve a random maze using BFS:
    # Run > python solve_random_maze.py -dim 10 -ng 1 -s "bfs"

To solve a random maze using A* search:
    # Run > python solve_random_maze.py -dim 10 -ng 1 -s "astar"

To solve a random maze using Value Iteration:
    # Run > python solve_random_maze.py -dim 10 -ng 1 -s "valiter"
    # Default values (ideal):
        * discount factor = 0.99
        * change threshold = 1e-6
        * max iterations limit = None

To solve a random maze using Policy Iteration:
    # Run > python solve_random_maze.py -dim 10 -ng 1 -s "politer"
    # Default values (ideal):
        * discount factor = 0.99
        * max iterations limit = None

Here, -dim = dimension, -ng = no. of goals in the maze and -s = solver to use.

########## TO RUN ALL EXPERIMENTS FROM THE REPORT ##########

If you'd like to recreate results of the experiments mentioned 
in the report, please follow the steps below. 

WARNING!!! These experiments are time consuming.

If you'd like to save the animation of solutions as an mp4 file
just like it was done during the experiments, set the --save-anim 
flag while entering the command below. NOTE!!! This feature requires 
that your machine has ffmpeg installed and registered in PATH.

IMPORTANT!!! Before running any solver, make sure to create 
a new folder called "output" or anything else other than "__mazes" 
in the same directory as where the code files are. 
This is so that output in the "__mazes" folder which is what is 
referenced in the report, does not get overwritten.
Also ensure that this directory contains the "__mazes" folder with 
contents as submitted in the zip file. This folder contains mazes that 
were used for the experiments as mentioned in the report.

To run A* search experiments:
    python a_star.py --load-dir __mazes --save-dir output

To run DFS experiments:
    python dfs.py --load-dir __mazes --save-dir output

To run BFS experiments:
    python bfs.py --load-dir __mazes --save-dir output

To run Value Iteration experiments:
    python value_iteration.py --load-dir __mazes --save-dir output

To run Policy Iteration experiments:
    python policy_iteration.py --load-dir __mazes --save-dir output

If you'd like to generate 16 new mazes to run the experiments 
on fresh mazes, simply create a new folder where you'd like to
save your new random mazes and then, run the following:
    python init_mazes.py --dst-dir "path/to/your/new/folder"
Please note that this will generate 20 31x31 files from which 15 have to be
manually deleted and remaining ones will have to be renamed to have ids 1 to 5.

If you'd like to try any of the other functions on their own:
    # Please just import them and give it a go in the "playground.ipynb" python notebook file. 
    # That said, please note that the draw_maze() function does not display animations
      and solution path on the maze in a notebook environment as matplotlib.animation
      does not support this natively.
    # The playground.ipynb file is where experimentation with single functions
      was done during development.
    # Some of these experiments are still there and are commented out.

################## DEMONSTRATION VIDEO ##################

Please find demo video here: https://youtu.be/k1Deg2nf1EE

Kindly note that the demo was made prior to facilitating sending of 
command line arguments and the solve_random_maze.py file.

######################## GITHUB ########################
Please find the GitHub repository for this project here:
https://github.com/ggn1/cs7is2-artificial-intelligence-assignment1