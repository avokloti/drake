import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# function to read all info from headers
def readHeaderFile(filename):
    with open(filename) as f:
        content = f.readlines()
    N = int(content[1].strip())
    T = float(content[2].strip())
    x0 = np.array(content[3].strip().split('\t')).astype(np.float)
    xf = np.array(content[4].strip().split('\t')).astype(np.float)
    feas_tol = float(content[5].strip())
    opt_tol = float(content[6].strip())
    iterations = int(content[7].strip())
    time = float(content[8].strip())
    feas2norm = float(content[9].strip())
    feasinfnorm = float(content[10].strip())
    constraint2norm = float(content[11].strip())
    constraintinfnorm = float(content[12].strip())
    objective = float(content[13].strip())
    solve_result = content[14].strip()
    return [N, T, x0, xf, feas_tol, opt_tol, iterations, time, feas2norm, feasinfnorm, constraint2norm, constraintinfnorm, objective, solve_result]

dir = "/Users/ira/Documents/drake/examples/quadrotor/output/snopt_ipopt/"

# solvers / colors
solvers = ["admm", "snopt", "ipopt"]
colors = ['maroon', 'blue', 'green']

num_solvers = len(solvers)
num_feas_tols = 6
num_opt_tols = 3
num_trials = 10
total_trials = num_feas_tols * num_opt_tols * num_trials

# read data from headers
headers = [dir + name + "_header_" + str(index) + ".txt" for name in solvers for index in np.arange(total_trials)]
header_output = list(map(readHeaderFile, headers))

# separate data
feas_tols = [entry[4] for entry in header_output]
opt_tols = [entry[5] for entry in header_output]
iterations = [entry[6] for entry in header_output]
times = [entry[7] for entry in header_output]
objectives = [entry[12] for entry in header_output]
feasinfnorm = [entry[9] for entry in header_output]
constraintinfnorm = [entry[11] for entry in header_output]
results = [entry[13] for entry in header_output]

# find problems resulting in success, iteration limit, divergence
successes = [result == 'SolutionFound' for result in results]
itlim = [result == 'IterationLimit' for result in results]
divs = [result == 'ExceptionThrown' for result in results]

# reshaped form
feas_tols = np.reshape(feas_tols, (num_solvers, total_trials))
opt_tols = np.reshape(opt_tols, (num_solvers, total_trials))
successes = np.reshape(successes, (num_solvers, total_trials))
iterations = np.reshape(iterations, (num_solvers, total_trials))
objectives = np.reshape(objectives, (num_solvers, total_trials))
constraints = np.reshape(constraintinfnorm, (num_solvers, total_trials))
feasibility = np.reshape(feasinfnorm, (num_solvers, total_trials))
times = np.reshape(times, (num_solvers, total_trials))

# for each
