import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

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

def readObstacleFile(filename):
    with open(filename) as f:
        content = f.readlines()
    xpos = np.array(content[0].strip().split()).astype('float')
    ypos = np.array(content[1].strip().split()).astype('float')
    rx = np.array(content[2].strip().split()).astype('float')
    ry = np.array(content[3].strip().split()).astype('float')
    return [xpos, ypos, rx, ry]

def cylinder(x0, y0, zmin, zmax, rx, ry):
    z_res = 5
    theta_res = 100
    theta = np.linspace(0, 2 * np.pi, theta_res)
    xx = np.reshape(np.tile(rx * np.cos(theta) + x0, z_res), [z_res, theta_res]);
    yy = np.reshape(np.tile(ry * np.sin(theta) + y0, z_res), [z_res, theta_res]);
    zz = np.reshape(np.repeat(np.linspace(zmin, zmax, z_res), theta_res), [z_res, theta_res]);
    return [xx, yy, zz]

# function to check equality of all elements in list
def checkEqual(iterator):
    return len(set(iterator)) <= 1

# function to check equality of list of lists
def checkEqualList(listoflists):
    if (len(listoflists) == 1):
        return True
    else:
        return ((listoflists[0] == listoflists[1]).all() and checkEqualList(listoflists[1:]))

def round_to_n(x, n):
    if np.isnan(x):
        return float('nan')
    elif (x == 0):
        return 0
    else:
        return round(x, -int(np.floor(np.log10(np.abs(x)))) + n - 1)

def processData(entry_index):
    data = [entry[entry_index] for entry in header_output]
    data = np.reshape(data, (num_solvers, num_opt_tols, num_feas_tols))
    return data

def processDataMeans(entry_index):
    data = [entry[entry_index] for entry in header_output]
    data = np.reshape(data, (num_solvers, num_trials))
    data_means = np.zeros((num_solvers, num_opt_tols, num_feas_tols))
    data_stds = np.zeros((num_solvers, num_opt_tols, num_feas_tols))
    for s in np.arange(num_solvers):
        dd = np.reshape(data[s], (num_trials, num_opt_tols, num_feas_tols))
        for ot in np.arange(num_opt_tols):
            for ft in np.arange(num_feas_tols):
                data_means[s, ot, ft] = np.mean(dd[:, ot, ft])
                data_stds[s, ot, ft] = np.std(dd[:, ot, ft])
    return data, data_means, data_stds


dir = "/Users/ira/Documents/drake/examples/quadrotor/output/snopt_ipopt_obstacles/"
#dir = "/Users/ira/Documents/drake/examples/quadrotor/output/snopt_ipopt_more/"

# solvers / colors
solvers = ["admm", "snopt", "ipopt"]
solver_names = ["ADMM", "SNOPT", "IPOPT"]
colors = ['maroon', 'blue', 'green']

# look at specific values defined by...
index = int(sys.argv[1])

num_solvers = len(solvers)
num_feas_tols = 6
num_opt_tols = 3
num_trials = 10
trial_start = num_feas_tols * num_opt_tols * index
trial_end = num_feas_tols * num_opt_tols * (index + 1)
num_trials = num_feas_tols * num_opt_tols


## ---- PROCESS DATA ---- ##

# read data from headers
headers = [dir + name + "_header_" + str(ind) + ".txt" for name in solvers for ind in np.arange(trial_start, trial_end)]
header_output = list(map(readHeaderFile, headers))

# separate data
feas_tols = processData(4)
opt_tols = processData(5)

feas_tols_list = np.flip(np.unique(feas_tols))
opt_tols_list = np.flip(np.unique(opt_tols))

iterations = processData(6)
time = processData(7)
feas = processData(9)
constraints = processData(11)
objectives = processData(12)
results = processData(13)

# find problems resulting in success, iteration limit, divergence
successes = [result == 'SolutionFound' for result in results]
itlim = [result == 'IterationLimit' for result in results]
divs = [result == 'ExceptionThrown' for result in results]


## ---- REPORT SOME INFORMATION ABOUT SOLVES ---- ##

# how many successes are there for specific tolerance combinations?
print("Grid of success/not success by each solver across tolerance grid:")
for s in np.arange(num_solvers):
    print("------------------------")
    print(solvers[s] + ":")
    print(successes[s])

print("------------------------")


## ---- PLOTTING ---- ##

# read in state and input data
states = [dir + s + "_x_" + str(ind) + ".txt" for s in solvers for ind in np.arange(trial_start, trial_end)]
inputs = [dir + s + "_u_" + str(ind) + ".txt" for s in solvers for ind in np.arange(trial_start, trial_end)]
obstacles = dir + "obstacles" + str(index) + ".txt"

# read all data
states_output = [np.loadtxt(f) for f in states]
inputs_output = [np.loadtxt(f) for f in inputs]
obstacle_output = readObstacleFile(obstacles)

# make 3d plot of trajectories
"""
for ot in np.arange(num_opt_tols):
    fig = plt.figure()
    for ft in np.arange(num_feas_tols):
        ax = fig.add_subplot(2, 3, ft + 1, projection='3d')
        plt.title("OT = " + str(opt_tols_list[ot]) + ", FT = " + str(feas_tols_list[ft]))
        for s in np.arange(num_solvers):
            ind = num_feas_tols * num_opt_tols * s + ot * num_feas_tols + ft
            ax.plot(states_output[ind][:, 1], states_output[ind][:, 2], states_output[ind][:, 3], color = colors[s], label = solver_names[s], marker='o', markersize=5)
        for obs in np.arange(len(obstacle_output[0])):
            [xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], -6, 6, obstacle_output[2][obs], obstacle_output[3][obs])
            ax.plot_surface(xx, yy, zz)
        
        #print("Figure " + str(ot) + ", subplot " + str(ft + 1) + ", inds: " + str(ot * num_feas_tols + ft + np.arange(num_solvers) * num_feas_tols * num_opt_tols))
        ax.legend()

#ax.scatter(x0s[s][0], x0s[s][1], x0s[s][2], s=5, color='black')
#plt.title("Trajectories for Trial " + str(index))
#ax.legend()

plt.show() """

lines = ["-","--", ":"]

# make 3d plot of trajectories
for ft in np.arange(num_feas_tols):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    plt.title("FT = " + str(feas_tols_list[ft]))
    for ot in np.arange(num_opt_tols):
        for s in np.arange(num_solvers):
            ind = num_feas_tols * num_opt_tols * s + ot * num_feas_tols + ft
            ax.plot(states_output[ind][:, 1], states_output[ind][:, 2], states_output[ind][:, 3], color = colors[s], label = solver_names[s], ls = lines[ot], marker='o', markersize=5)
        for obs in np.arange(len(obstacle_output[0])):
            [xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], -6, 6, obstacle_output[2][obs], obstacle_output[3][obs])
            ax.plot_surface(xx, yy, zz)
        
        #print("Figure " + str(ot) + ", subplot " + str(ft + 1) + ", inds: " + str(ot * num_feas_tols + ft + np.arange(num_solvers) * num_feas_tols * num_opt_tols))
        ax.legend()
        ax.set_xlim(-1, 7)
        ax.set_ylim(-1, 7)
        ax.set_zlim(-1, 7)

#ax.scatter(x0s[s][0], x0s[s][1], x0s[s][2], s=5, color='black')
#plt.title("Trajectories for Trial " + str(index))
#ax.legend()

plt.show()

# need to do
# resolve constraint errors occurring when they shouldn't
# clean up ADMM code
# 



"""
# Print info:
print()
print("----- Summary for Trial " + str(index) + " -----")
for s in np.arange(len(solvers)):
    print(solver_names[s] + ":\t" + solve_results[s] + ", objective = " + str(round_to_n(objectives[s], 3)) + ", runtime = " + str(round_to_n(times[s], 3)) + ", feasibility error = " + str(round_to_n(feasinfnorms[s], 3)) + ", constraint error = " + str(round_to_n(constraintinfnorms[s], 3)))
print()

# Print info:
print()
print("----- Summary for Trial " + str(index) + " -----")
for s in np.arange(len(solvers)):
print(solver_names[s] + " -------- ")
for ot in np.arange(num_opt_tols):
for ft in np.arange(num_opt_tols):
+ results[s] + ", objective = " + str(round_to_n(objectives[s], 3)) + ", runtime = " + str(round_to_n(times[s], 3)) + ", feasibility error = " + str(round_to_n(feasinfnorms[s], 3)) + ", constraint error = " + str(round_to_n(constraintinfnorms[s], 3)))
print()

# Plot 3D trajectories
states = [dir + s + "_x_" + str(index) + ".txt" for s in solvers]
inputs = [dir + s + "_u_" + str(index) + ".txt" for s in solvers]
#obstacles = dir + "obstacles" + str(index) + ".txt"

# read all data
states_output = [np.loadtxt(f) for f in states]
inputs_output = [np.loadtxt(f) for f in inputs]
obstacle_output = readObstacleFile(obstacles)

# make 3d plot of trajectories
fig = plt.figure()
ax = fig.gca(projection='3d')
for s in np.arange(len(solvers)):
    ax.plot(states_output[s][:, 1], states_output[s][:, 2], states_output[s][:, 3], color = colors[s], label = solver_names[s], marker='o', markersize=5)
ax.scatter(x0s[s][0], x0s[s][1], x0s[s][2], s=5, color='black')
for obs in np.arange(len(obstacle_output[0])):
    [xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], -6, 6, obstacle_output[2][obs], obstacle_output[3][obs])
    ax.plot_surface(xx, yy, zz)
plt.title("Trajectories for Trial " + str(index))
ax.legend()

#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#plt.savefig(dir + "kuka_states_with_opt.eps")

"""
