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
    tol = float(content[5].strip())
    time = float(content[6].strip())
    feas2norm = float(content[7].strip())
    feasinfnorm = float(content[8].strip())
    constraint2norm = float(content[9].strip())
    constraintinfnorm = float(content[10].strip())
    objective = float(content[11].strip())
    solve_result = content[12].strip()
    return [N, T, x0, xf, tol, time, feas2norm, feasinfnorm, constraint2norm, constraintinfnorm, objective, solve_result]

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


# overall experimental parameters/values
tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
knot_points = [10, 20, 30, 40]
num_trials = 10

solvers = ["admm_pen", "admm_al_ineq", "snopt", "ipopt"]
dir = "/Users/ira/Documents/drake/examples/quadrotor/output/param_sweep_v2/"
solver_names = ["ADMM", "ALI", "SNOPT", "IPOPT"]
colors = ['maroon', 'orange', 'blue', 'green']
trials = np.arange(num_trials);

# look at specific values defined by...
#index = knot_points.index(N) * len(tolerances) * num_trials + tolerances.index(tol) * num_trials;
# in this case, index = knot_points.index(N) * 50 + tolerances.index(tol) * 10 + trial
index = int(sys.argv[1])

headers = [dir + s + "_header_" + str(index) + ".txt" for s in solvers]

# for each solver, read header
header_output = list(map(readHeaderFile, headers))

# do some checks
Ns = [entry[0] for entry in header_output]
Ts = [entry[1] for entry in header_output]
x0s = [entry[2] for entry in header_output]
xfs = [entry[3] for entry in header_output]
tols = [entry[4] for entry in header_output]

if not checkEqual(Ns) or not checkEqual(Ts) or not checkEqualList(x0s) or not checkEqualList(xfs):
    print("HEADERS DO NOT MATCH!")

times = [entry[5] for entry in header_output]
feas2norms = [entry[6] for entry in header_output]
feasinfnorms = [entry[7] for entry in header_output]
constraint2norms = [entry[8] for entry in header_output]
constraintinfnorms = [entry[9] for entry in header_output]
objectives = [entry[10] for entry in header_output]
solve_results = [entry[11] for entry in header_output]

# Print info:
print()
print("----- Summary for Trial " + str(index) + " -----")
for s in np.arange(len(solvers)):
    print(solver_names[s] + ":\t" + solve_results[s] + ", objective = " + str(round_to_n(objectives[s], 3)) + ", runtime = " + str(round_to_n(times[s], 3)) + ", feasibility error = " + str(round_to_n(feasinfnorms[s], 3)) + ", constraint error = " + str(round_to_n(constraintinfnorms[s], 3)))
print()

# Plot 3D trajectories
states = [dir + s + "_x_" + str(index) + ".txt" for s in solvers]
inputs = [dir + s + "_u_" + str(index) + ".txt" for s in solvers]
obstacles = dir + "obstacles" + str(index) + ".txt"

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

