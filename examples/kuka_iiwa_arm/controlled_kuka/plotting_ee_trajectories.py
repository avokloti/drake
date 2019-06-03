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
    rho0 = float(content[5].strip())
    rho1 = float(content[6].strip())
    rho2 = float(content[7].strip())
    iterations = int(content[8].strip())
    time = float(content[9].strip())
    feas2norm = float(content[10].strip())
    feasinfnorm = float(content[11].strip())
    constraint2norm = float(content[12].strip())
    constraintinfnorm = float(content[13].strip())
    objective = float(content[14].strip())
    solve_result = content[15].strip()
    return [N, T, x0, xf, rho0, rho1, rho2, iterations, time, feas2norm, feasinfnorm, constraint2norm, constraintinfnorm, objective, solve_result]


def readEndEffectorFile(filename):
    with open(filename) as f:
        content = f.readlines()
    N = len(content)
    t = np.zeros(N)
    ee = np.zeros((N, 3))
    for i in np.arange(N):
        line = np.array(content[i].strip().split()).astype(np.float)
        t[i] = line[0]
        ee[i, :] = line[1:4]
    return t, ee




## --- Main part of script --- ##


num_trials = 1

# set solver information
solvers = ["admm", "snopt", "ipopt"]
dir = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/obstacles/";
solver_names = ["ADMM", "SNOPT", "IPOPT"]
styles = ['-', '--', ':']
colors = ['maroon', 'green', 'blue']

# read data from headers
ee_trajs = [dir + solver + "_ee_" + str(index) + ".txt" for index in np.arange(num_trials) for solver in solvers]
ees = list(map(readEndEffectorFile, ee_trajs))

# --- plot end effector trajectories in 3d --- #
fig = plt.figure()
ax = fig.gca(projection='3d')
for s in np.arange(len(solvers)):
    ax.plot(ees[s][1][:, 0], ees[s][1][:, 1], ees[s][1][:, 2], color = colors[s], label = solver_names[s], marker='o', linestyle=styles[s], markersize=5)

#for obs in np.arange(len(obstacle_output[0])):
#[xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], -6, 6, obstacle_output[2][obs], obstacle_output[3][obs])
#ax.plot_surface(xx, yy, zz)
plt.title("Trajectories for Trial " + str(0))
ax.legend()
plt.show()


"""

# read headers later
header_output = list(map(readHeaderFile, headers))

# separate data
iterations = [entry[7] for entry in header_output]
times = [entry[8] for entry in header_output]
objectives = [entry[13] for entry in header_output]
rho0list = [entry[4] for entry in header_output]
rho1list = [entry[5] for entry in header_output]
rho2list = [entry[6] for entry in header_output]
results = [entry[14] for entry in header_output]

# find problems resulting in success, iteration limit, divergence
successes = [result == 'SolutionFound' for result in results]
itlim = [result == 'IterationLimit' for result in results]
divs = [result == 'ExceptionThrown' for result in results]

# reshaped form
successes = np.reshape(successes, (num_problems, num_trials))
iterations = np.reshape(iterations, (num_problems, num_trials))
objectives = np.reshape(objectives, (num_problems, num_trials))
times = np.reshape(times, (num_problems, num_trials))
rho0list = np.reshape(rho0list, (num_problems, num_trials))
print(np.mean(rho0list, axis=1))

# make arrays to hold sd and means
iterations_mean = np.zeros(num_problems)
objectives_mean = np.zeros(num_problems)
times_mean = np.zeros(num_problems)
iterations_sd = np.zeros(num_problems)
objectives_sd = np.zeros(num_problems)
times_sd = np.zeros(num_problems)

# calculate all sd and means
for i in np.arange(num_problems):
    iterations_mean[i] = np.mean(iterations[i, successes[i,:]])
    objectives_mean[i] = np.mean(objectives[i, successes[i,:]])
    times_mean[i] = np.mean(times[i, successes[i,:]])
    iterations_sd[i] = np.std(iterations[i, successes[i,:]])
    objectives_sd[i] = np.std(objectives[i, successes[i,:]])
    times_sd[i] = np.std(times[i, successes[i,:]])

# find number of successful runs (out of 10) for each rho1/rho2 combination
successes_sum = np.sum(successes, axis=1)

## -------------------- PLOTTING -------------------- ##

"""
