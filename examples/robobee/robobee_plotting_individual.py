import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3


# function to read all info from headers
def readHeaderFile(filename):
    with open(filename) as f:
        content = f.readlines()
    N = int(content[1].strip())
    T = float(content[2].strip())
    x0 = np.array(content[3].strip().split('\t')).astype(np.float)
    xf = np.array(content[4].strip().split('\t')).astype(np.float)
    time = float(content[5].strip())
    feas2norm = float(content[6].strip())
    feasinfnorm = float(content[7].strip())
    constraint2norm = float(content[8].strip())
    constraintinfnorm = float(content[9].strip())
    objective = float(content[10].strip())
    solve_result = content[11].strip()
    return [N, T, x0, xf, time, feas2norm, feasinfnorm, constraint2norm, constraintinfnorm, objective, solve_result]

# directory/solver names
dir = "/Users/ira/Documents/drake/examples/complete_robobee/output/random/"
solvers = ["admm", "snopt", "ipopt"]
colors = ["maroon", "green", "blue"]
index = 5

num_states = 12
num_inputs = 4

# for each solver, make file names
headers = [dir + s + "_header_" + str(index) + ".txt" for s in solvers]
states = [dir + s + "_x_" + str(index) + ".txt" for s in solvers]
inputs = [dir + s + "_u_" + str(index) + ".txt" for s in solvers]

# read header file
header_output = list(map(readHeaderFile, headers))

Ns = [entry[0] for entry in header_output]
Ts = [entry[1] for entry in header_output]
x0s = [entry[2] for entry in header_output]
xfs = [entry[3] for entry in header_output]

# read all data
states_output = [np.loadtxt(f) for f in states]
inputs_output = [np.loadtxt(f) for f in inputs]


## --- PLOTTING ---

# plot all states over time
plt.figure()
for i in np.arange(6):
    plt.subplot(2, 3, i+1)
    plt.title("State " + str(i) + " for Trial " + str(index))
    for s in np.arange(len(solvers)):
        plt.plot(np.linspace(0, Ts[s], Ns[s]), states_output[s][:, i+1], color=colors[s], label = solvers[s])

plt.legend()
plt.tight_layout()

# plot all inputs over time
plt.figure()
for i in np.arange(num_inputs):
    plt.subplot(2, 2, i+1)
    plt.title("Input " + str(i) + " for Trial " + str(index))
    for s in np.arange(len(solvers)):
        plt.plot(np.linspace(0, Ts[s], Ns[s]), inputs_output[s][:, i+1], color=colors[s], label = solvers[s])

plt.legend()
plt.tight_layout()

# make 3d plot of trajectories
fig = plt.figure()
ax = fig.gca(projection='3d')

for s in np.arange(len(solvers)):
    ax.plot(states_output[s][:, 1], states_output[s][:, 2], states_output[s][:, 3], color = colors[s], label = solvers[s], marker='o', markersize=5)
    ax.scatter(x0s[s][0], x0s[s][1], x0s[s][2], s=5, color='black')
    ax.scatter(xfs[s][0], xfs[s][1], xfs[s][2], s=5, color='black')

plt.title("Trajectories for Trial " + str(index))
ax.legend()

plt.tight_layout()
plt.show()


