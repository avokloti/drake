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
dir = "/Users/ira/Documents/drake/examples/robobee/output/random/"
solvers = ["admm", "snopt", "ipopt"]
colors = ["forestgreen", "orange", "blue"]
states_labels = ['X-position (m)', 'Y-position (m)', 'Z-position (m)', 'Roll Angle (rad)', 'Pitch Angle (rad)', 'Yaw angle (rad)']
inputs_labels = ['Thrust Force (N)', 'Roll Torque (Nm)', 'Yaw Torque (Nm)', 'Pitch Torque (Nm)']
styles = ['-', '--', ':']
names = ['ADMM', 'SNOPT', 'IPOPT']

index = 0

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

scale_factor = [1e-4, 1e-10, 1e-10, 1e-10]


## --- PLOTTING ---

# plot all states over time
handles = []
plt.figure()
for i in np.arange(6):
    plt.subplot(2, 3, i+1)
    plt.title(states_labels[i])
    for s in np.arange(len(solvers)):
        p, = plt.plot(np.linspace(0, Ts[s], Ns[s]), states_output[s][:, i+1], color=colors[s], label = solvers[s], linestyle = styles[s])
        if (i == 0):
            handles.append(p)
    plt.legend(handles[::-1], names[::-1])

plt.tight_layout()
plt.show()

# plot all inputs over time
handles = []
plt.figure(figsize = (18, 4))
for i in np.arange(num_inputs):
    plt.subplot(1, 4, i+1)
    plt.title(inputs_labels[i])
    for s in np.arange(len(solvers)):
        p, = plt.plot(np.linspace(0, Ts[s], Ns[s]), inputs_output[s][:, i+1] * scale_factor[i], color=colors[s], label = solvers[s], linestyle = styles[s])
        if (i == 0):
            handles.append(p)
    plt.legend(handles[::-1], names[::-1])

plt.show()

# make 3d plot of trajectories
fig = plt.figure()
ax = fig.gca(projection='3d')

for s in np.arange(len(solvers)):
    ax.plot(states_output[s][:, 1], states_output[s][:, 2], states_output[s][:, 3], color = colors[s], label = solvers[s], marker='o', markersize=3, linestyle = styles[s])
    ax.scatter(x0s[s][0], x0s[s][1], x0s[s][2], s=5, color='black')
    ax.scatter(xfs[s][0], xfs[s][1], xfs[s][2], s=5, color='black')

plt.title("Example Trajectories for Two-Stage Problem")
ax.set_xlabel('x-axis (m)')
ax.set_ylabel('y-axis (m)')
ax.set_zlabel('z-axis (m)')

ax.legend()
plt.tight_layout()
plt.show()


