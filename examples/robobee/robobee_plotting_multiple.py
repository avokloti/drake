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
num_trials = 20
index = np.arange(num_trials)

num_states = 12
num_inputs = 4

# for each solver, make file names
headers = [dir + s + "_header_" + str(ind) + ".txt" for s in solvers for ind in index]
states = [dir + s + "_x_" + str(ind) + ".txt" for s in solvers for ind in index]
inputs = [dir + s + "_u_" + str(ind) + ".txt" for s in solvers for ind in index]

# read header file
header_output = list(map(readHeaderFile, headers))

Ns = [entry[0] for entry in header_output]
Ts = [entry[1] for entry in header_output]
x0s = [entry[2] for entry in header_output]
xfs = [entry[3] for entry in header_output]
times = [entry[4] for entry in header_output]
objectives = [entry[9] for entry in header_output]
feasinfnorms = [entry[5] for entry in header_output]
solve_results = [entry[10] for entry in header_output]

# read all data
states_output = [np.loadtxt(f) for f in states]
inputs_output = [np.loadtxt(f) for f in inputs]

# calculate mean and sd of successful objectives
success_admm = [i for i, e in enumerate(solve_results[0:20]) if e == 'SolutionFound']
success_snopt = [20 + i for i, e in enumerate(solve_results[20:40]) if e == 'SolutionFound']
success_ipopt = [40 + i for i, e in enumerate(solve_results[40:60]) if e == 'SolutionFound']

obj_admm = [objectives[i] for i in success_admm]
obj_snopt = [objectives[i] for i in success_snopt]
obj_ipopt = [objectives[i] for i in success_ipopt]

# correct admm objectives[5]
obj_admm = []
obj_snopt = []
obj_ipopt = []

for i in success_admm:
    this_states = states_output[i]
    this_obj = 0
    for j in np.arange(Ns[0]):
        this_obj = this_obj + np.linalg.norm(this_states[j,1:])**2
    obj_admm.append(this_obj)

for i in success_snopt:
    this_states = states_output[i]
    for j in np.arange(Ns[0]):
        this_obj = this_obj + np.linalg.norm(this_states[j,1:])**2
    obj_snopt.append(this_obj)

for i in success_ipopt:
    this_states = states_output[i]
    for j in np.arange(Ns[0]):
        this_obj = this_obj + np.linalg.norm(this_states[j,1:])**2
    obj_ipopt.append(this_obj)



obj_all = [obj_admm, obj_snopt, obj_ipopt]


feas_admm = [feasinfnorms[i] for i in success_admm]
feas_snopt = [feasinfnorms[i] for i in success_snopt]
feas_ipopt = [feasinfnorms[i] for i in success_ipopt]
feas_all = [feas_admm, feas_snopt, feas_ipopt]

times_admm = [times[i] for i in success_admm]
times_snopt = [times[i] for i in success_snopt]
times_ipopt = [times[i] for i in success_ipopt]
times_all = [times_admm, times_snopt, times_ipopt]

plt.figure()

plt.subplot(3, 1, 1)
ax = plt.gca()
plt.boxplot(obj_all)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
plt.title("Objective Value Across Solvers")
#plt.ylabel("objective value")
plt.ylim([0, 1000])

plt.subplot(3, 1, 2)
ax = plt.gca()
plt.boxplot(feas_all)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
plt.title("Constraint Violation Across Solvers")
plt.ylabel("infinity-norm error")

plt.subplot(3, 1, 3)
ax = plt.gca()
plt.boxplot(times_all)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
ax.ticklabel_format(style = 'sci', axis='y')
plt.title("Runtime Across Solvers (seconds)")
#plt.ylabel("seconds")

plt.tight_layout()
plt.show()



## --- PLOTTING ---

# make a 3D plot of all trajectories
fig = plt.figure()
ax = fig.gca(projection='3d')
for s in np.arange(len(solvers)):
    for i in np.arange(num_trials):
        ind = num_trials * s + i
        ax.plot(states_output[ind][:, 1], states_output[ind][:, 2], states_output[ind][:, 3], color = colors[s], label = solvers[s], marker='o', markersize=1, linewidth=1)
        ax.scatter(x0s[ind][0], x0s[ind][1], x0s[ind][2], s=2, color='black')
        ax.scatter(xfs[ind][0], xfs[ind][1], xfs[ind][2], s=2, color='black')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
plt.show()

styles = ['-', '--', ':']
names = ['ADMM', 'SNOPT', 'IPOPT']

# make a 3D plot of all trajectories
handles = []

fig = plt.figure()
ax = fig.gca(projection='3d')
for s in np.arange(len(solvers)):
    for i in np.arange(num_trials):
        ind = num_trials * s + i
        p, = ax.plot(states_output[ind][:, 1], states_output[ind][:, 2], states_output[ind][:, 3], color = colors[s], label = solvers[s], linewidth=2, linestyle = styles[s])
        if (i == 0):
            handles.append(p)
        ax.scatter(x0s[ind][0], x0s[ind][1], x0s[ind][2], s=3, color='black')
        ax.scatter(xfs[ind][0], xfs[ind][1], xfs[ind][2], s=3, color='black')

ax.set_xlabel('x-axis (m)')
ax.set_ylabel('y-axis (m)')
ax.set_zlabel('z-axis (m)')
plt.legend(handles[::-1], names[::-1])
plt.tight_layout()
#plt.title('Solution Trajectories', fontweight='bold')
plt.show()


styles = ['-', '--', ':']
names = ['ADMM', 'SNOPT', 'IPOPT']

# make a 3D plot of all trajectories
handles = []

fig = plt.figure()
for s in np.arange(len(solvers)):
    for i in np.arange(num_trials):
        ind = num_trials * s + i
        p, = plt.plot(states_output[ind][:, 1], states_output[ind][:, 3], color = colors[s], label = solvers[s], linewidth=2, linestyle = styles[s])
        if (i == 0):
            handles.append(p)
        plt.scatter(x0s[ind][0], x0s[ind][2], s=3, color='black')
        plt.scatter(xfs[ind][0], xfs[ind][2], s=3, color='black')

#handles_ord = [handles[]
plt.title('Solution Trajectories (2-D Projection)')
plt.xlabel('x-axis (m)')
plt.ylabel('z-axis (m)')
plt.legend(handles[::-1], names[::-1])
plt.show()


index = 0

# make a bar plot for mean/sd of optimal objective value
plt.figure()

               # to do now
               # make a nice plot of states and inputs


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


