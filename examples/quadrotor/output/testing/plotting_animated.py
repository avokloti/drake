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
    solve_string = content[12].strip()
    solve_result = int(solve_string == "SolutionFound")
    return [N, T, x0, xf, tol, time, feas2norm, feasinfnorm, constraint2norm, constraintinfnorm, objective, solve_result]

def readObstacleFile(filename):
    with open(filename) as f:
        content = f.readlines()
    xpos = np.array(content[0].strip().split()).astype('float')
    ypos = np.array(content[1].strip().split()).astype('float')
    rx = np.array(content[2].strip().split()).astype('float')
    ry = np.array(content[3].strip().split()).astype('float')
    return [xpos, ypos, rx, ry]


# function to check equality of all elements in list
def checkEqual(iterator):
    return len(set(iterator)) <= 1

# function to check equality of list of lists
def checkEqualList(listoflists):
    if (len(listoflists) == 1):
        return True
    else:
        return ((listoflists[0] == listoflists[1]).all() and checkEqualList(listoflists[1:]))

def cylinder(x0, y0, zmin, zmax, r):
    z_res = 5
    theta_res = 100
    theta = np.linspace(0, 2 * np.pi, theta_res)
    xx = np.reshape(np.tile(r * np.cos(theta) + x0, z_res), [z_res, theta_res]);
    yy = np.reshape(np.tile(r * np.sin(theta) + y0, z_res), [z_res, theta_res]);
    zz = np.reshape(np.repeat(np.linspace(zmin, zmax, z_res), theta_res), [z_res, theta_res]);
    print(xx.shape)
    print(yy.shape)
    print(zz.shape)
    return [xx, yy, zz]

# directory
solvers = ["ali_res", "admm_al_ineq", "snopt", "ipopt"]
#solvers = ["admm_pen", "admm_al", "admm_al_ineq", "snopt"]
dir = "/Users/ira/Documents/drake/examples/quadrotor/output/al_ineq/"
trials = [0, 1, 2, 3]
#trials = [4]
experiment_names = ["Basic Problem", "Basic, Warm Start", "Obstacles", "Obstacles, Warm Start", "Obstacles Interpolated"]
solver_names = ["ALIres", "ALI", "SNOPT", "IPOPT"]
#solver_names = ["ADMM Penalty", "ADMM AuLa", "ADMM AuLaIneq", "SNOPT"]
colors = ['maroon', 'orange', 'blue', 'green']


# make bar plot with statistics
#plt.figure(figsize=(18.0, 12.0))

for trial in trials:
    headers = [dir + s + "_header_" + str(trial) + ".txt" for s in solvers]
    
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
    
    # make bar graphs
    y_pos = np.arange(len(solvers))

    plt.subplot(len(trials), 4, trial * 4 + 1)
    plt.bar(y_pos, times, align='center', alpha=0.5, color = colors)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if trial == (len(trials) - 1):
        plt.xticks(y_pos, solver_names, rotation='vertical')
    if trial == 0:
        plt.title('Runtime (sec)')
    plt.ylabel(experiment_names[trial])
    plt.ylim(0, 3)

    plt.subplot(len(trials), 4, trial * 4 + 2)
    plt.bar(y_pos, objectives, align='center', alpha=0.5, color = colors)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if trial == (len(trials) - 1):
        plt.xticks(y_pos, solver_names, rotation='vertical')
    if trial == 0:
        plt.title('Objective Value')
    plt.ylim(0, 1.8e2)
    
    plt.subplot(len(trials), 4, trial * 4 + 3)
    plt.bar(y_pos, feasinfnorms, align='center', alpha=0.5, color = colors)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if trial == (len(trials) - 1):
        plt.xticks(y_pos, solver_names, rotation='vertical')
    if trial == 0:
        plt.title('Feasibility Constraint Violation')
    plt.ylim(0, 6e-6)

    plt.subplot(len(trials), 4, trial * 4 + 4)
    plt.bar(y_pos, constraintinfnorms, align='center', alpha=0.5, color = colors)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if trial == (len(trials) - 1):
        plt.xticks(y_pos, solver_names, rotation='vertical')
    if trial == 0:
            plt.title('Other Constraint Violation')
    #plt.ylim(0, 6e-6)

#plt.savefig(dir + "bar_plot.eps")
plt.show()

# Plot 3D trajectories
for trial in trials:
    # state and input directories
    states = [dir + s + "_x_" + str(trial) + ".txt" for s in solvers]
    inputs = [dir + s + "_u_" + str(trial) + ".txt" for s in solvers]
    obstacles = dir + "obstacles" + str(trial) + ".txt"
    
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
        [xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], 0, 6, obstacle_output[2][obs])
        ax.plot_surface(xx, yy, zz)
    plt.title(experiment_names[trial])
    ax.legend()

plt.show()







# next step:
# add objective value to header
# add obstacle constraint violation to header
# repeat for all four experiments currently coded up


# read in standard weighted version data
x1 = np.loadtxt("/Users/ira/Documents/drake/examples/quadrotor/output/al_ineq/ali_res_traj_3.txt");
x2 = np.loadtxt("/Users/ira/Documents/drake/examples/quadrotor/output/al_ineq/admm_al_ineq_traj_3.txt");

num_states = 12
num_inputs = 4

if (x1.shape[1] != x2.shape[1]):
    print("N does not match!!")

N = int((x1.shape[1])/(num_states + num_inputs))
n1 = x1.shape[0]
n2 = x2.shape[0]

# reshape and process
y1 = x1[:, 0:(N * num_states)]
u1 = x1[:, (N * num_states):]
y1 = np.reshape(y1, [n1, N, num_states])
u1 = np.reshape(u1, [n1, N, num_inputs])

y2 = x2[:, 0:(N * num_states)]
u2 = x2[:, (N * num_states):]
y2 = np.reshape(y2, [n2, N, num_states])
u2 = np.reshape(u2, [n2, N, num_inputs])


def readHeaderFile(filename):
    with open(filename) as f:
        content = f.readlines()
    N = int(content[1].strip())
    T = float(content[2].strip())
    x0 = np.array(content[3].strip().split('\t')).astype(np.float)
    xf = np.array(content[4].strip().split('\t')).astype(np.float)
    max_iter = int(content[5].strip())
    time = float(content[6].strip())
    error2norm = float(content[7].strip())
    errorinfnorm = float(content[8].strip())
    return [N, T, x0, xf, max_iter, time, error2norm, errorinfnorm]





#    ----- 2D Plots -----


labels = ['ADMM', 'ALI']

fig = plt.figure()

opt1 = np.zeros(n1)
opt2 = np.zeros(n2)

for i in np.arange(n1):
    opt1[i] = sum(sum(u1[i,:,:]**2))

for i in np.arange(n2):
    opt2[i] = sum(sum(u2[i,:,:]**2))

plt.subplot(3, 1, 1)
plt.plot(np.arange(n1)+1, opt1, label = labels[0])
plt.plot(np.arange(n2)+1, opt2, label = labels[1])
plt.title("Optimality")

plt.subplot(3, 1, 2)
plt.title("Dynamic Feasibility")

plt.subplot(3, 1, 3)
plt.title("Obstacle Avoidance Violation")

plt.tight_layout()
plt.show()


#    ----- 3D Animation -----

import mpl_toolkits.mplot3d.axes3d as p3

def update_lines(i, dataLines, lines):
    for line, y in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        index = min(i, y.shape[0]-1)
        line.set_data(y[index, :, 0:2].transpose().tolist())
        line.set_3d_properties(y[index, :, 2])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)
for obs in np.arange(len(obstacle_output[0])):
    [xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], 0, 6, obstacle_output[2][obs])
    ax.plot_surface(xx, yy, zz)

# Fifty lines of random 3-D lines
data = [y1, y2]
lines = [ax.plot(y[0, :, 0], y[0, :, 1], y[0, :, 2], label = l)[0] for y, l in zip(data, labels)]

# Setting the axes properties
ax.set_xlim3d([0.0, 6.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 6.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 6.0])
ax.set_zlabel('Z')

ax.set_title('Solution Trajectories Over Time')
ax.legend(loc = 'lower right')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, min(n1, n2), fargs=(data, lines), interval=200, blit=False)

plt.show()





"""
def update_lines(i, dataLines, lines):
    for line, y in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        index = min(i, y.shape[0]-1)
        line.set_data(y[index, :, 0:2].transpose().tolist())
        line.set_3d_properties(y[index, :, 2])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)
for obs in np.arange(len(obstacle_output[0])):
    [xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], 0, 6, obstacle_output[2][obs])
    ax.plot_surface(xx, yy, zz)

# Fifty lines of random 3-D lines
data = [y1, y2]
lines = [ax.plot(y[0, :, 0], y[0, :, 1], y[0, :, 2], label = l)[0] for y, l in zip(data, labels)]

# Setting the axes properties
ax.set_xlim3d([0.0, 6.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 6.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 6.0])
ax.set_zlabel('Z')

ax.set_title('Solution Trajectories Over Time')
ax.legend(loc = 'lower right')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, min(n1, n2), fargs=(data, lines), interval=200, blit=False)

plt.show()
"""



