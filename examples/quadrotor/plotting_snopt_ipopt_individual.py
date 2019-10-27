import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

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

def circle(x0, y0, rx, ry):
    theta_res = 100
    theta = np.linspace(0, 2 * np.pi, theta_res)
    xx = rx * np.cos(theta) + x0
    yy = ry * np.sin(theta) + y0
    return [xx, yy]

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


#dir = "/Users/ira/Documents/drake/examples/quadrotor/output/snopt_ipopt_obstacles/"
dir = "/Users/ira/Documents/drake/examples/quadrotor/output/alg_compare_obstacles/"

# solvers / colors
solvers = ["admm", "snopt", "ipopt"]
solver_names = ["ADMM", "SNOPT", "IPOPT"]
colors = ['maroon', 'blue', 'green']

# look at specific values defined by...
index = int(sys.argv[1])
sub_index = int(sys.argv[2])

num_solvers = len(solvers)
num_feas_tols = 6
num_opt_tols = 3
#num_trials = 10

trial_start = num_feas_tols * num_opt_tols * index
trial_end = num_feas_tols * num_opt_tols * (index + 1)

obs_flag = True
try:
    obstacles = dir + "obstacles" + str(index) + ".txt"
    obstacle_output = readObstacleFile(obstacles)
except FileNotFoundError:
    obs_flag = False




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


## ---- PLOTTING CHANGES IN METRICS BETWEEN DIFFERENT TOLERANCES ---- ##

# plot runtime for snopt, ipopt, admm over tolerance values
lines = ["-","--", ":"]

plt.figure(figsize=(15, 10))

# plot time with error bars on single plot
plt.subplot(2, 2, 1)
for s in np.arange(num_solvers):
    for ot in np.arange(num_opt_tols):
        plt.plot(np.arange(num_feas_tols), time[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])

plt.xticks(np.arange(num_feas_tols), feas_tols_list)
plt.title('Runtimes Curves Across Tolerances')
plt.xlabel('Constraint Tolerance Value')
plt.ylabel('Runtime (sec)')

# plot objectives with error bars
plt.subplot(2, 2, 3)
for s in np.arange(num_solvers):
    for ot in np.arange(num_opt_tols):
        plt.plot(np.arange(num_feas_tols), objectives[s, ot, :],  color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])

plt.xticks(np.arange(num_feas_tols), feas_tols_list)
plt.title('Objectives for Solvers Across Tolerances')
plt.xlabel('Constraint Tolerance Value')
plt.ylabel('Objective')


# plot constraints with error bars on same plot
plt.subplot(2, 2, 2)
for s in np.arange(num_solvers):
    for ot in np.arange(num_opt_tols):
        plt.plot(np.arange(num_feas_tols), constraints[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])
    plt.xticks(np.arange(num_feas_tols), feas_tols_list)
    plt.title('Other Constraint Error Across Tolerances')
    plt.xlabel('Other Constraint Tolerance Value')
    plt.ylabel('Error (inf-norm)')

plt.legend()

plt.subplot(2, 2, 4)
for s in np.arange(num_solvers):
    for ot in np.arange(num_opt_tols):
        plt.plot(np.arange(num_feas_tols), np.log10(feas[s, ot, :]), color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])
    plt.xticks(np.arange(num_feas_tols), feas_tols_list)
    plt.title('Log10 of Dynamics Error Across Tolerances')
    plt.xlabel('Constraint Tolerance Value')
    plt.ylabel('Log10 of Error (inf-norm)')

plt.tight_layout()
plt.show()





## ---- PLOTTING TRAJECTORIES AND STATES/INPUTS ---- ##

# read in state and input data
states = [dir + s + "_x_" + str(ind) + ".txt" for s in solvers for ind in np.arange(trial_start, trial_end)]
inputs = [dir + s + "_u_" + str(ind) + ".txt" for s in solvers for ind in np.arange(trial_start, trial_end)]

# read all data
states_output = [np.loadtxt(f) for f in states]
inputs_output = [np.loadtxt(f) for f in inputs]

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




"""
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

plt.show() """

# need to do
# resolve constraint errors occurring when they shouldn't
# clean up ADMM code
#


## --- MAKE ADMM ANIMATION FOR A TRIAL --- ##

# transform 'y' vector into 'x' and 'u' state/input matrices
def y2xu(y):
    x = np.reshape(y[0:(N * num_states)], (N, num_states))
    u = np.reshape(y[(N * num_states):(N * num_states + N * num_inputs)], (N, num_inputs))
    return [x, u]

N = 20
T = 5.0
num_states = 12
num_inputs = 4


# directory
traj_filename = dir + "admm_traj_" + str(index * num_feas_tols * num_opt_tols + sub_index) + ".txt"
print(traj_filename)
with open(traj_filename) as f:
    content = f.readlines()

traj = [np.array(line.strip().split()).astype(np.float) for line in content]
snopt_traj = states_output[2 * num_opt_tols * num_feas_tols - 1]
ipopt_traj = states_output[3 * num_opt_tols * num_feas_tols - 1]

def drawAnimation3D():
    # animate 3d trajectory?
    def update_lines(num, line):
        x, u = y2xu(traj[num])
        line.set_data(x[:, 0], x[:, 1])
        line.set_3d_properties(x[:, 2])
        return line
    
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = Axes3D(fig)
    x, u = y2xu(traj[0])
    
    if (obs_flag):
        ax.set_xlim3d([-1.0, 6.0])
        ax.set_ylim3d([-1.0, 6.0])
        ax.set_zlim3d([-2.0, 2.0])
        for obs in np.arange(len(obstacle_output[0])):
            [xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], -6, 6, obstacle_output[2][obs], obstacle_output[3][obs])
            ax.plot_surface(xx, yy, zz)
    
    line = ax.plot(x[:, 0], x[:, 1], x[:, 2], c=colors[0])
    ax.plot(snopt_traj[:, 1], snopt_traj[:, 2], snopt_traj[:, 3], c=colors[1])
    ax.plot(ipopt_traj[:, 1], ipopt_traj[:, 2], ipopt_traj[:, 3], c=colors[2])
    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title('3D Test')
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, len(traj), fargs=line, interval=100, blit=False)
    plt.show()


def drawAnimation2D():
    # animate 3d trajectory?
    def update_lines(num, line):
        x, u = y2xu(traj[num])
        line.set_data(x[:, 0], x[:, 1])
        return line
    
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = plt.gca()
    x, u = y2xu(traj[0])
    
    ax.set_xlim([-1.0, 7.0])
    ax.set_ylim([-1.0, 7.0])
    
    if obs_flag:
        for obs in np.arange(len(obstacle_output[0])):
            ell = Ellipse((obstacle_output[0][obs], obstacle_output[1][obs]), 2 * obstacle_output[2][obs], 2 * obstacle_output[3][obs])
            ax.add_artist(ell)
            ell.set_alpha(0.8)
            ell.set_facecolor('grey')
    
    line = ax.plot(x[:, 0], x[:, 1], c=colors[0])
    ax.plot(snopt_traj[:, 1], snopt_traj[:, 2], c=colors[1])
    ax.plot(ipopt_traj[:, 1], ipopt_traj[:, 2], c=colors[2])
    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_title('2D Test')
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, len(traj), fargs=line, interval=500, blit=False)
    plt.show()


def drawFinal2D():
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = plt.gca()
    x, u = y2xu(traj[-1])
    
    if obs_flag:
        for obs in np.arange(len(obstacle_output[0])):
            ell = Ellipse((obstacle_output[0][obs], obstacle_output[1][obs]), 2 * obstacle_output[2][obs], 2 * obstacle_output[3][obs])
            ax.add_artist(ell)
            ell.set_alpha(0.8)
            ell.set_facecolor('grey')
    
    line = ax.plot(x[:, 0], x[:, 1], c=colors[0])
    ax.plot(snopt_traj[:, 1], snopt_traj[:, 2], c=colors[1])
    ax.plot(ipopt_traj[:, 1], ipopt_traj[:, 2], c=colors[2])
    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_title('2D Test')
    # Creating the Animation object
    plt.show()

def drawFinal3D():
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = Axes3D(fig)
    x, u = y2xu(traj[-1])
    
    if (obs_flag):
        ax.set_xlim3d([-1.0, 6.0])
        ax.set_ylim3d([-1.0, 6.0])
        ax.set_zlim3d([-2.0, 2.0])
        for obs in np.arange(len(obstacle_output[0])):
            [xx, yy, zz] = cylinder(obstacle_output[0][obs], obstacle_output[1][obs], -6, 6, obstacle_output[2][obs], obstacle_output[3][obs])
            ax.plot_surface(xx, yy, zz)
    
    line = ax.plot(x[:, 0], x[:, 1], x[:, 2], c=colors[0])
    ax.plot(snopt_traj[:, 1], snopt_traj[:, 2], snopt_traj[:, 3], c=colors[1])
    ax.plot(ipopt_traj[:, 1], ipopt_traj[:, 2], ipopt_traj[:, 3], c=colors[2])
    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


drawFinal2D()
drawFinal3D()
drawAnimation2D()
drawAnimation3D()


"""
# animate all inputs
fig, all_axes = plt.subplots(2, 4, figsize=(14, 8))
all_axes = np.ndarray.flatten(all_axes)
all_lines = []
t = np.linspace(0, T, N)

for i in np.arange(num_inputs):
    ax = all_axes[i]
    ax.set_title("Input " + str(i+1))
    ax.set_xlim(0, T)
    index = N * num_states + i * num_inputs
    ax.set_ylim(min(l[-1][index : index + num_inputs]), max(l[-1][index : index + num_inputs]))
    line, = ax.plot([], [], lw=2)
    all_lines.append(line)

def update_input_plot(i):
    t = np.linspace(0, T, N)
    for ii in np.arange(num_inputs):
        x, u = y2xu(l[i])
        all_lines[ii].set_data(t, u[:, ii])
    return all_lines

# Setting the axes properties
plt.title('Solution Trajectories Over Time')
#plt.legend(loc = 'lower right')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_input_plot, len(l), interval=100, blit=False)

plt.show() """




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
