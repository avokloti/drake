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
solvers = ["admm", "ipopt", "snopt"]
solver_names = ["ADMM", "IPOPT", "SNOPT"]
trials = [0]
dir = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/basic40/"
experiment_names = ["Basic Problem", "Basic, Warm Start", "Obstacles", "Obstacles, Warm Start"]
colors = ['orange', 'blue', 'green']

# Plot 3D trajectories
for trial in trials:
    # state and input directories
    states = [dir + s + "_x_" + str(trial) + ".txt" for s in solvers]
    inputs = [dir + s + "_u_" + str(trial) + ".txt" for s in solvers]
    
    # read all data
    states_output = [np.loadtxt(f) for f in states]
    inputs_output = [np.loadtxt(f) for f in inputs]
    
    x1 = states_output[0]
    x2 = states_output[1]
    x3 = states_output[2]
    #x4 = states_output[3]
    u1 = inputs_output[0]
    u2 = inputs_output[1]
    u3 = inputs_output[2]
    #u4 = inputs_output[3]
    t = x1[:, 0]
    
    # make 3d plot of trajectories
    fig = plt.figure(figsize=(14.0, 8.0))
    ax = fig.gca(projection='3d')
    for ii in np.arange(7):
        plt.subplot(2, 4, ii+1)
        plt.title("State " + str(ii+1))
        plt.plot(t, x1[:, ii+1], colors[0], linestyle='-', label = solver_names[0])
        plt.plot(t, x2[:, ii+1], colors[1], linestyle='-', label = solver_names[1])
        plt.plot(t, x3[:, ii+1], colors[2], linestyle='-', label = solver_names[2])
        #plt.plot(t, x4[:, ii+1], colors[3], linestyle='-', label = solver_names[3])
    plt.suptitle("States for Kuka Arm Solution")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # make 3d plot of trajectories
    fig = plt.figure(figsize=(14.0, 8.0))
    ax = fig.gca(projection='3d')
    for ii in np.arange(7):
        plt.subplot(2, 4, ii+1)
        plt.title("State " + str(ii+8))
        plt.plot(t, x1[:, ii+8], colors[0], linestyle='-', label = solver_names[0])
        plt.plot(t, x2[:, ii+8], colors[1], linestyle='-', label = solver_names[1])
        plt.plot(t, x3[:, ii+8], colors[2], linestyle='-', label = solver_names[2])
        #plt.plot(t, x4[:, ii+8], colors[3], linestyle='-', label = solver_names[3])
    plt.suptitle("Velocities for Kuka Arm Solution")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig = plt.figure(figsize=(14.0, 8.0))
    ax = fig.gca(projection='3d')
    for ii in np.arange(7):
        plt.subplot(2, 4, ii+1)
        plt.title("Input " + str(ii+1))
        plt.plot(t, u1[:, ii+1], colors[0], linestyle='-', label = solver_names[0])
        plt.plot(t, u2[:, ii+1], colors[1], linestyle='-', label = solver_names[1])
        plt.plot(t, u3[:, ii+1], colors[2], linestyle='-', label = solver_names[2])
        #plt.plot(t, u4[:, ii+1], colors[3], linestyle='-', label = solver_names[3])
    plt.suptitle("Inputs for Kuka Arm Solution")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.show()
#plt.savefig(dir + "kuka_states_with_opt.eps")


## Plot trajectory information comparison
headers = [dir + s + "_header_" + str(0) + ".txt" for s in solvers]
y_pos = np.arange(len(solvers))
    
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

plt.figure()

plt.subplot(len(trials), 4, trial * 4 + 1)
plt.bar(y_pos, times, align='center', alpha=0.5, color = colors)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
if trial == (len(trials) - 1):
    plt.xticks(y_pos, solver_names, rotation='vertical')
if trial == 0:
    plt.title('Runtime (sec)')
plt.ylabel(experiment_names[trial])
#plt.ylim(0, 3)

plt.subplot(len(trials), 4, trial * 4 + 2)
plt.bar(y_pos, objectives, align='center', alpha=0.5, color = colors)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
if trial == (len(trials) - 1):
    plt.xticks(y_pos, solver_names, rotation='vertical')
if trial == 0:
    plt.title('Objective Value')
#plt.ylim(0, 1.8e2)

plt.subplot(len(trials), 4, trial * 4 + 3)
plt.bar(y_pos, feasinfnorms, align='center', alpha=0.5, color = colors)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
if trial == (len(trials) - 1):
    plt.xticks(y_pos, solver_names, rotation='vertical')
if trial == 0:
    plt.title('Feasibility Violation')

plt.subplot(len(trials), 4, trial * 4 + 4)
plt.bar(y_pos, constraintinfnorms, align='center', alpha=0.5, color = colors)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
if trial == (len(trials) - 1):
    plt.xticks(y_pos, solver_names, rotation='vertical')
if trial == 0:
    plt.title('Constraint Violation')

plt.tight_layout()
plt.show()
