import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

T = 4.0
N = 40
num_states = 14
num_inputs = 7

# function to read all info from headers
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
    objective = float(content[9].strip())
    return [N, T, x0, xf, max_iter, time, error2norm, errorinfnorm, objective]

# function to check equality of all elements in list
def checkEqual(iterator):
    return len(set(iterator)) <= 1

# function to check equality of list of lists
def checkEqualList(listoflists):
    if (len(listoflists) == 1):
        return True
    else:
        return ((listoflists[0] == listoflists[1]).all() and checkEqualList(listoflists[1:]))

# transform 'y' vector into 'x' and 'u' state/input matrices
def y2xu(y):
    x = np.reshape(y[0:(N * num_states)], (N, num_states))
    u = np.reshape(y[(N * num_states):(N * num_states + N * num_inputs)], (N, num_inputs))
    return [x, u]


# directory
dir = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/basic/"
file = dir + "ali_traj_0.txt"
with open(file) as f:
    content = f.readlines()
l = [np.array(line.strip().split()).astype(np.float) for line in content]

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

def update_lines(i):
    t = np.linspace(0, T, N)
    for ii in np.arange(num_inputs):
        x, u = y2xu(l[i])
        all_lines[ii].set_data(t, u[:, ii])
    return all_lines

# Setting the axes properties
plt.title('Solution Trajectories Over Time')
#plt.legend(loc = 'lower right')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, len(l), interval=100, blit=False)

plt.show()

# next, for now...
# animate all inputs and compare against standard ADMM?
# check if the constraint satisfaction issue was resolved with quadrotors, or if ALI still draws away from the constraint boundary

