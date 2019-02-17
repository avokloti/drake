import numpy as np
from numpy import random
from matplotlib import pyplot as plt
from matplotlib import animation

# read in standard weighted version data
x1 = np.loadtxt("/Users/ira/Documents/drake/examples/quadrotor/output/testing/weighted_x_0.txt");
x2 = np.loadtxt("/Users/ira/Documents/drake/examples/quadrotor/output/testing/al_x_0.txt");

t = x1[:, 0]
x1 = x1[:, 1:]
x2 = x2[:, 1:]

u1 = np.loadtxt("/Users/ira/Documents/drake/examples/quadrotor/output/testing/weighted_u_0.txt");
u2 = np.loadtxt("/Users/ira/Documents/drake/examples/quadrotor/output/testing/al_u_0.txt");

u1 = u1[:, 1:]
u2 = u2[:, 1:]

# plot and compare final solution for each solver
handles = []
labels = ['Weighted', 'Weighted AL']
fig = plt.figure()
for i in np.arange(12):
    plt.subplot(3, 4, i+1)
    l1, = plt.plot(t, x1[:, i], 'r-')
    l2, = plt.plot(t, x2[:, i], 'b-')
    if (not handles):
        handles = [l1, l2]
    #plt.xlabel("Time (sec)")
#plt.ylabel("State value")

fig.legend(handles, labels)
#fig.legend([l1, l2], ['Weighted', 'Weighted AL'], loc = 'center')
plt.suptitle("Comparison of State Trajectories Across Solvers")
plt.show()


# set up figure for trajectories
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    line.set_data(t, x1[:,i])
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=10, interval=20, blit=True)

plt.show()
