import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3

def getConstraints(z):
    g = np.asarray([[z[1, 0]**2 - z[0, 0]],
                    [z[0, 0] - (z[1, 0]**2)/4 - 2],
                    [0.7 - (z[0, 0]+1)**2 - z[1, 0]**2]])
    G = np.matrix([[-1, 2 * z[1, 0]],
                   [1, -z[1, 0]/2],
                   [-2 * (z[0, 0]+1), -2 * z[1, 0]]])
    return [g, G]

def objectiveUpdate(z, Q, q, lambda1, rho1):
    matrix = 2 * Q + rho1 * np.identity(2)
    return np.linalg.inv(matrix) @ (rho1 * z - lambda1 - q)

def constraintsUpdate(w, G, h, lambda1, lambda2, rho1, rho2):
    matrix = rho1 * np.identity(2) + rho2 * G.T * G
    return np.linalg.inv(matrix) @ (rho1 * w + lambda1 + G.T * (rho2 * h - lambda2))


rho1 = 4
rho2 = 2

w = np.array([[3], [2]])
z = np.copy(w)

lambda1 = np.asarray([[0], [0]])
lambda2 = np.asarray([[0], [0], [0]])

Q = np.identity(2)
q = np.asarray([[2], [1]])

all_ws1 = []
all_ws2 = []
all_zs1 = []
all_zs2 = []
all_lambda1_norms = []
all_lambda2_norms = []
all_objectives = []
all_constraints = []

n = 50

# next steps
# decrease all
# for real algorithm -- don't want to treat inequalities as equality constraints. For example, this will probably push trajectory towards state and input bounds.

for i in np.arange(n):
    [g, G] = getConstraints(z)
    if (g[0, 0] <= -lambda2[0]/rho2): #(g[0, 0] <= -lambda2[0]/rho2):
        g[0, 0] = 0
        G[0, :] = np.array([0, 0])
        lambda2[0] = lambda2[0]/1.5
    if (g[1, 0] <= -lambda2[1]/rho2): #(g[1, 0] <= -lambda2[1]/rho2):
        g[1, 0] = 0
        G[1, :] = np.array([0, 0])
        lambda2[1] = lambda2[1]/1.5
    if (g[2, 0] <= -lambda2[2]/rho2): #(g[2, 0] <= -lambda2[1]/rho2):
        g[2, 0] = 0
        G[2, :] = np.array([0, 0])
        lambda2[2] = lambda2[2]/1.5
    h = G @ z - g
    w = objectiveUpdate(z, Q, q, lambda1, rho1)
    z = constraintsUpdate(w, G, h, lambda1, lambda2, rho1, rho2)
    lambda1 = lambda1 + rho1 * (w - z)
    lambda2 = lambda2 + rho2 * (G * z - h)
    all_ws1.append(w[0, 0])
    all_ws2.append(w[1, 0])
    all_zs1.append(z[0, 0])
    all_zs2.append(z[1, 0])
    print(lambda2.transpose())
    print(str(i) + " -- objective = " + str(z.T @ Q @ z + q.T @ z) + " -- g = " + str(g[0, 0]**2 + g[1, 0]**2) + ", rho2 = " + str(rho2))
    all_lambda1_norms.append(np.linalg.norm(lambda1, 2))
    all_lambda2_norms.append(np.linalg.norm(lambda2, 2))
    all_objectives.append(np.asscalar(z.T @ Q @ z + q.T @ z))
    all_constraints.append(np.linalg.norm(g, 2))
    rho2 = rho2 * 2

min_x = -3.0
max_x = 3.0
min_y = -3.0
max_y = 3.0

plt.figure()
t = np.linspace(-np.sqrt(8/3), np.sqrt(8/3), 50)
ty = (t**2).tolist() + list(reversed((t**2/4 + 2).tolist()))
tx = t.tolist() + list(reversed(t.tolist()))
ky = 0.7 * np.sin(np.linspace(0, 2 * np.pi, 50))
kx = 0.7 * np.cos(np.linspace(0, 2 * np.pi, 50)) - 1
plt.plot(ty, tx, 'k')
plt.plot(kx, ky, 'k')
cx, cy = np.meshgrid(np.linspace(min_x, max_x, 20), np.linspace(min_y, max_y, 20))
plt.contourf(cx, cy, Q[0, 0] * cx**2 + Q[1, 1] * cy**2 + q[0] * cx + q[1] * cy)
plt.plot(all_ws1, all_ws2, '*-', c='r')
plt.plot(all_zs1, all_zs2, '*-', c='w')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('ADMM Iterations')
#plt.savefig('toy_example_iterations_zero_1.eps')

plt.figure()
plt.subplot(4, 1, 1)
plt.plot(all_lambda1_norms)
plt.title("Lambda1 Norms")
plt.subplot(4, 1, 2)
plt.plot(all_lambda2_norms)
plt.title("Lambda2 Norms")
plt.subplot(4, 1, 3)
plt.plot(all_objectives)
plt.title("Objectives")
plt.subplot(4, 1, 4)
plt.plot(all_constraints)
plt.title("Constraint Norms")
plt.tight_layout()
#plt.savefig('toy_example_norms_zero_1.eps')


fig = plt.figure()
# plot polygon
t = np.linspace(-np.sqrt(8/3), np.sqrt(8/3), 50)
ty = (t**2).tolist() + list(reversed((t**2/4 + 2).tolist()))
tx = t.tolist() + list(reversed(t.tolist()))
ky = 0.7 * np.sin(np.linspace(0, 2 * np.pi, 50))
kx = 0.7 * np.cos(np.linspace(0, 2 * np.pi, 50)) - 1
plt.plot(ty, tx, 'k')
plt.plot(kx, ky, 'k')
cx, cy = np.meshgrid(np.linspace(min_x, max_x, 20), np.linspace(min_y, max_y, 20))
plt.contourf(cx, cy, Q[0, 0] * cx**2 + Q[1, 1] * cy**2 + q[0] * cx + q[1] * cy)

points_z, = plt.plot(all_zs1[0], all_zs2[0], '*-', c='w')
points_w, = plt.plot(all_ws1[0], all_ws2[0], '*-', c='r')

plt.xlim([min_x, max_x])
plt.ylim([min_y, max_y])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('ADMM Iterations')

def animate(i, all_zs1, all_zs2, all_ws1, all_ws2, points_w, points_z):
    ii = int(np.floor(i/2))
    if (i % 2 == 0):
        points_z.set_xdata(all_zs1[0:ii])
        points_z.set_ydata(all_zs2[0:ii])
        return points_z,
    else:
        points_w.set_xdata(all_ws1[0:ii])
        points_w.set_ydata(all_ws2[0:ii])
        return points_w,

ani = animation.FuncAnimation(fig, animate, 2 * n, fargs=(all_zs1, all_zs2, all_ws1, all_ws2, points_w, points_z), interval=100, blit=False)

plt.show()

# next steps:
# draw background
# figure out divergence issue
# experiment with Lagrange multiplier and different updates



"""
# next steps: visualize
def updateFigure(i, data, lines):
    line.set_data(all_zs[i][0, 0], all_zs[i][1, 0])
    return lines

fig = plt.figure()
ax = p3.Axes3D(fig)

data = all_zs
lines = ax.plot(all_zs[0][0, 0], all_zs[0][1, 0])[0]

line_ani = animation.FuncAnimation(fig, updateFigure, n, fargs=(data, lines), interval=100, blit=False)

# Setting the axes properties
ax.set_xlim3d([0.0, 6.0])
ax.set_ylim3d([0.0, 6.0])
ax.set_zlim3d([0.0, 6.0])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_title('Solution Trajectories Over Time')
ax.legend(loc = 'lower right')
plt.show()
"""
