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
solvers = ["admm", "ali", "ipopt", "snopt"]
solver_names = ["ADMM", "ALI", "IPOPT", "SNOPT"]
trials = [0]
dir = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/rho_sweep/"
experiment_names = ["Basic Problem", "Basic, Warm Start", "Obstacles", "Obstacles, Warm Start"]
colors = ['maroon', 'orange', 'blue', 'green']

#all parameters
rho0s = [10, 50, 100, 500, 1000]
rho1s = [10, 50, 100, 500, 1000]
rho2s = [10, 50, 100, 500, 1000]

# from current data:
# list all objective values across rho combinations for both solvers
# list runtimes across rho combinations for both solvers
# what do the curves look like?

# does method work with random initialization?

# do quadrotor problem and kuka arm problem scale similarly?
#   -- scale rho0 based on objective cost matrix norm
#   -- scale rho1 and rho2 based on initial matrix norm given a random initialization of trajectory
#   -- this means: initialize randomly (if successful), measure M.norm and G.norm and first iteration, check if reasonable, scale by this value. If norm is less than 1, set it equal to 1 (no scaling)...? If norm is 0... is this ever going to happen?

# after constructing scaling in this way, run parameter sweep on quadrotors (with and without obstacles) and on kuka arm problem (with shaped cost and with simple cost)
# this time, keep rho2 = rho3, and do a 2D sweep

#0-24:
# rho1 = 10, rho2 and rho3 vary
#25-49:
# rho1 = 50, rho2 and rho3 vary

headers_al_ineq = [dir + "admm_header_" + str(n) + ".txt" for n in np.arange(54)]
headers_ali_res = [dir + "ali_header_" + str(n) + ".txt" for n in np.arange(54)]

output_al_ineq = list(map(readHeaderFile, headers_al_ineq))
output_ali_res = list(map(readHeaderFile, headers_ali_res))

times_al_ineq = [entry[5] for entry in output_al_ineq]
objectives_al_ineq = [entry[10] for entry in output_al_ineq]
times_ali_res = [entry[5] for entry in output_ali_res]
objectives_ali_res = [entry[10] for entry in output_ali_res]

# reshape these arrays into correct size

# for admm:
#   -- objective is constant across parameters...
#   -- runtime is better for lowest values of each parameter

# for ali:
#   -- divergence for larger values of rho2 and rho3
#   -- better performance for smaller values as well

# shift everything down...

# calculate scale factor given this, assuming rho=10, and try values around there
# hmmm so 10/(0.01 + 10) ? if |R| about 1, then maybe rho0 = 1000?

# what should scale factor be?
#

# honestly, ADMM seems to work better... maybe experiment with ADMM and residual balancing?

# try scaled ADMM, and see if same results can be gotten
# try shaped cost, see results there
# check quadrotor problem, and find better results for it

# note:
# there will be more difficulties with ADMM with constraints because it looks like constraints aren't even being hit at all
