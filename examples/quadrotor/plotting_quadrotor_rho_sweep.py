import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# function to read all info from headers
def readHeaderFile(filename):
    with open(filename) as f:
        content = f.readlines()
    N = int(content[1].strip())
    T = float(content[2].strip())
    x0 = np.array(content[3].strip().split('\t')).astype(np.float)
    xf = np.array(content[4].strip().split('\t')).astype(np.float)
    rho0 = float(content[5].strip())
    rho1 = float(content[6].strip())
    rho2 = float(content[7].strip())
    feas_tol = float(content[8].strip())
    opt_tol = float(content[9].strip())
    iterations = int(content[10].strip())
    time = float(content[11].strip())
    feas2norm = float(content[12].strip())
    feasinfnorm = float(content[13].strip())
    constraint2norm = float(content[14].strip())
    constraintinfnorm = float(content[15].strip())
    objective = float(content[16].strip())
    solve_result = content[17].strip()
    return [N, T, x0, xf, rho0, rho1, rho2, feas_tol, opt_tol, iterations, time, feas2norm, feasinfnorm, constraint2norm, constraintinfnorm, objective, solve_result]

# function to check all objectives
def checkObjectives(Q, Qf, R):
    print('---------------------')
    error_count = 0
    for index in np.arange(num_trials * num_problems):
        # load input file, calculate u' R u
        inputs = np.loadtxt(dir + solvers[0] + "_u_" + str(index) + ".txt")
        sum_of_squares = np.sum(R[0, 0] * inputs[:, 1]**2 + R[1, 1] * inputs[:, 2]**2 + R[2, 2] * inputs[:, 3]**2 + R[3, 3] * inputs[:, 4]**2)
        diff = np.abs(objectives[index] - sum_of_squares)
        if (diff > 1e-5):
            print(str(index) + ": error in objective calculation! Difference is " + str(diff))
            error_count = error_count + 1
        if np.isnan(objectives[index]):
            print(str(index) + ": objective is NAN. Replacing with SOS.")
            objectives[index] = sum_of_squares
    if (error_count == 0):
        print("All objectives matched.")
    print('---------------------')


## ----- notes -------
# in unconstrained problem, objectives[2, 4, 6] = nan (corresponds to problem 118)
# why is objective occasionally NaN? check if calculated objective values match with manual calculation of objective values.
# check if "success" filter is being applied correctly -- why is iteration number so high if it satisfied constraints? is there an optimality termination condition issue?
# looks okay... takes a very long time to get a feasible solution
# next steps...
# put all old files in an old files folder, clean out workspace in file system a bit
# go over quadrotor code, make sure it matches what I want
# run comparison against SNOPT and IPOPT
# run plotting
# based on this, find parameters for SNOPT that best match
# why does SNOPT take forever with the printing constraint? fix it so that it doesn't get stuck in it?


## --- Main part of script --- ##

# define the lists of rho's
rho0s = [0.001, 0.01, 0.1, 1, 10, 100]
rho1s = [0.1, 1, 10, 100, 1000, 5000, 10000]
rho2s = rho1s

num_trials = 10
num_problems = len(rho0s) * len(rho1s)

# set solver information
solvers = ["admm"]
solver_names = ["ADMM"]
colors = ['maroon']

dir = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_sweep/"

# read data from headers
headers = [dir + "admm_header_" + str(index) + ".txt" for index in np.arange(num_trials * num_problems)]
header_output = list(map(readHeaderFile, headers))

# separate data
iterations = [entry[9] for entry in header_output]
times = [entry[10] for entry in header_output]
objectives = [entry[15] for entry in header_output]
results = [entry[16] for entry in header_output]

checkObjectives(0, 0, 0.001 * np.eye(4))

rho0list = [entry[4] for entry in header_output]
rho1list = [entry[5] for entry in header_output]
rho2list = [entry[6] for entry in header_output]

# find problems resulting in success, iteration limit, divergence
successes = [result == 'SolutionFound' for result in results]
itlim = [result == 'IterationLimit' for result in results]
divs = [result == 'ExceptionThrown' for result in results]

# reshaped form
successes = np.reshape(successes, (num_trials, len(rho0s), len(rho1s)))
iterations = np.reshape(iterations, (num_trials, len(rho0s), len(rho1s)))
objectives = np.reshape(objectives, (num_trials, len(rho0s), len(rho1s)))
times = np.reshape(times, (num_trials, len(rho0s), len(rho1s)))
rho0list = np.reshape(rho0list, (num_trials, len(rho0s), len(rho1s)))
rho1list = np.reshape(rho1list, (num_trials, len(rho0s), len(rho1s)))
print(np.mean(rho0list, axis=0))

# make arrays to hold sd and means
iterations_mean = np.zeros((len(rho0s), len(rho1s)))
objectives_mean = np.zeros((len(rho0s), len(rho1s)))
times_mean = np.zeros((len(rho0s), len(rho1s)))
iterations_sd = np.zeros((len(rho0s), len(rho1s)))
objectives_sd = np.zeros((len(rho0s), len(rho1s)))
times_sd = np.zeros((len(rho0s), len(rho1s)))

# calculate all sd and means
for i in np.arange(len(rho0s)):
    for j in np.arange(len(rho1s)):
        iterations_mean[i, j] = np.mean(iterations[successes[:, i, j], i, j])
        objectives_mean[i, j] = np.mean(objectives[successes[:, i, j], i, j])
        times_mean[i, j] = np.mean(times[successes[:, i, j], i, j])
        iterations_sd[i, j] = np.std(iterations[successes[:, i, j], i, j])
        objectives_sd[i, j] = np.std(objectives[successes[:, i, j], i, j])
        times_sd[i, j] = np.std(times[successes[:, i, j], i, j])

# find number of successful runs (out of 10) for each rho1/rho2 combination
successes_sum = np.sum(successes, axis=0)


## -------------------- PLOTTING -------------------- ##

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
ax = plt.gca()
im = plt.imshow(np.transpose(successes_sum), norm=Normalize(vmin=0, vmax=10), cmap='viridis')
plt.colorbar(im)
plt.title('Success rate')
plt.xlabel('Rho 0s')
plt.ylabel('Rho 1s')
plt.xticks(np.arange(len(rho0s)), rho0s)
plt.yticks(np.arange(len(rho1s)), rho1s)

plt.subplot(1, 3, 2)
im = plt.imshow(np.transpose(iterations_mean), cmap='viridis', interpolation='nearest')
plt.colorbar(im)
plt.title('Mean Iterations')
plt.xlabel('Rho 0s')
plt.ylabel('Rho 1s')
plt.xticks(np.arange(len(rho0s)), rho0s)
plt.yticks(np.arange(len(rho1s)), rho1s)

plt.subplot(1, 3, 3)
im = plt.imshow(np.transpose(objectives_mean), cmap='viridis', interpolation='nearest')
plt.colorbar(im)
plt.title('Mean Objectives')
plt.xlabel('Rho 0s')
plt.ylabel('Rho 1s')
plt.xticks(np.arange(len(rho0s)), rho0s)
plt.yticks(np.arange(len(rho1s)), rho1s)

plt.tight_layout()
plt.show()
#plt.savefig('quad_rho_sweep_obstacles.eps')


"""

def printTableRhos(values, successes, itlims, divs, rho2s, rho3s, caption, label):
    print()
    print("\\begin{table}")
    print("    \\begin{tabular}{l | llllllllll}\\toprule")
    print("         ", end='')
    for rho2 in rho2s:
        print(" & " + str(rho2), end='')
    print("         \\\\")
    print("         \\midrule")
    for i in np.arange(len(rho3s)):
        print("         " + str(rho3s[i]), end='')
        for j in np.arange(len(rho2s)):
            if (successes[j, i]):
                print(" & " + str(values[j, i]), end='')
            if (itlims[j, i]):
                print(" & ITLIM", end='')
            if (divs[j, i]):
                print(" & DIV", end='')
        print(" \\\\")
    print("\\\\ \\bottomrule")
    print("\\end{tabular}")
    print("    \\caption{" + caption + "} \\label{" + label + "}")
    print("\\end{table}")
    print()


for ind in np.arange(len(rho1s)):
    tt = np.reshape(times[(100 * ind):(100 * (ind + 1))], (10, 10))
    ss = np.reshape(successes[(100 * ind):(100 * (ind + 1))], (10, 10))
    it = np.reshape(itlim[(100 * ind):(100 * (ind + 1))], (10, 10))
    dd = np.reshape(divs[(100 * ind):(100 * (ind + 1))], (10, 10))
    printTableRhos(tt, ss, it, dd, rho2s, rho3s, "Runtimes across rho1 and rho2 for rho0 = " + str(rho1s[ind]), "")


for ind in np.arange(len(rho1s)):
    ii = np.reshape(iterations[(100 * ind):(100 * (ind + 1))], (10, 10))
    ss = np.reshape(successes[(100 * ind):(100 * (ind + 1))], (10, 10))
    it = np.reshape(itlim[(100 * ind):(100 * (ind + 1))], (10, 10))
    dd = np.reshape(divs[(100 * ind):(100 * (ind + 1))], (10, 10))
    printTableRhos(ii, ss, it, dd, rho2s, rho3s, "Iterations across rho1 and rho2 for rho0 = " + str(rho1s[ind]), "")


for ind in np.arange(len(rho1s)):
    ii = np.reshape(objectives[(100 * ind):(100 * (ind + 1))], (10, 10))
    ss = np.reshape(successes[(100 * ind):(100 * (ind + 1))], (10, 10))
    it = np.reshape(itlim[(100 * ind):(100 * (ind + 1))], (10, 10))
    dd = np.reshape(divs[(100 * ind):(100 * (ind + 1))], (10, 10))
    printTableRhos(ii, ss, it, dd, rho2s, rho3s, "Objectives values across rho1 and rho2 for rho0 = " + str(rho1s[ind]), "")


    
    # make contour plot of success rate (out of 10)
    plt.figure()
    [xx, yy] = np.meshgrid(np.log(rho2s), np.log(rho1s))
    
    im = plt.subplot(1, 3, 1)
    im_contour = plt.contourf(xx, yy, np.reshape(successes_sum, (len(rho1s), len(rho2s))), vmin=0, vmax=10, cmap='summer')
    plt.colorbar(im_contour)
    plt.title('Success rate')
    plt.xlabel('Rho 1s')
    plt.ylabel('Rho 2s')
    
    im = plt.subplot(1, 3, 2)
    im_contour = plt.contourf(xx, yy, np.reshape(iterations_mean, (len(rho1s), len(rho2s))), vmin=0, vmax=10, cmap='summer')
    plt.colorbar(im_contour)
    plt.title('Iterations')
    plt.xlabel('Rho 1s')
    plt.ylabel('Rho 2s')
    
    im = plt.subplot(1, 3, 3)
    im_contour = plt.contourf(xx, yy, np.reshape(objectives_mean, (len(rho1s), len(rho2s))), vmin=0, vmax=10, cmap='summer')
    plt.colorbar(im_contour)
    plt.title('Objectives')
    plt.xlabel('Rho 1s')
    plt.ylabel('Rho 2s')
    
    plt.tight_layout()
    plt.show()
    
"""
