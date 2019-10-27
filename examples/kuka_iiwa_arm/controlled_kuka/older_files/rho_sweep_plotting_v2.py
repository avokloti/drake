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
    iterations = int(content[8].strip())
    time = float(content[9].strip())
    feas2norm = float(content[10].strip())
    feasinfnorm = float(content[11].strip())
    constraint2norm = float(content[12].strip())
    constraintinfnorm = float(content[13].strip())
    objective = float(content[14].strip())
    solve_result = content[15].strip()
    return [N, T, x0, xf, rho0, rho1, rho2, iterations, time, feas2norm, feasinfnorm, constraint2norm, constraintinfnorm, objective, solve_result]


## --- Main part of script --- ##

# define the lists of rho's
rho0s = [0.05, 0.1, 0.5, 1, 2, 5, 10, 50]
rho1s = [1, 10, 50, 100, 500, 1000, 2000, 5000]
rho2s = [2000]

num_trials = 10
num_problems = len(rho0s) * len(rho1s) * len(rho2s)

# set solver information
solvers = ["admm"]
dir = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_sweep_obstacles/"
#dir = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_sweep_v3/"
#dir = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/rho_sweep6/";
solver_names = ["ADMM"]
colors = ['maroon']

# read data from headers
headers = [dir + "admm_header_" + str(index) + ".txt" for index in np.arange(num_trials * num_problems)]
header_output = list(map(readHeaderFile, headers))

# separate data
iterations = [entry[7] for entry in header_output]
times = [entry[8] for entry in header_output]
objectives = [entry[13] for entry in header_output]
rho0list = [entry[4] for entry in header_output]
rho1list = [entry[5] for entry in header_output]
rho2list = [entry[6] for entry in header_output]
results = [entry[14] for entry in header_output]

# find problems resulting in success, iteration limit, divergence
successes = [result == 'SolutionFound' for result in results]
itlim = [result == 'IterationLimit' for result in results]
divs = [result == 'ExceptionThrown' for result in results]

# reshaped form
successes = np.reshape(successes, (num_problems, num_trials))
iterations = np.reshape(iterations, (num_problems, num_trials))
objectives = np.reshape(objectives, (num_problems, num_trials))
times = np.reshape(times, (num_problems, num_trials))
rho0list = np.reshape(rho0list, (num_problems, num_trials))
print(np.mean(rho0list, axis=1))

# make arrays to hold sd and means
iterations_mean = np.zeros(num_problems)
objectives_mean = np.zeros(num_problems)
times_mean = np.zeros(num_problems)
iterations_sd = np.zeros(num_problems)
objectives_sd = np.zeros(num_problems)
times_sd = np.zeros(num_problems)

# calculate all sd and means
for i in np.arange(num_problems):
    iterations_mean[i] = np.mean(iterations[i, successes[i,:]])
    objectives_mean[i] = np.mean(objectives[i, successes[i,:]])
    times_mean[i] = np.mean(times[i, successes[i,:]])
    iterations_sd[i] = np.std(iterations[i, successes[i,:]])
    objectives_sd[i] = np.std(objectives[i, successes[i,:]])
    times_sd[i] = np.std(times[i, successes[i,:]])

# find number of successful runs (out of 10) for each rho1/rho2 combination
successes_sum = np.sum(successes, axis=1)


## -------------------- PLOTTING -------------------- ##

plt.figure(figsize=(12, 3.2))

plt.subplot(1, 3, 1)
ax = plt.gca()
im = plt.imshow(np.transpose(np.reshape(successes_sum, (len(rho0s), len(rho1s)))), norm=Normalize(vmin=0, vmax=10), cmap='viridis')
plt.colorbar(im)
plt.title('Success rate')
plt.xlabel('Rho 0s')
plt.ylabel('Rho 1s')
plt.xticks(np.arange(len(rho0s)), rho0s)
plt.yticks(np.arange(len(rho1s)), rho1s)

plt.subplot(1, 3, 2)
im = plt.imshow(np.transpose(np.reshape(iterations_mean, (len(rho0s), len(rho1s)))), norm=Normalize(vmin=min(iterations_mean), vmax=max(iterations_mean)), cmap='viridis', interpolation='nearest')
plt.colorbar(im)
plt.title('Mean Iterations')
plt.xlabel('Rho 0s')
plt.ylabel('Rho 1s')
plt.xticks(np.arange(len(rho0s)), rho0s)
plt.yticks(np.arange(len(rho1s)), rho1s)

plt.subplot(1, 3, 3)
im = plt.imshow(np.transpose(np.reshape(objectives_mean, (len(rho0s), len(rho1s)))), norm=Normalize(vmin=min(objectives_mean), vmax=max(objectives_mean)), cmap='viridis', interpolation='nearest')
plt.colorbar(im)
plt.title('Mean Objectives')
plt.xlabel('Rho 0s')
plt.ylabel('Rho 1s')
plt.xticks(np.arange(len(rho0s)), rho0s)
plt.yticks(np.arange(len(rho1s)), rho1s)

plt.tight_layout()
#plt.show()
plt.savefig('quad_rho_sweep_obs.png')


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
