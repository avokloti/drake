import sys
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
    modified_count = 0
    for index in np.arange(num_trials):
        for s in np.arange(num_solvers):
            # load input file, calculate u' R u
            inputs = np.loadtxt(dir + solvers[s] + "_u_" + str(index) + ".txt")
            sum_of_squares = np.sum(R[0, 0] * inputs[:, 1]**2 + R[1, 1] * inputs[:, 2]**2 + R[2, 2] * inputs[:, 3]**2 + R[3, 3] * inputs[:, 4]**2)
            diff = np.abs(objectives[index, s] - sum_of_squares)
            if (diff > 1e-5):
                print(str(index) + ": error in objective calculation! Difference is " + str(diff))
                error_count = error_count + 1
            if np.isnan(objectives[index, s]):
                print(str(index) + ": objective is NAN. Replacing with SOS.")
                objectives[index, s] = sum_of_squares
                modified_count = modified_count + 1
    if (error_count == 0):
        print("All objectives matched.")
    if (modified_count == 0):
        print("No objectives modified.")
    else:
        print(str(modified_count) + " modified objectives.")
    print('---------------------')

def round_to_n(x, n):
    if np.isnan(x):
        return float('nan')
    elif (x == 0):
        return 0
    else:
        return round(x, -int(np.floor(np.log10(np.abs(x)))) + n - 1)



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

# total trials
num_trials = 10
num_solvers = 3

# set solver information
solvers = ["admm", "snopt", "ipopt"]
solver_names = ["ADMM", "SNOPT", "IPOPT"]
colors = ['maroon', 'blue', 'green']

folder = str(sys.argv[1])
dir = "/Users/ira/Documents/drake/examples/quadrotor/output" + folder

# read data from headers
headers = [dir + s + "_header_" + str(index) + ".txt" for index in np.arange(num_trials) for s in solvers]
header_output = list(map(readHeaderFile, headers))

# separate data
iterations = [entry[9] for entry in header_output]
times = [entry[10] for entry in header_output]
feasinfnorms = [entry[12] for entry in header_output]
constraintinfnorms = [entry[14] for entry in header_output]
objectives = [entry[15] for entry in header_output]
results = [entry[16] for entry in header_output]

# find problems resulting in success, iteration limit, divergence
successes = [result == 'SolutionFound' for result in results]
itlim = [result == 'IterationLimit' for result in results]
divs = [result == 'ExceptionThrown' for result in results]

# reshaped form
successes = np.reshape(successes, (num_trials, num_solvers))
iterations = np.reshape(iterations, (num_trials, num_solvers))
consinfnorms = np.reshape(constraintinfnorms, (num_trials, num_solvers))
feasinfnorms = np.reshape(feasinfnorms, (num_trials, num_solvers))
objectives = np.reshape(objectives, (num_trials, num_solvers))
times = np.reshape(times, (num_trials, num_solvers))

checkObjectives(0, 0, 0.001 * np.eye(4))

# make arrays to hold sd and means
objectives_mean = np.zeros((num_solvers))
times_mean = np.zeros((num_solvers))
consinfnorms_mean = np.zeros((num_solvers))
feasinfnorms_mean = np.zeros((num_solvers))
infnorms_mean = np.zeros((num_solvers))
objectives_sd = np.zeros((num_solvers))
times_sd = np.zeros((num_solvers))
consinfnorms_sd = np.zeros((num_solvers))
feasinfnorms_sd = np.zeros((num_solvers))
infnorms_sd = np.zeros((num_solvers))

# calculate all sd and means
for i in np.arange(num_solvers):
    objectives_mean[i] = np.mean(objectives[successes[:, i], i])
    times_mean[i] = np.mean(times[successes[:, i], i])
    consinfnorms_mean[i] = np.mean(consinfnorms[successes[:, i], i])
    feasinfnorms_mean[i] = np.mean(feasinfnorms[successes[:, i], i])
    infnorms_mean[i] = np.mean(np.maximum(consinfnorms[successes[:, i], i], feasinfnorms[successes[:, i], i]))
    objectives_sd[i] = np.std(objectives[successes[:, i], i])
    times_sd[i] = np.std(times[successes[:, i], i])
    consinfnorms_sd[i] = np.std(consinfnorms[successes[:, i], i])
    feasinfnorms_sd[i] = np.std(feasinfnorms[successes[:, i], i])
    infnorms_sd[i] = np.std(np.maximum(consinfnorms[successes[:, i], i], feasinfnorms[successes[:, i], i]))

# find number of successful runs (out of 10) for each rho1/rho2 combination
successes_sum = np.sum(successes, axis=0)

print(successes)

print("\nNumber of successful solves:")
print("ADMM: " + str(successes_sum[0]/num_trials))
print("SNOPT: " + str(successes_sum[1]/num_trials))
print("IPOPT: " + str(successes_sum[2]/num_trials))

round_n = 3

for i in np.arange(3):
    print(" & " + solver_names[i] + " & " + str(successes_sum[i]/num_trials) + " & " + str(round_to_n(times_mean[i], round_n)) + " $\pm$ " + str(round_to_n(times_sd[i], round_n)) + " & " + str(round_to_n(objectives_mean[i], round_n)) + " $\pm$ " + str(round_to_n(objectives_sd[i], round_n)) + " & " + str(round_to_n(infnorms_mean[i], round_n)) + " $\pm$ " + str(round_to_n(infnorms_sd[i], round_n)) + "\\\\")

print()



## -------------------- PLOTTING -------------------- ##

plt.figure()

plt.subplot(2, 2, 1)
ax = plt.gca()
plt.boxplot(objectives)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
plt.title("Objective Value Across Solvers")
#plt.ylabel("objective value")
#plt.ylim([0, 1000])

plt.subplot(2, 2, 2)
ax = plt.gca()
plt.boxplot(times)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
ax.ticklabel_format(style = 'sci', axis='y')
plt.title("Runtime Across Solvers (seconds)")
#plt.ylabel("seconds")

plt.subplot(2, 2, 3)
ax = plt.gca()
plt.boxplot(feasinfnorms)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
plt.title("Dynamics Violation Across Solvers")
plt.ylabel("infinity-norm error")

plt.subplot(2, 2, 4)
ax = plt.gca()
plt.boxplot(consinfnorms)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
plt.title("Obstacle/Bound Violation Across Solvers")
plt.ylabel("infinity-norm error")

plt.tight_layout()


plt.figure()

plt.subplot(2, 2, 1)
ax = plt.gca()
plt.bar([1, 2, 3], objectives_mean, yerr=objectives_sd)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
plt.title("Objective Value Across Solvers")
#plt.ylabel("objective value")
#plt.ylim([0, 1000])

plt.subplot(2, 2, 2)
ax = plt.gca()
plt.bar([1, 2, 3], times_mean, yerr=times_sd)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
ax.ticklabel_format(style = 'sci', axis='y')
plt.title("Runtime Across Solvers (seconds)")
#plt.ylabel("seconds")

plt.subplot(2, 2, 3)
ax = plt.gca()
plt.bar([1, 2, 3], feasinfnorms_mean, yerr=feasinfnorms_sd)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
plt.title("Dynamics Violation Across Solvers")
plt.ylabel("infinity-norm error")

plt.subplot(2, 2, 4)
ax = plt.gca()
plt.bar([1, 2, 3], consinfnorms_mean, yerr=consinfnorms_sd)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ADMM", "SNOPT", "IPOPT"])
plt.title("Obstacle/Bound Violation Across Solvers")
plt.ylabel("infinity-norm error")

plt.tight_layout()
plt.show()
