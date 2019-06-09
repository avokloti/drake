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

def processData(entry_index):
    data = [entry[entry_index] for entry in header_output]
    data = np.reshape(data, (num_solvers, total_trials))
    return data

def processDataMeans(entry_index):
    data = [entry[entry_index] for entry in header_output]
    data = np.reshape(data, (num_solvers, total_trials))
    data_means = np.zeros((num_solvers, num_opt_tols, num_feas_tols))
    data_stds = np.zeros((num_solvers, num_opt_tols, num_feas_tols))
    for s in np.arange(num_solvers):
        dd = np.reshape(data[s], (num_trials, num_opt_tols, num_feas_tols))
        succ = np.reshape(successes[s], (num_trials, num_opt_tols, num_feas_tols))
        for ot in np.arange(num_opt_tols):
            for ft in np.arange(num_feas_tols):
                success_mask = succ[:, ot, ft]
                data_means[s, ot, ft] = np.mean(dd[success_mask, ot, ft])
                data_stds[s, ot, ft] = np.std(dd[success_mask, ot, ft])
    return data, data_means, data_stds


dir = "/Users/ira/Documents/drake/examples/quadrotor/output/snopt_ipopt_obstacles_lower_rhos/"
#dir = "/Users/ira/Documents/drake/examples/quadrotor/output/snopt_ipopt_more2/"

# solvers / colors
solvers = ["admm", "snopt", "ipopt"]
colors = ['maroon', 'blue', 'green']
#solvers = ["snopt", "ipopt"]
#colors = ['blue', 'green']

num_solvers = len(solvers)
num_feas_tols = 6
num_opt_tols = 3
num_trials = 10
#num_feas_tols = 6
#num_opt_tols = 3
#num_trials = 10
total_trials = num_feas_tols * num_opt_tols * num_trials


## ---- PROCESS DATA ---- ##

# read data from headers
headers = [dir + name + "_header_" + str(index) + ".txt" for name in solvers for index in np.arange(total_trials)]
header_output = list(map(readHeaderFile, headers))

# separate data
feas_tols = processData(4)
opt_tols = processData(5)

feas_tols_list = np.flip(np.unique(feas_tols))
opt_tols_list = np.flip(np.unique(opt_tols))

# find successful runs! the means and stds will only be computed over these
results = processData(13)

# find problems resulting in success, iteration limit, divergence
successes = [result == 'SolutionFound' for result in results]
itlim = [result == 'IterationLimit' for result in results]
divs = [result == 'ExceptionThrown' for result in results]

iterations = processData(6)
[times, time_means, time_stds] = processDataMeans(7)
[feas, feas_means, feas_stds] = processDataMeans(9)
[constraints, constraint_means, constraint_stds] = processDataMeans(11)
[objectives, objective_means, objective_stds] = processDataMeans(12)


## ---- REPORT SOME INFORMATION ABOUT SOLVES ---- ##

# how many successes are there for specific tolerance combinations?
print("Fraction of problems solved successfully by each solvers across tolerance grid:")
for s in np.arange(num_solvers):
    print("------------------------")
    print(solvers[s] + ":")
    print(np.sum(np.reshape(successes[s], (num_trials, num_opt_tols, num_feas_tols)), axis=0)/num_trials)

print("------------------------")




## ---- PLOTTING ---- ##

# plot runtime for snopt, ipopt, admm over tolerance values
lines = ["-","--", ":"]

plt.figure()

# plot time with error bars on single plot
plt.subplot(2, 1, 1)
for s in np.arange(num_solvers):
    for ot in np.arange(num_opt_tols):
        plt.errorbar(np.arange(num_feas_tols), time_means[s, ot, :], yerr=time_stds[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])

plt.xticks(np.arange(num_feas_tols), feas_tols_list)
plt.title('Runtimes Curves Across Tolerances')
plt.xlabel('Constraint Tolerance Value')
plt.ylabel('Runtime (sec)')

# plot objectives with error bars
plt.subplot(2, 1, 2)
for s in np.arange(num_solvers):
    for ot in np.arange(num_opt_tols):
        plt.errorbar(np.arange(num_feas_tols), objective_means[s, ot, :], yerr=objective_stds[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])

plt.xticks(np.arange(num_feas_tols), feas_tols_list)
plt.title('Objectives for Solvers Across Tolerances')
plt.xlabel('Constraint Tolerance Value')
plt.ylabel('Objective')

plt.tight_layout()


# plot constraints with error bars on same plot
plt.figure()
plt.subplot(2, 1, 1)
for s in np.arange(num_solvers):
    for ot in np.arange(num_opt_tols):
        plt.errorbar(np.arange(num_feas_tols), constraint_means[s, ot, :], yerr=constraint_stds[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])
    plt.xticks(np.arange(num_feas_tols), feas_tols_list)
    plt.title('Other Constraint Error Across Tolerances')
    plt.xlabel('Other Constraint Tolerance Value')
    plt.ylabel('Error (inf-norm)')

plt.subplot(2, 1, 2)
for s in np.arange(num_solvers):
    for ot in np.arange(num_opt_tols):
        plt.errorbar(np.arange(num_feas_tols), feas_means[s, ot, :], yerr=feas_stds[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])
    plt.xticks(np.arange(num_feas_tols), feas_tols_list)
    plt.title('Dynamics Error Across Tolerances')
    plt.xlabel('Dynamics Tolerance Value')
    plt.ylabel('Error (inf-norm)')

plt.legend(loc='right')

plt.tight_layout()
plt.show()


# plot time with error bars on separate subplots
plt.figure()
for s in np.arange(num_solvers):
    plt.subplot(num_solvers, 1, s+1)
    for ot in np.arange(num_opt_tols):
        plt.errorbar(np.arange(num_feas_tols), time_means[s, ot, :], yerr=time_stds[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])
    plt.xticks(np.arange(num_feas_tols), feas_tols_list)
    plt.title('Runtimes Curves Across Tolerances')
    plt.xlabel('Constraint Tolerance Value')
    plt.ylabel('Runtime (sec)')
    plt.legend()

plt.tight_layout()

# plot constraints with error bars on separate subplots
plt.figure()
for s in np.arange(num_solvers):
    plt.subplot(num_solvers, 2, 2*s+1)
    for ot in np.arange(num_opt_tols):
        plt.errorbar(np.arange(num_feas_tols), constraint_means[s, ot, :], yerr=constraint_stds[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])
    plt.xticks(np.arange(num_feas_tols), feas_tols_list)
    plt.title('Constraint Error Across Tolerances')
    plt.xlabel('Constraint Tolerance Value')
    plt.ylabel('Error (inf-norm)')
    plt.legend()
    plt.subplot(num_solvers, 2, 2*s+2)
    for ot in np.arange(num_opt_tols):
        plt.errorbar(np.arange(num_feas_tols), feas_means[s, ot, :], yerr=feas_stds[s, ot, :], color = colors[s], label = str(solvers[s]) + ': ' + str(opt_tols_list[ot]), ls=lines[ot])
    plt.xticks(np.arange(num_feas_tols), feas_tols_list)
    plt.title('Feasibility Error Across Tolerances')
    plt.xlabel('Feasibility Tolerance Value')
    plt.ylabel('Error (inf-norm)')
    plt.legend()

plt.tight_layout()
plt.show()



# next steps...
# make script to plot trajectories!



# plot grid representation?
# make sure above representation is correct

# why does snopt take so long sometimes??

# finish running version with obstacles

# read snopt document in the meantime?


