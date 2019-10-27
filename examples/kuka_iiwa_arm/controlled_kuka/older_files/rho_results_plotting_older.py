import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

N = 20
num_states = 12
num_inputs = 4

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

"""
def readStateFile(filename):
    with open(filename) as f:
        content = f.readlines()
    N = len(content)
    x = np.zeros((N, 12))
    for i in np.arange(N):
        x[i,:] = np.array(content[i].strip().split()).astype(np.float)
    states = np.reshape(x[:, 1:num_states], (1, N * num_states))
"""
def readStateFile(filename):
    with open(filename) as f:
        content = f.read()
    states = np.array(content.strip().split()).astype(np.float)
    return states

def readInputFile(filename):
    with open(filename) as f:
        content = f.read()
    inputs = np.array(content.strip().split()).astype(np.float)
    return inputs




## --- Main part of script --- ##

# next steps...
# snopt is very sensitive to change in tolerance
# map out runtime and ability to solve in some way?
#
# to test whether snopt truly fails or not:
# try running collection of experiments...
# admm is run normally, snopt and ipopt are initialized to ADMM solution and vice versa
# snopt is run normally, admm and ipopt initialized to snopt solution
# ipopt is run normally, admm and snopt initialized to snopt solution
# after running, plot all as before?
#
# to test sensitivity to tolerance:
# make plots...
# make grid -- admm, snopt, ipopt times, also times for when they are initialized to each other's solutions
# make grid of how similar output trajectories are
#
# also need to do:
# finish kuka arm with obstacles code
# run!

num_solvers = 9
num_trials = 20

# set solver information
#dir = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_results_1e4/"
#dir = "/Users/ira/Documents/drake/examples/quadrotor/output/rho_results_obstacles_noautodiff/"
dir = "/Users/ira/Documents/drake/examples/quadrotor/output/warm_start/"
#dir = "/Users/ira/Documents/drake/examples/kuka_iiwa_arm/controlled_kuka/output/rho_results

# solvers / colors
solvers = ["admm", "snopt", "ipopt", "admm_ws_s", "admm_ws_i", "snopt_ws_a", "snopt_ws_i", "ipopt_ws_a", "ipopt_ws_s"]
latex_solvers = ["admm", "snopt", "ipopt", "admm(snopt)", "admm(ipopt)", "snopt(admm)", "snopt(ipopt)", "ipopt(admm)", "ipopt(snopt)"]
solver_names = ["ADMM", "SNOPT", "IPOPT", "ADMM (ws with snopt)", "ADMM (ws with ipopt)", "SNOPT (ws with admm)", "SNOPT (ws with ipopt)", "IPOPT (ws with admm)", "IPOPT (ws with snopt)"]
colors = ['maroon', 'blue', 'green']


# --- compare closeness of trajectories ---

# read in state and input files
x_traj_names = [dir + name + "_x_" + str(index) + ".txt" for name in solvers for index in np.arange(num_trials)]
u_traj_names = [dir + name + "_u_" + str(index) + ".txt" for name in solvers for index in np.arange(num_trials)]
x_output = list(map(readStateFile, x_traj_names))
u_output = list(map(readInputFile, u_traj_names))


def calculateTrajDiffs(trial):
    indeces = trial + num_trials * np.arange(len(solvers))
    traj_diffs = np.zeros((len(solvers), len(solvers)))
    for i in np.arange(len(solvers)):
        for j in np.arange(len(solvers)):
            yi = np.concatenate((x_output[indeces[i]], u_output[indeces[i]]))
            yj = np.concatenate((x_output[indeces[j]], u_output[indeces[j]]))
            traj_diffs[i, j] = np.linalg.norm(yi - yj)
    return traj_diffs



def printTable(traj_diffs):
    matches = [(0, 5), (0, 7), (1, 3), (1, 8), (2, 4), (2, 6), (3, 9), (4, 6), (5, 7)]
    print()
    print("\\begin{table}")
    print("    \\centering")
    print("    \\resizebox{\\textwidth}{!}{")
    print("    \\begin{tabular}{l | lllllllll}\\toprule")
    print("         ", end='')
    for i in np.arange(len(solvers)):
        print(" & " + latex_solvers[i], end='')
    print("\\\\")
    print("         \\hline")
    for i in np.arange(len(solvers)):
        print("         ", end='')
        print(latex_solvers[i], end='')
        for j in list(range(0, i)):
            print("&", end='')
        for j in list(range(i, len(solvers))):
            if (i == j):
                print(" & \cellcolor{red!25} " + str(np.around(traj_diffs[i, j], 2)), end='')
            elif (i, j) in matches or (j, i) in matches:
                print(" & \cellcolor{blue!25} " + str(np.around(traj_diffs[i, j], 2)), end='')
            else:
                print(" & " + str(np.around(traj_diffs[i, j], 2)), end='')
        print("\\\\")
    print("\\\\ \\bottomrule")
    print("\\end{tabular}}")
    #print("    \\caption{" + caption + "} \\label{" + label + "}")
    print("\\end{table}")
    print()


# calculate all trajectory differences
all_traj_diffs = [calculateTrajDiffs(trial) for trial in np.arange(num_trials)]

# check how often alg A converges when initialized with algB
def printComparison():
    converges_from_ws = np.zeros((3, 3))
    for i in np.arange(num_trials):
        converges_from_ws[0, 1] = converges_from_ws[0, 1] + np.int(all_traj_diffs[i][1, 3] < 5) #admm to snopt
        converges_from_ws[0, 2] = converges_from_ws[0, 2] + np.int(all_traj_diffs[i][2, 4] < 5) #admm to ipopt
        converges_from_ws[1, 0] = converges_from_ws[1, 0] + np.int(all_traj_diffs[i][0, 5] < 5) #snopt to admm
        converges_from_ws[1, 2] = converges_from_ws[1, 2] + np.int(all_traj_diffs[i][2, 6] < 5) #snopt to ipopt
        converges_from_ws[2, 0] = converges_from_ws[2, 0] + np.int(all_traj_diffs[i][0, 7] < 5) #ipopt to admm
        converges_from_ws[2, 1] = converges_from_ws[2, 1] + np.int(all_traj_diffs[i][0, 8] < 5) #ipopt to snopt
    print("ADMM warm-started with the SNOPT solution converges to it " + str(converges_from_ws[0, 1]) + "/" + str(num_trials) + " times.")
    print("ADMM warm-started with the IPOPT solution converges to it " + str(converges_from_ws[0, 2]) + "/" + str(num_trials) + " times.")
    print("SNOPT warm-started with the ADMM solution converges to it " + str(converges_from_ws[1, 0]) + "/" + str(num_trials) + " times.")
    print("SNOPT warm-started with the IPOPT solution converges to it " + str(converges_from_ws[1, 2]) + "/" + str(num_trials) + " times.")
    print("IPOPT warm-started with the ADMM solution converges to it " + str(converges_from_ws[2, 0]) + "/" + str(num_trials) + " times.")
    print("IPOPT warm-started with the SNOPT solution converges to it " + str(converges_from_ws[2, 1]) + "/" + str(num_trials) + " times.")

                                                                                    
# also plot average runtimes with and without warmstart







# solvers / colors
solvers = ["admm", "snopt", "ipopt"]
solver_names = ["ADMM", "SNOPT", "IPOPT"]
colors = ['maroon', 'blue', 'green']

# read data from headers
headers = [dir + name + "_header_" + str(index) + ".txt" for name in solvers for index in np.arange(num_trials)]
header_output = list(map(readHeaderFile, headers))

# separate data
iterations = [entry[7] for entry in header_output]
times = [entry[8] for entry in header_output]
objectives = [entry[13] for entry in header_output]
feasinfnorm = [entry[10] for entry in header_output]
constraintinfnorm = [entry[12] for entry in header_output]
results = [entry[14] for entry in header_output]

# find problems resulting in success, iteration limit, divergence
successes = [result == 'SolutionFound' for result in results]
itlim = [result == 'IterationLimit' for result in results]
divs = [result == 'ExceptionThrown' for result in results]

# reshaped form
successes = np.reshape(successes, (num_solvers, num_trials))
iterations = np.reshape(iterations, (num_solvers, num_trials))
objectives = np.reshape(objectives, (num_solvers, num_trials))
constraints = np.reshape(constraintinfnorm, (num_solvers, num_trials))
feasibility = np.reshape(feasinfnorm, (num_solvers, num_trials))
times = np.reshape(times, (num_solvers, num_trials))

# make arrays to hold sd and means
iterations_mean = np.zeros(num_solvers)
objectives_mean = np.zeros(num_solvers)
times_mean = np.zeros(num_solvers)
constraints_mean = np.zeros(num_solvers)
feas_mean = np.zeros(num_solvers)

iterations_sd = np.zeros(num_solvers)
objectives_sd = np.zeros(num_solvers)
times_sd = np.zeros(num_solvers)
constraints_sd = np.zeros(num_solvers)
feas_sd = np.zeros(num_solvers)

# calculate all sd and means
for i in np.arange(num_solvers):
    iterations_mean[i] = np.mean(iterations[i, successes[i,:]])
    objectives_mean[i] = np.mean(objectives[i, successes[i,:]])
    times_mean[i] = np.mean(times[i, successes[i,:]])
    constraints_mean[i] = np.mean(constraints[i, successes[i,:]])
    feas_mean[i] = np.mean(feasibility[i, successes[i,:]])
    iterations_sd[i] = np.std(iterations[i, successes[i,:]])
    objectives_sd[i] = np.std(objectives[i, successes[i,:]])
    times_sd[i] = np.std(times[i, successes[i,:]])
    constraints_sd[i] = np.std(constraints[i, successes[i,:]])
    feas_sd[i] = np.std(feasibility[i, successes[i,:]])

# find number of successful runs (out of 10) for each rho1/rho2 combination
successes_sum = np.sum(successes, axis=1)


## -------------------- PLOTTING -------------------- ##

print("Success results across solvers:")
print(successes)

plt.figure()

ax = plt.gca()
ax.bar(np.arange(num_solvers), successes_sum/num_trials)
ax.set_xticks(np.arange(num_solvers))
ax.set_xticklabels(solvers)
plt.title('Percentage of Successful Solves')
plt.xlabel('Solvers')
plt.ylabel('Value')


plt.figure()

plt.subplot(2, 2, 1)
ax = plt.gca()
ax.bar(np.arange(num_solvers), objectives_mean, yerr = objectives_sd)
ax.set_xticks(np.arange(num_solvers))
ax.set_xticklabels(solvers)
plt.title('Objective Value')
plt.xlabel('Solvers')
plt.ylabel('Value')

plt.subplot(2, 2, 2)
ax = plt.gca()
ax.bar(np.arange(num_solvers), times_mean, yerr = times_sd)
ax.set_xticks(np.arange(num_solvers))
ax.set_xticklabels(solvers)
plt.title('Runtime')
plt.xlabel('Solvers')
plt.ylabel('sec')

plt.subplot(2, 2, 3)
ax = plt.gca()
ax.bar(np.arange(num_solvers), constraints_mean, yerr = constraints_sd)
ax.set_xticks(np.arange(num_solvers))
ax.set_xticklabels(solvers)
plt.title('State/Input Bound Satisfaction')
plt.xlabel('Solvers')
plt.ylabel('sec')

plt.subplot(2, 2, 4)
ax = plt.gca()
ax.bar(np.arange(num_solvers), feas_mean, yerr = feas_sd)
ax.set_xticks(np.arange(num_solvers))
ax.set_xticklabels(solvers)
plt.title('Dynamic Feasibility Satisfaction')
plt.xlabel('Solvers')
plt.ylabel('sec')

plt.tight_layout()
plt.show()



"""
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

