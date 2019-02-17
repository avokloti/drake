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
    solve_result = int(content[12].strip())
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


## --- Main part of script --- ##

tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
knot_points = [10, 20, 30, 40]

solvers = ["admm_pen", "admm_al", "snopt", "ipopt"]
dir = "/Users/ira/Documents/drake/examples/quadrotor/output/param_sweep/"
solver_names = ["ADMM Penalty", "ADMM AuLa", "SNOPT", "IPOPT"]
colors = ['maroon', 'orange', 'blue', 'green']
trials = np.arange(10);

# for each number of knot points
fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()

index = 0
for j in np.arange(len(tolerances)):
    knot_point_time_means = np.zeros((len(knot_points), len(solvers)))
    knot_point_time_sd = np.zeros((len(knot_points), len(solvers)))
    knot_point_feas_means = np.zeros((len(knot_points), len(solvers)))
    knot_point_feas_sd = np.zeros((len(knot_points), len(solvers)))
    knot_point_constraint_means = np.zeros((len(knot_points), len(solvers)))
    knot_point_constraint_sd = np.zeros((len(knot_points), len(solvers)))
    knot_point_obj_means = np.zeros((len(knot_points), len(solvers)))
    knot_point_obj_sd = np.zeros((len(knot_points), len(solvers)))
    
    ax5 = plt5.add_subplot(len(tolerances), 1, j)
    
    for i in np.arange(len(knot_points)):
        all_times = np.zeros((len(trials), len(solvers)))
        all_feas_errors = np.zeros((len(trials), len(solvers)))
        all_constraint_errors = np.zeros((len(trials), len(solvers)))
        all_objs = np.zeros((len(trials), len(solvers)))
        all_solve_results = np.zeros((len(trials), len(solvers)))
        
        for trial in trials:
            headers = [dir + s + "_header_" + str(index) + ".txt" for s in solvers]
            
            # for each solver, read header
            header_output = list(map(readHeaderFile, headers))
            
            # do some checks
            Ns = [entry[0] for entry in header_output]
            Ts = [entry[1] for entry in header_output]
            x0s = [entry[2] for entry in header_output]
            xfs = [entry[3] for entry in header_output]
            tols = [entry[4] for entry in header_output]
            
            if not checkEqual(Ns) or not checkEqual(Ts) or not checkEqualList(x0s) or not checkEqualList(xfs) or not checkEqual(tols):
                print("HEADERS DO NOT MATCH!")
        
            times = [entry[5] for entry in header_output]
            feas2norms = [entry[6] for entry in header_output]
            feasinfnorms = [entry[7] for entry in header_output]
            constraint2norms = [entry[8] for entry in header_output]
            constraintinfnorms = [entry[9] for entry in header_output]
            objectives = [entry[10] for entry in header_output]
            solve_results = [entry[11] for entry in header_output]
            
            all_times[trial, :] = times
            all_feas_errors[trial, :] = feasinfnorms
            all_constraint_errors[trial, :] = constraintinfnorms
            all_objs[trial, :] = objectives
            all_solve_results[trial, :] = solve_results
            
            index = index + 1

        mean_times = np.mean(all_times, axis=0)
        mean_feas_errors = np.mean(all_feas_errors, axis=0)
        mean_constraint_errors = np.mean(all_constraint_errors, axis=0)
        mean_objs = np.mean(all_objs, axis=0)
        
        sd_times = np.std(all_times, axis=0)
        sd_feas_errors = np.std(all_feas_errors, axis=0)
        sd_constraint_errors = np.std(all_constraint_errors, axis=0)
        sd_objs = np.std(all_objs, axis=0)
        
        #if (i == 1) and (j == 2):
        #    print(all_times)
        #    print(all_constraint_errors)
        #    print(all_solve_results)
        #print(np.sum(all_solve_results, axis=0))
        
        valid_times = []
        valid_feas_errors = []
        valid_constraint_errors = []
        valid_objs = []
        
        for ss in np.arange(4):
            valid = all_solve_results[:, ss]
            valid_times.append(all_times[np.nonzero(valid), ss].flatten())
            valid_feas_errors.append(all_feas_errors[np.nonzero(valid), ss].flatten())
            valid_constraint_errors.append(all_constraint_errors[np.nonzero(valid), ss].flatten())
            valid_objs.append(all_objs[np.nonzero(valid), ss].flatten())

        if (i == 2) and (j == 2):
            print("Solve results:")
            print(all_solve_results)
            print("All times:")
            print(all_times)
            print("Valid times:")
            print(valid_times)
        
        ax1 = fig1.add_subplot(len(knot_points), len(tolerances), i * len(tolerances) + j + 1)
        #ax1.boxplot(all_times)
        ax1.boxplot(valid_times)
        if (i == 0):
            ax1.set_title("tol = " + str(tolerances[j]))
        if (j == 0):
            ax1.set_ylabel("N = " + str(knot_points[i]))

        ax2 = fig2.add_subplot(len(knot_points), len(tolerances), i * len(tolerances) + j + 1)
        #ax2.boxplot(all_feas_errors)
        ax2.boxplot(valid_feas_errors)

        ax3 = fig3.add_subplot(len(knot_points), len(tolerances), i * len(tolerances) + j + 1)
        #ax3.boxplot(all_constraint_errors)
        ax3.boxplot(valid_constraint_errors)
        
        ax4 = fig4.add_subplot(len(knot_points), len(tolerances), i * len(tolerances) + j + 1)
        #ax4.boxplot(all_objs)
        ax4.boxplot(valid_objs)
    #plot
    ax5.

fig1.suptitle("Runtime (sec)")
fig2.suptitle("Dynamics Error")
fig3.suptitle("Constraint Error")
fig4.suptitle("Objective Value")

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
plt.show()


# for each fixed tolerance, make runtime plot against different knot points
fig5 = plt.figure()
for j in np.arange(len(tolerances)):
    ax5 = fig5.add_subplot(len(tolerances), 1, j)
    for i in np.arange(len(knot_points)):
        # read all solver data across all trials












# NEED TO DO
# for each solve, check if successful or unsuccessful based on tolerance satisfaction
# keep track of
# find above statistics for correct solution



"""

            index = index + 1
        # read all trials
        # for each solver, find average and sd of runtime per trial
        # for each solver, find average and sd of objective value per trial
        # for each solver, find average and sd of objective value per trial
        # for each solver, find average and sd of feasibility per trial
        # for each solver, find average and sd of constraint satisfaction per trial


plt.subplot(len(trials), 3, trial * 3 + 1)
plt.bar(y_pos, times, align='center', alpha=0.5, color = colors)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
if trial == (len(trials) - 1):
    plt.xticks(y_pos, solver_names, rotation='vertical')
    if trial == 0:
        plt.title('Runtime (sec)')
            plt.ylabel(experiment_names[trial])
            plt.ylim(0, 3)

            
            
            ax1 = fig1.add_subplot(len(knot_points), len(tolerances), i * len(tolerances) + j + 1)
            ax1.bar(np.arange(len(solvers)), mean_times, align='center', alpha=0.5, color = colors)
            ax1.errorbar(np.arange(len(solvers)), mean_times, sd_times)
            if (i == 0):
            ax1.set_title("tol = " + str(tolerances[j]))
            if (j == 0):
            ax1.set_ylabel("N = " + str(knot_points[i]))
            
            ax2 = fig2.add_subplot(len(knot_points), len(tolerances), i * len(tolerances) + j + 1)
            ax2.bar(np.arange(len(solvers)), mean_feas_errors, align='center', alpha=0.5, color = colors)
            ax2.errorbar(np.arange(len(solvers)), mean_feas_errors, sd_feas_errors)
            
            ax3 = fig3.add_subplot(len(knot_points), len(tolerances), i * len(tolerances) + j + 1)
            ax3.bar(np.arange(len(solvers)), mean_constraint_errors, align='center', alpha=0.5, color = colors)
            ax3.errorbar(np.arange(len(solvers)), mean_constraint_errors, sd_constraint_errors)
            
            ax4 = fig4.add_subplot(len(knot_points), len(tolerances), i * len(tolerances) + j + 1)
            ax4.bar(np.arange(len(solvers)), mean_objs, align='center', alpha=0.5, color = colors)
            ax4.errorbar(np.arange(len(solvers)), mean_objs, sd_objs)
"""
