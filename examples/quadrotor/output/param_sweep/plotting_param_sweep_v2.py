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
    solve_result = content[12].strip()
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

tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
knot_points = [10, 20, 30, 40]

solvers = ["admm_pen", "admm_al_ineq", "snopt", "ipopt"]
dir = "/Users/ira/Documents/drake/examples/quadrotor/output/param_sweep_v2/"
solver_names = ["ADMM Penalty", "ADMM AuLa Ineq", "SNOPT", "IPOPT"]
colors = ['maroon', 'orange', 'blue', 'green']
trials = np.arange(10);

# for each number of knot points
all_times = np.zeros((len(solvers), len(tolerances), len(knot_points), len(trials)))
all_feas_errors = np.zeros((len(solvers), len(tolerances), len(knot_points), len(trials)))
all_constraint_errors = np.zeros((len(solvers), len(tolerances), len(knot_points), len(trials)))
all_objs = np.zeros((len(solvers), len(tolerances), len(knot_points), len(trials)))
all_solve_results = np.zeros((len(solvers), len(tolerances), len(knot_points), len(trials)))

index = 0
for j in np.arange(len(tolerances)):
    for i in np.arange(len(knot_points)):
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
            
            all_times[:, j, i, trial] = times
            all_feas_errors[:, j, i, trial] = feasinfnorms
            all_constraint_errors[:, j, i, trial] = constraintinfnorms
            all_objs[:, j, i, trial] = objectives
            all_solve_results[:, j, i, trial] = [(solve_result == "SolutionFound") for solve_result in solve_results]
            all_solve_results[:, j, i, trial] = [(solve_result == "SolutionFound") for solve_result in solve_results]
            
            index = index + 1


def round_to_n(x, n):
    if np.isnan(x):
        return float('nan')
    elif (x == 0):
        return 0
    else:
        return round(x, -int(np.floor(np.log10(np.abs(x)))) + n - 1)


def printTableValidSolves():
    print()
    print("\\begin{table}")
    print("    \\begin{tabular}{llllll}\\toprule")
    print("        &&\\textbf{" + solver_names[0] + "} & \\textbf{" + solver_names[1] + "} & \\textbf{" + solver_names[2] + "} & \\textbf{" + solver_names[3] + "} \\\\ \\midrule")
    for i in np.arange(len(knot_points)):
        print(str(knot_points[i]), end='')
        for j in np.arange(len(tolerances)):
            print(" & " + str(tolerances[j]), end='')
            for s in np.arange(len(solvers)):
                print(" & " + str(sum(all_solve_results[s, j, i, :])/10.0), end='')
            print("\\\\")
    print("\\\\ \\bottomrule")
    print("\\end{tabular}")
    print("    \\caption{Proportion of Successful Solves} \\label{quad_solves_param_sweep}")
    print("\\end{table}")
    print()

def printTable(all_data, caption, label):
    print()
    print("\\begin{table}")
    print("    \\begin{tabular}{llllll}\\toprule")
    print("        &&\\textbf{" + solver_names[0] + "} & \\textbf{" + solver_names[1] + "} & \\textbf{" + solver_names[2] + "} & \\textbf{" + solver_names[3] + "} \\\\ \\midrule")
    for i in np.arange(len(knot_points)):
        print(str(knot_points[i]), end='')
        for j in np.arange(len(tolerances)):
            print(" & " + str(tolerances[j]), end='')
            for s in np.arange(len(solvers)):
                valid = all_solve_results[s, j, i, :]
                print(" & " + str(round_to_n(np.mean(all_data[s, j, i, np.nonzero(valid)]), 3)) + "$\pm$" + str(round_to_n(np.std(all_data[s, j, i, np.nonzero(valid)]), 3)), end='')
            print("\\\\")
    print("\\\\ \\bottomrule")
    print("\\end{tabular}")
    print("    \\caption{" + caption + "} \\label{" + label + "}")
    print("\\end{table}")
    print()

# make bar plot of solve information
print(sum(all_solve_results[0, :, :, :].flatten()))
print(sum(all_solve_results[1, :, :, :].flatten()))
print(sum(all_solve_results[2, :, :, :].flatten()))
print(sum(all_solve_results[3, :, :, :].flatten()))

# print all tables
printTableValidSolves()
printTable(all_times, "Runtime across Parameter Sweep", "quad_time_param_sweep")
printTable(all_objs, "Objective Value across Parameter Sweep", "quad_objs_param_sweep")
printTable(all_feas_errors, "Dynamic Feasibility Error across Parameter Sweep", "quad_feas_error_param_sweep")
printTable(all_constraint_errors, "Constraint Error across Parameter Sweep", "quad_constraint_error_param_sweep")

