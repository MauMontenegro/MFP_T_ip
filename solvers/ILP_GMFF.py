import time as tm
from pathlib import Path
import matplotlib.pyplot as plt

import networkx as nx
import numpy
import pulp as pl
from gurobipy import *

from utils.utils import GDN
from utils.utils import generateInstance
from utils.utils import tracing_mem
from utils.utils import tracing_start

BN = 10000  # Big Number for Restrictions
import numpy as np


"""
Moving Firefighter Problem on General Graphs
Linear Programming Formulation
Author: Mauro Alejandro Montenegro Meza & Bruno Rodríguez
"""

class ILP_GMFF():
    def __init__(self,mode,load,path,config):
        self.n_variables = []
        self.n_restrictions = []

        # Variables to plot with average
        self.times = []
        self.total_times = []
        self.total_saved = []
        self.saved = []

        # Control Variables
        self.config = config
        self.mode = mode
        self.load = load
        if self.mode == 'batch':
            self.path = path
        else:
            self.w_path = os.walk(Path.cwd() / 'Instance')
            self.path = path + '/Instance'
    def solve(self):
        # Traverse and save Tree Node Sizes dirs
        size_dirs = []
        for d in next(os.walk((self.path))):
            size_dirs.append(d)
        size_dirs = sorted(size_dirs[1])
        for dir in size_dirs:
            instance_path = self.path / str(dir)
            # Traverse each instance
            inst_dirs = []
            for i in next(os.walk((instance_path))):
                inst_dirs.append(i)
            inst_dirs = sorted(inst_dirs[1])
            # Solve IQP problem for each instance
            for inst in inst_dirs:
                print("\n\nCompute solution for size: {n}, instance: {i}".format(n=dir, i=inst))

                # Load Instance
                instance = generateInstance(self.load, instance_path, str(inst))
                T = instance[0]
                print(T)
                n = instance[1]
                print(n)
                starting_fire = instance[2]
                print(starting_fire)
                T_Ad_Sym = instance[3]
                print(T_Ad_Sym)
                weights = instance[10]
                fighter_pos = [instance[6] , instance[7]]
                leaf_nodes = [
                    node for node in T.nodes() if T.in_degree(node) != 0 and T.out_degree(node) == 0
                ]

                D = len(leaf_nodes)
                B = D
                A = nx.to_numpy_array(T).astype(int)
                Dm = T_Ad_Sym
                time = None
                return_matrices = False

                # --- MODEL-------
                m = Model("ILP_FF")
                m.Params.outputFlag = 1  # 0 - Off  //  1 - On
                m.setParam("MIPGap", self.config['experiment']['mip_gap'])
                m.setParam("Method", self.config['experiment']['method'])
                m.setParam("Presolve", self.config['experiment'][
                    'presolve'])  # -1 - Automatic // 0 - Off // 1 - Conservative // 2 - Aggresive
                m.setParam("NodefileStart", self.config['experiment']['nodefilestart'])
                m.setParam("NodefileDir", self.config['experiment']['nodefiledir'])
                m.setParam("Threads", self.config['experiment']['threads'])

                # ------------------------------INPUT --------------------------------
                I = [starting_fire]

                b0 = []
                d0 = []
                for i in range(n):
                    if i in I:
                        b0.append(1)  # These are the vertices burned at time j=0
                    else:
                        b0.append(0)
                for i in range(n):
                    if i == n - 1:
                        d0.append(1)  # The bulldozer begins at vertex n-1
                    else:
                        d0.append(0)

                # ---------------------------- VARIABLES -----------------------------------------------------------

                b = []
                for i in range(n):
                    temp = []
                    for j in range(B):
                        temp.append(0)
                    b.append(temp)
                for i in range(n):
                    for j in range(B):
                        b[i][j] = m.addVar(vtype=GRB.BINARY, name="b,%s" % str(i + 1) + "," + str(j + 1))

                d = []
                for i in range(n):
                    temp = []
                    for j in range(B):
                        temp.append(0)
                    d.append(temp)
                for i in range(n):
                    for j in range(B):
                        d[i][j] = m.addVar(vtype=GRB.BINARY, name="d,%s" % str(i + 1) + "," + str(j + 1))

                d_prime = []
                for j in range(B):
                    temp_1 = []
                    for i in range(n):
                        temp_2 = []
                        for l in range(D):
                            temp_2.append(0)
                        temp_1.append(temp_2)
                    d_prime.append(temp_1)
                for j in range(B):
                    for i in range(n):
                        for l in range(D):
                            d_prime[j][i][l] = m.addVar(vtype=GRB.BINARY,
                                                        name="d_prime,%s" % str(j + 1) + "," + str(i + 1) + "," + str(
                                                            l + 1))

                p = []
                for j in range(B):
                    temp_1 = []
                    for i in range(n):
                        temp_2 = []
                        for l in range(D):
                            temp_2.append(0)
                        temp_1.append(temp_2)
                    p.append(temp_1)
                for j in range(B):
                    for i in range(n):
                        for l in range(D):
                            p[j][i][l] = m.addVar(vtype=GRB.BINARY,
                                                  name="p,%s" % str(j + 1) + "," + str(i + 1) + "," + str(l + 1))

                t = []
                for j in range(B):
                    t.append(0)
                for j in range(B):
                    t[j] = m.addVar(vtype=GRB.CONTINUOUS, name="t,%s" % str(j + 1))

                y = []
                for j in range(B):
                    y.append(0)
                for j in range(B):
                    y[j] = m.addVar(vtype=GRB.BINARY, name="y,%s" % str(j + 1))

                # ---------------------------- CONSTRAINTS ---------------------------------------------------------

                for i in range(n):  # ---------------------------( 2 )
                    for j in range(B):
                        if j == 0:
                            m.addConstr(b[i][j] >= b0[i])
                        else:
                            m.addConstr(b[i][j] >= b[i][j - 1])

                for i in range(n):  # -------------------------( 3 )
                    for j in range(B):
                        if j == 0:
                            m.addConstr(d[i][j] >= d0[i])
                        else:
                            m.addConstr(d[i][j] >= d[i][j - 1])

                for i in range(n):  # ---------------------------( 4 )
                    for j in range(B):
                        m.addConstr(b[i][j] + d[i][j] <= 1)

                for i in range(n - 1):  # ---------------------------( 5 )
                    for j in range(B):
                        for k in range(n - 1):
                            if j == 0:
                                m.addConstr(b[i][j] + d[i][j] >= b0[k] * A[k, i])
                            else:
                                m.addConstr(b[i][j] + d[i][j] >= b[k][j - 1] * A[k, i])
                        # k == n such as d[n][0] = 1, remains defended for every t.

                for i in range(n):  # ---------------------------( 6 y 7 )
                    d0[i] = 0
                d0[n - 1] = 1

                for i in range(n):  # ---------------------------( 8 )
                    for j in range(B):
                        if j == 0:
                            m.addConstr(d_prime[j][i][0] >= d0[i])
                        else:
                            m.addConstr(d_prime[j][i][0] >= d[i][j - 1])

                for i in range(n):  # ---------------------------( 9 )
                    for j in range(B):
                        m.addConstr(d_prime[j][i][D - 1] == d[i][j])

                for i in range(n):  # ---------------------------( 10 )
                    for j in range(B):
                        for k in range(1, D):
                            m.addConstr(d_prime[j][i][k] >= d_prime[j][i][k - 1])

                for j in range(B):  # ---------------------------( 11 y 12 )
                    for i in range(n):
                        for k in range(D):
                            if k == 0:
                                if j == 0:
                                    m.addConstr(p[j][i][k] >= d_prime[j][i][k] - d0[i])
                                else:
                                    m.addConstr(p[j][i][k] >= d_prime[j][i][k] - d[i][j - 1])
                            else:
                                m.addConstr(p[j][i][k] >= d_prime[j][i][k] - d_prime[j][i][k - 1])

                for j in range(B):  # ---------------------------( 13 )
                    for k in range(D):
                        sum_ = 0
                        for i in range(n):
                            sum_ = sum_ + p[j][i][k]
                        m.addConstr(sum_ == 1)

                for j in range(B):  # ---------------------------( 14, 15 y 16)
                    for k in range(D):
                        sum_ = 0
                        for i in range(n):
                            if k == 0:
                                if j == 0:
                                    sum_ = sum_ + d_prime[j][i][k] - d0[i]
                                else:
                                    sum_ = sum_ + d_prime[j][i][k] - d[i][j - 1]
                            else:
                                sum_ = sum_ + d_prime[j][i][k] - d_prime[j][i][k - 1]
                        for i in range(n):
                            if k == 0:
                                if j == 0:
                                    m.addConstr(p[j][i][k] >= d0[i] * (1 - sum_))
                                else:
                                    m.addConstr(p[j][i][k] >= p[j - 1][i][D - 1] * (1 - sum_))
                            else:
                                m.addConstr(p[j][i][k] >= p[j][i][k - 1] * (1 - sum_))

                for j in range(B):  # ---------------------------( 17 y 18 )
                    sum_1 = 0
                    for l in range(n):
                        sum_1_a = 0
                        for i in range(n):
                            sum_1_a = sum_1_a + p[j][i][0] * Dm[i][l]
                        if j == 0:
                            sum_1 = sum_1 + sum_1_a * d0[l]
                        else:
                            sum_1 = sum_1 + sum_1_a * p[j - 1][l][D - 1]
                    sum_2 = 0
                    for k in range(1, D):
                        sum_2_a = 0
                        for l in range(n):
                            sum_2_b = 0
                            for i in range(n):
                                sum_2_b = sum_2_b + p[j][i][k] * Dm[i][l]
                            sum_2_a = sum_2_a + sum_2_b * p[j][l][k - 1]
                        sum_2 = sum_2 + sum_2_a
                    sum_3 = sum_1 + sum_2
                    if j == 0:
                        m.addConstr(t[j] == sum_3 + 0)
                    else:
                        m.addConstr(t[j] == sum_3 + t[j - 1])

                for j in range(B):  # ---------------------------( 19 )
                    sum_ = 0
                    for i in range(n):
                        for k in range(D):
                            if k == 0:
                                if j == 0:
                                    m.addConstr(y[j] >= d_prime[j][i][k] - d0[i])
                                    sum_ = sum_ + d_prime[j][i][k] - d0[i]
                                else:
                                    m.addConstr(y[j] >= d_prime[j][i][k] - d[i][j - 1])
                                    sum_ = sum_ + d_prime[j][i][k] - d[i][j - 1]
                            else:
                                m.addConstr(y[j] >= d_prime[j][i][k] - d_prime[j][i][k - 1])
                                sum_ = sum_ + d_prime[j][i][k] - d_prime[j][i][k - 1]
                    m.addConstr(y[j] <= sum_)

                for j in range(B):  # ---------------------------( 20 y 21 )
                    # m.addConstr(t[j] >= j  * y[j])
                    # m.addConstr(t[j] <= (j+1) + (1 - y[j]) * M )
                    m.addConstr(t[j] <= j + 1)

                # ---------------------------- OBJECTIVE FUNCTION --------------------------------------------------
                b_transpose = np.array(b).T.tolist()
                m.setObjective(sum(b_transpose[B - 1]),
                               GRB.MINIMIZE)  # -----------------------------------------------(1)
                # ---------------------------- OPTIMIZATION -------------------------------------------------------

                m.optimize()
                runtime = m.Runtime
                self.saved.append(m.ObjVal)
                self.times.append(runtime)
                not_interrupted = True
                if time is not None:
                    if runtime > time:
                        not_interrupted = False
                        return None, runtime, not_interrupted, None, None, None

                feasible = False
                if m.status == GRB.INFEASIBLE:
                    model_feasible = False
                    feasible = True
                    return feasible, runtime, not_interrupted, None, None, None
                else:

                    # print("Obj:", m.objVal)
                    b_out = []
                    d_out = []
                    d_prime_out = []
                    p_out = []
                    t_out = []
                    for v in m.getVars():
                        varName = v.varName
                        varNameSplit = varName.split(',')
                        if varNameSplit[0] == 'd':
                            d_out.append(v.x)
                        if varNameSplit[0] == 'b':
                            b_out.append(v.x)
                        if varNameSplit[0] == 'd_prime':
                            d_prime_out.append(v.x)
                        if varNameSplit[0] == 't':
                            t_out.append(v.x)
                        if varNameSplit[0] == 'p':
                            p_out.append(v.x)
                            # print(str(varName) + ": " + str(v.x))

                    d = []
                    for i in range(n):
                        temp = []
                        for j in range(B):
                            temp.append(0)
                        d.append(temp)
                    j = 0
                    k = 0
                    for i in range(len(d_out)):
                        if d_out[i] > 0.9:
                            d[j][k] = 1
                        else:
                            d[j][k] = 0
                        k = k + 1
                        if (i + 1) % B == 0:
                            j = j + 1
                            k = 0

                    b = []
                    for i in range(n):
                        temp = []
                        for j in range(B):
                            temp.append(0)
                        b.append(temp)
                    j = 0
                    k = 0
                    for i in range(len(b_out)):
                        if b_out[i] > 0.9:
                            b[j][k] = 1
                        else:
                            b[j][k] = 0
                        k = k + 1
                        if (i + 1) % B == 0:
                            j = j + 1
                            k = 0

                    d_prime = []
                    for j in range(B):
                        temp1 = []
                        for i in range(n):
                            temp2 = []
                            for j in range(D):
                                temp2.append(0)
                            temp1.append(temp2)
                        d_prime.append(temp1)
                    j = 0
                    k = 0
                    l = 0
                    for i in range(len(d_prime_out)):
                        if d_prime_out[i] > 0.9:
                            d_prime[l][j][k] = 1
                        else:
                            d_prime[l][j][k] = 0
                        k = k + 1
                        if (i + 1) % D == 0:
                            j = j + 1
                            k = 0
                        if (i + 1) % (n * D) == 0:
                            j = 0
                            k = 0
                            l += 1

                    p = []
                    for j in range(B):
                        temp1 = []
                        for i in range(n):
                            temp2 = []
                            for j in range(D):
                                temp2.append(0)
                            temp1.append(temp2)
                        p.append(temp1)
                    j = 0
                    k = 0
                    l = 0
                    for i in range(len(p_out)):
                        if p_out[i] > 0.9:
                            p[l][j][k] = 1
                        else:
                            p[l][j][k] = 0
                        k = k + 1
                        if (i + 1) % D == 0:
                            j = j + 1
                            k = 0
                        if (i + 1) % (n * D) == 0:
                            j = 0
                            k = 0
                            l += 1

                    defense = []
                    defense.append([n - 1, 0, 0])
                    for l in range(B):
                        for k in range(D):
                            for i in range(n):
                                if p[l][i][k] == 1:
                                    defense.append([i, l + 1, k])
                                    break

                    t = []
                    for i in range(len(t_out)):
                        t.append(t_out[i])
                    print(defense)
                    print(m.objval)
                    print(runtime)
                    #if return_matrices:
                    #    return feasible, runtime, not_interrupted, m.objVal, defense, t, d, b, d_prime, p
                    #else:
                    #    return feasible, runtime, not_interrupted, m.objVal, defense, t
                self.saveSolution(instance_path, str(inst), defense, m.Objval, runtime)
            # Save
            self.total_saved.append(self.saved)
            self.total_times.append(self.times)
            # Reset
            self.saved = []
            self.times = []
        self.Statistics()

    def saveSolution(self, instance_path, instance, solution, saved, time):
        output_path = instance_path / instance / "Solution_Summary_GILP"
        with open(output_path, "w") as writer:
            writer.write("Solution: {}\n".format(solution))
            writer.write("Saved: {}\n".format(saved))
            writer.write("RunTime: {}\n".format(time))

            writer.write("G_mipgap: {}\n".format(self.config['experiment']['mip_gap']))
            writer.write("G_threads: {}\n".format(self.config['experiment']['threads']))
            writer.write("presolve: {}\n".format(self.config['experiment']['presolve']))
            writer.write("method: {}\n".format(self.config['experiment']['method']))

    def Statistics(self):
        time_mean = []
        time_std_dv=[]
        saved_mean = []
        saved_std_dv = []
        # Statistics for run time
        for node_size in self.total_times:
            m = np.mean(node_size)
            std = np.std(node_size)
            time_mean.append(m)
            time_std_dv.append(std)
        time_std_dv = np.asarray(time_std_dv)
        time_mean = np.asarray(time_mean)

        # Statistics for saved vertices
        for node_size in self.total_saved:
            m = np.mean(node_size)
            std = np.std(node_size)
            saved_mean.append(m)
            saved_std_dv.append(std)
        saved_std_dv=np.asarray(saved_std_dv)
        saved_mean=np.asarray(saved_mean)

        print(saved_mean)
        print(saved_std_dv)

        numpy.save(self.path / "Statistics_GILP", numpy.array([saved_mean, saved_std_dv, time_mean, time_std_dv]))
        y= np.arange(0,len(time_mean), 1, dtype=int)
        fig, ax = plt.subplots(1)
        ax.plot(y,saved_mean, label="Mean saved Vertices",color="blue")
        ax.set_title("ILP Mean Saved Vertices")
        ax.set_xlabel('Tree Size')
        ax.set_ylabel('Saved Vertices')
        ax.fill_between(y, saved_mean-saved_std_dv,saved_mean+saved_std_dv,facecolor="blue",alpha=0.5)
        plt.savefig(self.path / 'GILP_Saved.png')

        fig, ax = plt.subplots(1)
        ax.plot(y, time_mean, label="Mean Time Vertices", color="red")
        ax.set_title("ILP Mean Runtime")
        ax.set_xlabel('Tree Size')
        ax.set_ylabel('Runtime (s)')
        ax.fill_between(y, time_mean - time_std_dv, time_mean + time_std_dv, facecolor="red", alpha=0.5)
        plt.savefig(self.path / 'GILP_Time.png')