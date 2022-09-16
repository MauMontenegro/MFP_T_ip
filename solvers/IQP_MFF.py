"""
Moving Firefighter Problem on Trees
Integer Quadratic Programming Solution
Author: Mauro Alejandro Montenegro Meza
"""

import os
from pathlib import Path

import networkx as nx
import numpy
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
from utils.utils import generateInstance

BN = 10000  # Big Number for Restrictions

class IQP_MFF():
    def __init__(self, mode, load, path,config):
        # Individual Variables
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
        for d in next(os.walk(( self.path ))):
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
                N = instance[1]
                starting_fire = instance[2]
                T_Ad_Sym = instance[3]

                # --- MODEL---
                m = Model("mip1")
                m.Params.outputFlag = 1  # 0 - Off  //  1 - On
                m.setParam("MIPGap", self.config['experiment']['mip_gap'])
                m.setParam("Method", self.config['experiment']['method'])
                m.setParam("Presolve", self.config['experiment']['presolve'])  # -1 - Automatic // 0 - Off // 1 - Conservative // 2 - Aggresive
                m.setParam("NodefileStart", self.config['experiment']['nodefilestart'])
                m.setParam("Threads", self.config['experiment']['threads'])
                # m.setParam("PreQLinearize", -1); # -1 - Automatic // 0 - Off // 1 - Strong LP relaxation // 2 - Compact relaxation
                # m.params.BestObjStop = k

                # ---VARIABLES----
                vars = []
                for i in range(N):
                    temp = []
                    for j in range(N):
                        temp.append(0)
                    vars.append(temp)

                for phase in range(N):
                    for node in range(N):
                        vars[phase][node] = m.addVar(vtype=GRB.BINARY, name="x,%s" % str(phase) + "," + str(node))
                m.update()
                self.n_variables.append(N*N)

                # -------- OBJECTIVE FUNCTION ----------
                Nodes = list(T.nodes)
                Nodes.remove(N)
                Nodes.sort()
                weights = np.zeros(N)
                i = 0

                for node in Nodes:
                    weights[i] = len(nx.descendants(T, node)) + 1
                    i += 1
                weights = np.delete(weights, starting_fire)

                objective = 0
                weights_transpose = np.array(weights).T
                for i in range(N):
                    vars_tmp = np.delete(vars[i], starting_fire)
                    objective += np.dot(weights_transpose, vars_tmp)
                m.setObjective(objective, GRB.MAXIMIZE)

                count_const = 0
                # ----------------------First Constraint---------------------------------
                m.update()
                sum_vars = 0
                for phase in range(N):
                    sum_vars = 0
                    for node in range(N):
                        sum_vars += vars[phase][node]
                    count_const += 1
                    m.addConstr(sum_vars <= 1)

                # --------------------------Second Constraint--------------------------------

                # Obtain level for each node with starting fire as root
                levels = nx.single_source_shortest_path_length(
                    T, starting_fire
                )
                sorted_burning_times = numpy.zeros(N)

                # Sorted Burnig time for each node (from 0 to N)
                for i in range(N):
                    sorted_burning_times[i] = levels[i]

                # Constraint for initial Position
                initial_const = np.dot(T_Ad_Sym[N, 0:N], vars[0])
                initial_const_ = np.dot(sorted_burning_times.T, vars[0])
                count_const += 1
                m.addConstr(initial_const <= initial_const_, name="Init_Const")

                for phase in range(1, N):
                    q_1 = 0
                    for phase_range in range(0, phase):
                        for node_i in range(N):
                            for node_j in range(N):
                                q_1 += T_Ad_Sym[node_i][node_j] * (
                                        vars[phase_range][node_i] * vars[phase_range + 1][node_j])
                    q_1 += initial_const

                    q_2 = np.dot(sorted_burning_times.T, vars[phase])
                    d = 0
                    for node in range(N):
                        d += vars[phase][node]
                    d = BN * (1 - d)
                    q_2 += d

                    count_const += 1
                    m.addConstr(q_1 <= q_2, name="Q,%s" % str(phase))

                # ----------------------Third Constraint --------------------------
                leaf_nodes = [
                    node for node in T.nodes() if T.in_degree(node) != 0 and T.out_degree(node) == 0
                ]

                restricted_ancestors = {}
                for leaf in leaf_nodes:
                    restricted_ancestors[leaf] = list(nx.ancestors(T, leaf))
                    restricted_ancestors[leaf].remove(starting_fire)
                    restricted_ancestors[leaf].insert(0, leaf)

                for leaf in restricted_ancestors:
                    l_q = 0
                    for node in restricted_ancestors[leaf]:
                        for phase in range(N):
                            l_q += vars[phase][node]
                    count_const += 1
                    m.addConstr(l_q <= 1)

                # ----------------------------------Fourth Constrain-----------------------------------------
                # Force Consecutive Node Strategy
                for phase in range(N - 1):
                    vn_ = 0
                    v_ = 0
                    for node in range(N):
                        v_ += vars[phase][node]
                        vn_ += vars[phase + 1][node]
                    count_const += 1
                    m.addConstr(v_ >= vn_)

                self.n_restrictions.append(count_const)

                # ----------------- Optimize Step--------------------------------
                m.update()
                m.optimize()
                runtime = m.Runtime
                print("The run time is %f" % runtime)
                print("Obj:", m.ObjVal)
                self.saved.append(m.ObjVal)
                self.times.append(runtime)
                sol=[]
                for v in m.getVars():
                    if v.X > 0:
                        sol.append(v)
                        print(v.varName)
                self.solution = sol
                #m.write('IQP_model.lp')

                # Save Solution
                self.saveSolution(instance_path, str(inst), sol, m.Objval, runtime)
            # Save
            self.total_saved.append(self.saved)
            self.total_times.append(self.times)
            # Reset
            self.saved = []
            self.times = []
        self.Statistics()

    def getSolution(self):
        return self.solution

    def getTimes(self):
        return self.times

    def getSaved(self):
        return self.saved

    def getVariables_Restrictions(self):
        return self.n_variables,self.n_restrictions
    
    def saveSolution(self,instance_path,instance,solution,saved,time):
        output_path = instance_path / instance / "Solution_Summary_IQP"
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

        time_mean=np.asarray(time_mean)
        time_std_dv=np.asarray(time_std_dv)

        print(saved_mean)
        print(type(saved_mean))
        print(saved_std_dv)
        print(type(saved_std_dv))

        numpy.save(self.path / "Statistics_IQP", numpy.array([saved_mean, saved_std_dv, time_mean, time_std_dv]))
        y= np.arange(0,len(time_mean), 1, dtype=int)
        fig, ax =plt.subplots(1)
        ax.plot(y,saved_mean, label="Mean saved Vertices",color="blue")
        ax.fill_between(y, saved_mean+saved_std_dv,saved_mean-saved_std_dv,facecolor="blue",alpha=0.5)
        plt.savefig(self.path / 'IQP_Saved.png')

        fig, ax = plt.subplots(1)
        ax.plot(y, time_mean, label="Mean Time Vertices", color="red")
        ax.fill_between(y, time_mean + time_std_dv, time_mean - time_std_dv, facecolor="red", alpha=0.5)
        plt.savefig(self.path / 'IQP_Time.png')
