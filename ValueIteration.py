import sys
import time

import math
import numpy as np
from typing import Dict, List, Tuple
from ortools.linear_solver import pywraplp

from probabilisticBismilarityDistancesPython.ProbBisimilarity import Distribution
from utils.constant import file, fileResult


class ValueIteration:
    def __init__(self, num_states: int, transitions: Dict[int, List[Tuple[int, np.ndarray]]],
                 labels: np.ndarray, discount: float, accuracy: float, time_limit: int):
        self.num_states = num_states
        self.transitions = transitions
        self.labels = labels
        self.discrepancy = [0] * (num_states * num_states)
        self.discount = discount  # discount \in (0, 1)
        self.accuracy = accuracy
        self.time_limit = time_limit
        self.running_time = 0
        self.num_tp = 0
        self.num_iter = 0

    def apply_delta(self):
        new_discrepancy = [0] * (self.num_states * self.num_states)
        for s in range(self.num_states):
            for t in range(s):
                st_index = s * self.num_states + t
                ts_index = t * self.num_states + s

                if self.labels[s] != self.labels[t]:
                    new_discrepancy[st_index] = 1
                    new_discrepancy[ts_index] = 1
                else:
                    mu_size = len(self.transitions[s])
                    nu_size = len(self.transitions[t])
                    # tmp_array = np.zeros((mu_size, nu_size))
                    tmp_array = [0] * (mu_size * nu_size)
                    i = 0
                    for mu in self.transitions[s]:
                        for nu in self.transitions[t]:
                            tmp_array[i] = self.solveTransportationProblem(mu, nu)
                            i += 1

                    # \max_{\mu \in \delta(s)}\min_{\nu \in \delta(t)}K(d)(\mu,\nu)
                    max_mu = 0
                    for i in range(mu_size):
                        min_nu = 2
                        for j in range(nu_size):
                            ij_index = i * nu_size + j
                            if tmp_array[ij_index] < min_nu:
                                min_nu = tmp_array[ij_index]
                        if min_nu > max_mu:
                            max_mu = min_nu

                    # \max_{\nu \in \delta(t)}\min_{\mu \in \delta(s)}K(d)(\mu,\nu)
                    for j in range(nu_size):
                        min_mu = 2
                        for i in range(mu_size):
                            ij_index = i * nu_size + j
                            if tmp_array[ij_index] < min_mu:
                                min_mu = tmp_array[ij_index]
                        if min_mu > max_mu:
                            max_mu = min_mu

                    new_discrepancy[st_index] = self.discount * max_mu
                    new_discrepancy[ts_index] = new_discrepancy[st_index]

        return new_discrepancy

    def solveTransportationProblem(self, mu, nu):
        self.num_tp += 1
        size = self.num_states * self.num_states
        solver = pywraplp.Solver(
            'LinearProgramming', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
        array = [solver.NumVar(0.0, 1, 'x' + str(i)) for i in range(size)]

        for i in range(self.num_states):
            c = solver.Constraint(mu.get_probability(i), mu.get_probability(i), 'c' + str(mu) + str(i))
            for j in range(self.num_states):
                c.SetCoefficient(array[i * self.num_states + j], 1)

        for j in range(self.num_states):
            c = solver.Constraint(nu.get_probability(j), nu.get_probability(j), 'c' + str(nu) + str(j))
            for i in range(self.num_states):
                c.SetCoefficient(array[i * self.num_states + j], 1)

        objective = solver.Objective()
        for i in range(size):
            objective.SetCoefficient(array[i], self.discrepancy[i])

        objective.SetMinimization()
        resultStatus = solver.Solve()
        # Check that the problem has an optimal solution.
        if resultStatus != pywraplp.Solver.OPTIMAL:
            print('The problem does not have an optimal solution!')
            return -1

        return solver.Objective().Value()

    def valueIteration(self):
        start_time = time.time()
        end_time = 0
        is_improved = True
        while is_improved:
            end_time = time.time()
            if end_time - start_time > self.time_limit:
                self.runningTime = end_time - start_time
                return
            self.num_iter += 1
            is_improved = False
            new_discrepancy = self.apply_delta()
            if self.compareDiscrepancy(self.discrepancy, new_discrepancy) != 0:
                self.discrepancy = new_discrepancy
                is_improved = True
            else:
                self.discrepancy = new_discrepancy
        self.runningTime = time.time() - start_time

    def compareDiscrepancy(self,disc1, disc2):
        isGreater = False
        isLess = False
        for i in range(len(disc1)):
            if abs(disc1[i] - disc2[i]) < self.accuracy:
                continue
            if disc1[i] < disc2[i]:
                isLess = True
            if disc1[i] > disc2[i]:
                isGreater = True
        if isGreater and isLess:
            return -321  # warning big problem!
        if isGreater:
            return 1
        if isLess:
            return -1
        return 0


def main():
    input_file = file
    output_file = fileResult
    discount = -1.0
    accuracy = 0.1
    numOfStates = -1
    labels = None
    transitions = {}

    # parse command line arguments
    # if len(sys.argv) != 5:
    #     print("Use python ValueIteration.py <inputFile> <outputDistanceFile> <discountFactor> <accuracy>")
    # else:
    input_file = open(input_file, 'r')
    output_file = open(output_file, 'w')
    # input_file = open(sys.argv[1], 'r')
    # output_file = open(sys.argv[2], 'w')

    # parse input file
    for line in input_file:
        values = list(map(int, line.split()))
        numOfStates = values[0]
        labels = values[1:numOfStates + 1]
        transitions = {}
        for i in range(numOfStates):
            nDistr = values[numOfStates + 1 + i]
            set1 = set()
            for iDistr in range(nDistr):
                doubleArray = []
                for j in range(numOfStates):
                    doubleArray.append(values[numOfStates * (1 + nDistr) + i * numOfStates + j])
                distr = Distribution(doubleArray)
                set1.add(distr)
            transitions[i] = set1

        # process command line arguments
        discount = float(0.8)
        assert discount < 1.0, "Discount factor should be less than 1"
        assert discount > 0.0, "Discount factor should be greater than 0"
        accuracy = float(0.000001)
        assert accuracy <= 1.0, "Accuracy should be less than or equal to 1"
        assert accuracy > 0.0, "Accuracy should be greater than 0"

        vi = ValueIteration(numOfStates, transitions, labels, discount, accuracy, sys.maxsize)
        vi.valueIteration()

        format_str = f"%.{-int(math.log10(vi.accuracy))}f"
        # output the distances
        with open(output_file, 'w') as output:
            for i in range(vi.num_states):
                for j in range(vi.num_states):
                    output.write(format_str % vi.discrepancy[i * vi.num_states + j])
                output.write('\n')
            output.write('\n')
