import time

import math
from typing import List

import numpy as np
from ortools.linear_solver import pywraplp

from probabilisticBismilarityDistancesPython.Pair import Pair
import sys
import os
from typing import Dict, List, Set, Tuple
from pathlib import Path

from probabilisticBismilarityDistancesPython.PerformanceCompare import Distribution
from utils.constant import file, fileResult


class SimplePolicyIterationUndiscounted:
    numOfStates = 0
    transitions = {}
    labels = []
    coupling = {}
    discrepancy = []
    toCompute = ()
    discount = 1
    accuracy = 0.0
    runningTime = 0
    numTP = 0
    numLP = 0
    numSetM = 0

    def __init__(self, numOfStates, transitions, labels, accuracy):
        self.numOfStates = numOfStates
        self.transitions = transitions
        self.labels = labels
        self.coupling = {}
        self.discrepancy = [0] * (numOfStates * numOfStates)
        self.toCompute = set()
        self.accuracy = accuracy
        self.runningTime = 0
        self.numTP = 0
        self.numLP = 0
        self.numSetM = 0
        self.distanceOneSet = set()

    def initialize(self):
        bisimulationSet = set()
        for i in range(self.numOfStates):
            for j in range(i):
                p = Pair(i, j)
                if self.labels[i] == self.labels[j] and p not in bisimulationSet:
                    self.toCompute.add(p)
                elif self.labels[i] != self.labels[j]:
                    self.discrepancy[i * self.numOfStates + j] = 1
                    self.discrepancy[j * self.numOfStates + i] = 1
        for p in self.toCompute:
            self.initCoupling(p)
        self.discrepancy = self.calculateDiscrepancy(self.coupling)
        if self.discrepancy is None:
            return False
        return True

    def calculateDiscrepancy(self, new_coupling):
        num_lp = 0
        size = self.numOfStates * self.numOfStates
        solver = pywraplp.Solver.CreateSolver('CLP_LINEAR_PROGRAMMING')
        infinity = solver.infinity()
        array = [solver.NumVar(0.0, 1.0, f'x{i}') for i in range(size)]

        for s in range(self.numOfStates):
            for t in range(s):
                st_id = s * self.numOfStates + t
                if self.labels[s] != self.labels[t]:
                    c = solver.Constraint(1, 1, f'c{st_id}')
                    c.SetCoefficient(array[st_id], 1)
                else:
                    pair = (s, t)
                    if pair in self.toCompute:
                        if pair not in new_coupling:
                            print(f'Coupling has no key: {pair}')
                            return None
                        for omega in new_coupling[pair]:
                            c = solver.Constraint(0, 1, f'c{st_id}{omega}')
                            for u in range(self.numOfStates):
                                for v in range(u):
                                    index = u * self.numOfStates + v
                                    index2 = v * self.numOfStates + u
                                    if s == u and t == v:
                                        c.SetCoefficient(array[index],
                                                         1 - omega.get_probability(index) - omega.get_probability(
                                                             index2))
                                    else:
                                        c.SetCoefficient(array[index],
                                                         -omega.get_probability(index) - omega.get_probability(index2))
                    else:
                        c = solver.Constraint(0, 0, f'c{st_id}')
                        c.SetCoefficient(array[st_id], 1)

        objective = solver.Objective()
        for i in range(self.numOfStates):
            for j in range(self.numOfStates):
                index = i * self.numOfStates + j
                if j < i:
                    objective.SetCoefficient(array[index], 1)
                else:
                    objective.SetCoefficient(array[index], 0)
        objective.SetMinimization()

        result_status = solver.Solve()
        if result_status != pywraplp.Solver.OPTIMAL:
            print('The problem does not have an optimal solution!')
            return None

        new_discrepancy = np.zeros(size)
        for i in range(self.numOfStates):
            for j in range(i):
                index = i * self.numOfStates + j
                index2 = j * self.numOfStates + i
                new_discrepancy[index] = array[index].solution_value()
                new_discrepancy[index2] = array[index].solution_value()

        return new_discrepancy

    def updateCoupling(self, pair):
        new_coupling = self.coupling.copy()
        set1 = new_coupling.get(pair)
        new_set = set()
        s = pair.row
        t = pair.column

        # new_distance = \Gamma^{new_coupling}_\lambda(\gamma^{old_coupling}_\lambda)(s, t)
        new_distance = 0

        mu_size = len(self.transitions[s])
        nu_size = len(self.transitions[t])
        tmp_array: List[float] = []
        distr_array: List[Distribution] = []
        i = 0

        for mu in self.transitions[s]:
            for nu in self.transitions[t]:
                opt_solution = self.solveTransportationProblem(mu, nu, self.discrepancy)
                self.numTP += 1
                tmp_array.append(self.discount * opt_solution.getValue())
                distr_array.append(Distribution(opt_solution.getPoint()))
                i += 1

        # for each mu choose a nu
        for i in range(mu_size):
            min_distance = 2
            min_distr = None
            for j in range(nu_size):
                ijIndex = i * nu_size + j
                if tmp_array[ijIndex] < min_distance:
                    min_distr = distr_array[ijIndex]
                    min_distance = tmp_array[ijIndex]
            if min_distance > new_distance:
                new_distance = min_distance
            new_set.add(min_distr)

        # for each nu choose a mu
        for j in range(nu_size):
            min_distance = 2
            min_distr = None
            for i in range(mu_size):
                ij_index = i * nu_size + j
                if tmp_array[ij_index] < min_distance:
                    min_distr = distr_array[ij_index]
                    min_distance = tmp_array[ij_index]
            if min_distance > new_distance:
                new_distance = min_distance
            new_set.add(min_distr)

        # coupling at (s, t) remains the same
        if len(new_set) == len(set1) and set1.issubset(new_set):
            return 0

        old_distance = self.discrepancy[s * self.numOfStates + t]
        if abs(new_distance - old_distance) < self.accuracy:
            return 0

        # update the coupling if
        # (\Gamma^{new_coupling}_\lambda(\gamma^{old_coupling}_\lambda)(s, t) < \gamma^{old_coupling}_\lambda(s, t))
        if new_distance < old_distance:
            new_coupling[pair] = new_set
            self.discrepancy = self.calculateDiscrepancy(new_coupling)
            if self.discrepancy is None:
                return -1
            return 1
        return 0

    def initCoupling(self, pair):
        newSet = set()
        s = pair.getRow()
        t = pair.getColumn()
        for mu in self.transitions[s]:
            for nu in self.transitions[t]:
                # newSet.add(Distribution(solveTransportationProblem(mu, nu, self.discrepancy).getPoint()))
                newSet.add(self.initialSolution(mu, nu))
        self.coupling[pair] = newSet

    # TP initial solution North-West Corner method
    def initialSolution(self, mu, nu):
        arr = [0] * (self.numOfStates * self.numOfStates)

        class Support:
            def __init__(self, s, p):
                self.state = s
                self.prob = p

        srcTmp = [Support(i, mu.get_probability(i)) for i in range(self.numOfStates)]
        tgtTmp = [Support(i, nu.get_probability(i)) for i in range(self.numOfStates)]

        class SupportComparator:
            def __init__(self):
                pass

            def compare(self, o1, o2):
                if o1.prob == o2.prob:
                    return 0
                elif o1.prob > o2.prob:
                    return -1
                else:
                    return 1

        srcTmp.sort(key=lambda x: SupportComparator().compare(x, x))
        tgtTmp.sort(key=lambda x: SupportComparator().compare(x, x))

        for i in range(self.numOfStates):
            for j in range(self.numOfStates):
                mini = min(srcTmp[i].prob, tgtTmp[j].prob)
                u = srcTmp[i].state
                v = tgtTmp[j].state

                arr[u * self.numOfStates + v] = mini
                # arr[v*self.numOfStates + u] = min
                srcTmp[i].prob -= mini
                tgtTmp[j].prob -= mini

        return Distribution(arr)

    def solveTransportationProblem(self, mu, nu, distance):
        size = self.numOfStates * self.numOfStates
        solver = pywraplp.Solver.CreateSolver('CLP_LINEAR_PROGRAMMING')

        array = [solver.NumVar(0.0, 1.0, str(i)) for i in range(size)]

        for i in range(self.numOfStates):
            c = solver.Constraint(mu.get_probability(i), mu.get_probability(i))
            for j in range(self.numOfStates):
                c.SetCoefficient(array[i * self.numOfStates + j], 1)

        for j in range(self.numOfStates):
            c = solver.Constraint(nu.get_probability(j), nu.get_probability(j))
            for i in range(self.numOfStates):
                c.SetCoefficient(array[i * self.numOfStates + j], 1)

        objective = solver.Objective()
        for i in range(size):
            objective.SetCoefficient(array[i], distance[i])

        objective.SetMinimization()
        result_status = solver.Solve()

        # Check that the problem has an optimal solution.
        if result_status != pywraplp.Solver.OPTIMAL:
            print("The TP problem does not have an optimal solution!")
            return None

        new_distr = [array[i].solution_value() for i in range(size)]
        sp = self.SolutionPair(new_distr, objective.Value())
        return sp

    def checkSupportSet(self, omega, set):
        for i in range(self.numOfStates):
            for j in range(self.numOfStates):
                if omega[i * self.numOfStates + j] > 0 and self.Pair(i, j) not in set:
                    return False
        return True

    from typing import List, Tuple, Set

    class Distribution:
        def __init__(self, values: List[float]):
            self.values = values

    class SolutionPair:
        def __init__(self, array, value):
            self.transportPlan = array
            self.opt = value

        def getValue(self):
            return self.opt

        def getPoint(self):
            return self.transportPlan

    class Pair:
        def __init__(self, row: int, column: int):
            self.row = row
            self.column = column

    def calculateSelfClosedSet(self):

        setM = set(self.toCompute)

        # exclude the pairs that have distance 0
        setM = set(filter(lambda p: self.discrepancy[p.row * self.numOfStates + p.column] != 0, setM))

        if not setM:
            return setM

        # iterate until reach the greatest fixed point
        isNotFixedPoint = False
        while not isNotFixedPoint:
            isNotFixedPoint = True
            newSet = set(setM)
            for p in setM:
                s = p.row
                t = p.column
                stIndex = s * self.numOfStates + t
                # check mu
                isRemoved = False

                for mu in self.transitions[s]:
                    for nu in self.transitions[t]:
                        sp = self.solveTransportationProblem(mu, nu, self.discrepancy)
                        sol = sp.getValue()
                        if abs(sol - self.discrepancy[stIndex]) <= self.accuracy:
                            omega = sp.getPoint()
                            if not self.checkSupportSet(omega, setM):
                                newSet.remove(p)
                                isNotFixedPoint = False
                                isRemoved = True
                                break
                    if isRemoved:
                        break
                if isRemoved:
                    continue

                # check nu
                for nu in self.transitions[t]:
                    for mu in self.transitions[s]:
                        sp = self.solveTransportationProblem(mu, nu, self.discrepancy)
                        sol = sp.getValue()
                        if abs(sol - self.discrepancy[stIndex]) <= self.accuracy:
                            omega = sp.getPoint()
                            if not self.checkSupportSet(omega, setM):
                                newSet.remove(p)
                                isNotFixedPoint = False
                                isRemoved = True
                                break
                    if isRemoved:
                        break

            setM = newSet

        return setM

    def calculateTheta(self, setM):
        theta = 1
        for p in setM:
            thetap = self.getMinTheta(p)
            theta = thetap if thetap < theta else theta
        return theta

    def getMinTheta(self, pair):
        s, t = pair
        muSize = len(self.transitions[s])
        nuSize = len(self.transitions[t])
        pointDistance = self.discrepancy[s * self.numOfStates + t]

        tmpArray = [self.solveTransportationProblem(mu, nu, self.discrepancy).getValue()
                    for mu in self.transitions[s] for nu in self.transitions[t]]

        thetaST = pointDistance
        # get theta_s
        for i in range(muSize):
            minNu = 1
            for j in range(nuSize):
                tmp = tmpArray[i * nuSize + j]
                minNu = tmp if tmp < minNu else minNu
            tmp = pointDistance - minNu
            thetaST = tmp if tmp > 0 and tmp < thetaST else thetaST

        # get theta_t
        for j in range(nuSize):
            minMu = 1
            for i in range(muSize):
                tmp = tmpArray[i * nuSize + j]
                minMu = tmp if tmp < minMu else minMu
            tmp = pointDistance - minMu
            thetaST = tmp if tmp > 0 and tmp < thetaST else thetaST
        return thetaST

    def validateDiscrepancy(self, pairSet):
        distArray = np.copy(self.discrepancy)
        for s in range(self.numOfStates):
            for t in range(s):
                p = self.Pair(s, t)
                if p in self.toCompute:
                    newDistance = 0
                    muSize = len(self.transitions[s])
                    nuSize = len(self.transitions[t])
                    tmpArray = np.zeros(muSize * nuSize)
                    i = 0
                    for mu in self.transitions[s]:
                        for nu in self.transitions[t]:
                            optSolution = self.solveTransportationProblem(mu, nu, self.discrepancy)
                            self.numTP += 1
                            tmpArray[i] = optSolution.getValue()
                            i += 1

                    for i in range(muSize):
                        minDistance = 2
                        for j in range(nuSize):
                            ijIndex = i * nuSize + j
                            if tmpArray[ijIndex] < minDistance:
                                minDistance = tmpArray[ijIndex]
                        if minDistance > newDistance:
                            newDistance = minDistance

                    for j in range(nuSize):
                        minDistance = 2
                        for i in range(muSize):
                            ijIndex = i * nuSize + j
                            if tmpArray[ijIndex] < minDistance:
                                minDistance = tmpArray[ijIndex]
                        if minDistance > newDistance:
                            newDistance = minDistance

                    distArray[s * self.numOfStates + t] = newDistance
                    distArray[t * self.numOfStates + s] = newDistance

        self.discrepancy = distArray

    def adjustDiscrepancy(self, theta, setM):
        for pair in setM:
            s, t = pair.getRow(), pair.getColumn()
            stIndex = s * self.numOfStates + t
            tmp = self.discrepancy[s * self.numOfStates + t]
            self.discrepancy[stIndex] = tmp - theta
            self.discrepancy[t * self.numOfStates + s] = self.discrepancy[stIndex]

    def updateAllCouplings(self):
        for pair in self.toCompute:
            s = pair.getRow()
            t = pair.getColumn()

            newSet = set()
            newDistance = 0

            muSize = len(self.transitions[s])
            nuSize = len(self.transitions[t])
            tmpArray = [0] * (muSize * nuSize)
            distrArray = [None] * (muSize * nuSize)
            i = 0
            for mu in self.transitions[s]:
                for nu in self.transitions[t]:
                    optSolution = self.solveTransportationProblem(mu, nu, self.discrepancy)
                    self.numTP += 1
                    tmpArray[i] = optSolution.getValue()
                    distrArray[i] = Distribution(optSolution.getPoint())
                    i += 1

            # for each mu choose a nu
            for i in range(muSize):
                minDistance = 2
                minDistr = None
                for j in range(nuSize):
                    ijIndex = i * nuSize + j
                    if tmpArray[ijIndex] < minDistance:
                        minDistr = distrArray[ijIndex]
                        minDistance = tmpArray[ijIndex]
                if minDistance > newDistance:
                    newDistance = minDistance
                newSet.add(minDistr)

            # for each nu choose a mu
            for j in range(nuSize):
                minDistance = 2
                minDistr = None
                for i in range(muSize):
                    ijIndex = i * nuSize + j
                    if tmpArray[ijIndex] < minDistance:
                        minDistr = distrArray[ijIndex]
                        minDistance = tmpArray[ijIndex]
                if minDistance > newDistance:
                    newDistance = minDistance
                newSet.add(minDistr)

            # update the coupling if
            # (\Gamma^{new_coupling}_\lambda(\gamma^{old_coupling}_\lambda)(s,
            # t)
            # \ls \gamma^{old_coupling}_\lambda(s, t))
            self.coupling[pair] = newSet
        self.discrepancy = self.calculateDiscrepancy(self.coupling)

    def policy_iteration(self):
        start_time = time.time()
        if not self.initialize():
            return False

        is_min = False
        while not is_min:
            is_min = True
            is_improved = True
            while is_improved:
                is_improved = False
                for pair in self.toCompute:
                    tmp = self.updateCoupling(pair)
                    if tmp < 0:
                        return False
                    is_improved |= tmp > 0

            set_M = self.calculateSelfClosedSet()
            if set_M:
                self.numSetM += 1
                is_min = False
                theta = self.calculateTheta(set_M)
                self.adjustDiscrepancy(theta, set_M)

                self.updateAllCouplings()

        self.running_time = time.time() - start_time
        return True


def main():
    input_file = file
    output_file = fileResult
    accuracy = 0.1
    num_of_states = -1
    labels = None
    transitions: Dict[int, Set[Distribution]] = {}

    # parse input file
    # if len(sys.argv) != 4:
    #     print("Use python SimplePolicyIteration.py 0: <inputFile> 1: <outputDistanceFile> 2: <accuracy>")
    #     sys.exit(1)
    # else:
    # input_file = open(input_file, 'r')
    # output_file = open(output_file, 'w')
    try:
        accuracy = float(0.000001)
        assert accuracy <= 1, f"Accuracy {accuracy} should be less than or equal to 1"
        assert accuracy > 0, f"Accuracy {accuracy} should be greater than 0"
    except ValueError:
        print("Accuracy not provided in the right format")
        sys.exit(1)

    with open(input_file, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            num_of_states = int(line)
            labels = list(map(int, f.readline().split()))

            for i in range(num_of_states):
                n_distr = int(f.readline())
                set_distr = set()
                for i_distr in range(n_distr):
                    double_array = list(map(float, f.readline().split()))
                    distr = Distribution(double_array)
                    set_distr.add(distr)
                transitions[i] = set_distr

            spi = SimplePolicyIterationUndiscounted(num_of_states, transitions, labels, accuracy)
            if not spi.policy_iteration():
                continue

            format_str = "{:.{}f} "
            precision = int(-math.log10(spi.accuracy))
            # output the distances
            with open(output_file, 'w') as output:
                for i in range(spi.numOfStates):
                    for j in range(spi.numOfStates):
                        output.write(format_str.format(spi.discrepancy[i * spi.numOfStates + j], precision))
                    output.write('\n')
                output.write('\n')


if __name__ == "__main__":
    main()
