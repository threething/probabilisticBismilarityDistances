import time

import math
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Set
from ortools.linear_solver import pywraplp
import numpy as np

from probabilisticBismilarityDistancesPython.Pair import Pair
from probabilisticBismilarityDistancesPython.ProbBisimilarity import Distribution
from utils.constant import fileResult, file


class SolutionPair:
    def __init__(self, array, value):
        self.transportPlan = array
        self.opt = value

    def getValue(self):
        return self.opt

    def getPoint(self):
        return self.transportPlan


class SimplePolicyIteration:
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

    def __init__(self, numOfStates, transitions, labels, discount, accuracy):
        self.numOfStates = numOfStates
        self.transitions = transitions
        self.labels = labels
        self.coupling = {}
        # self.discrepancy = [0.0] * (numOfStates * numOfStates)
        self.discrepancy = np.zeros(numOfStates * numOfStates)
        self.toCompute = set()
        self.discount = discount
        self.accuracy = accuracy

    #计算并初始化耦合
    def initialize(self) -> bool:
        # ProbBisimilarity probBis = new ProbBisimilarity(this.numOfStates, this.labels, this.transitions);
        bisimulationSet = set()  # ProbBisimilarity.computeProbabilisticBisimilarity()

        for i in range(self.numOfStates):
            for j in range(i):
                p = Pair(i, j)
                if self.labels[i] == self.labels[j] and p not in bisimulationSet:
                    self.toCompute.add(p)
                elif self.labels[i] != self.labels[j]:
                    self.discrepancy[i * self.numOfStates + j] = 1
                    self.discrepancy[j * self.numOfStates + i] = 1

        # 初始化耦合
        for p in self.toCompute:
            self.init_coupling(p)

        # calculate discrepancy
        #计算差异
        self.discrepancy = self.calculate_discrepancy(self.coupling)
        return self.discrepancy is not None

    # /**
    #     * * solve the following LP
    #     * * objective: \min \sum (x_{s, t}) where s,t \in S
    #     * * constraints:
    #     * * x_{s, t} - discount * \sum_{u, v \in S}(\omega(u, v) * x_{u, v}) \geq 0
    #     * * where l(s) = l(t) and s \not~ t and \omega \in C(s, t)
    #     * * x_{s, t} = 1 where l(s) \not= l(t)
    #     * * x_{s, t} = 0 where s ~ t
    #     * * result: the solution of the LP is the least fixed point of \Gamma^{new
    #     * coupling}_{discount}
    #     **/

    def calculate_discrepancy(self, new_coupling):
        self.numLP += 1
        size = self.numOfStates * self.numOfStates
        solver = pywraplp.Solver('LinearProgramming', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
        # solver = pywraplp.Solver.CreateSolver('GLOP')
        # solver = pywraplp.Solver.CreateSolver('CLP_LINEAR_PROGRAMMING')
        infinity = solver.infinity()
        array = [solver.NumVar(0, 1, f'x{i}') for i in range(size)]


        for s in range(self.numOfStates):
            for t in range(s):
                st_id = s * self.numOfStates + t
                if self.labels[s] != self.labels[t]:
                    c = solver.Constraint(1, 1, 'c{}'.format(st_id))
                    c.SetCoefficient(array[st_id], 1)
                else:
                    pair = Pair(s, t)
                    if pair in self.toCompute:
                        if pair not in new_coupling:
                            print('Coupling has no key:', pair)
                            return None
                        for omega in new_coupling[pair]:
                            c = solver.Constraint(0, 1, 'c{}{}'.format(st_id, omega))
                            for u in range(self.numOfStates):
                                for v in range(u):
                                    index = u * self.numOfStates + v
                                    index2 = v * self.numOfStates + u
                                    if s == u and t == v:
                                        c.SetCoefficient(array[index], 1 - self.discount * omega.get_probability(
                                            index) - self.discount * omega.get_probability(index2))
                                    else:
                                        c.SetCoefficient(array[index], -self.discount * omega.get_probability(
                                            index) - self.discount * omega.get_probability(index2))
                    else:
                        c = solver.Constraint(0.0, 0.0, 'c{}'.format(st_id))
                        c.SetCoefficient(array[st_id], 1.0)

        objective = solver.Objective()
        for i in range(self.numOfStates):
            for j in range(self.numOfStates):
                index = i * self.numOfStates + j
                if j < i:
                    objective.SetCoefficient(array[index], 1.0)
                else:
                    objective.SetCoefficient(array[index], 0.0)

        objective.SetMinimization()
        # for i in range(size):
        #     print(array[i].solution_value())
        result_status = solver.Solve()
        # print("---------------------")
        # for i in range(size):
        #     print(array[i].solution_value())
        if result_status != pywraplp.Solver.OPTIMAL:
            print('The problem does not have an optimal solution!')
            return None


        # new_discrepancy = [0.0] * size
        new_discrepancy = np.zeros(size)
        for i in range(self.numOfStates):
            for j in range(i):
                index = i * self.numOfStates + j
                index2 = j * self.numOfStates + i
                v = array[index].solution_value()
                new_discrepancy[index] = array[index].solution_value()
                new_discrepancy[index2] = array[index].solution_value()

        return new_discrepancy

    def update_coupling(self, pair):
        newCoupling: Dict[Pair, Set[Distribution]] = defaultdict(set, self.coupling)
        set_ = newCoupling[pair]
        newSet: Set[Distribution] = set()
        s = pair.getRow()
        t = pair.getColumn()

        # newDistance = \Gamma^{new_coupling}_\lambda(\gamma^{old_coupling}_\lambda)(s, t)
        newDistance = 0

        muSize = len(self.transitions[s])
        nuSize = len(self.transitions[t])
        tmpArray: List[float] = []
        distrArray: List[Distribution] = []

        for mu in self.transitions[s]:
            for nu in self.transitions[t]:
                optSolution = self.solveTransportationProblem(mu, nu)
                self.numTP += 1
                tmpArray.append(self.discount * optSolution.getValue())
                distrArray.append(Distribution(optSolution.getPoint()))

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

        # coupling at (s, t) remains the same
        if len(newSet) == len(set_) and all([distr in newSet for distr in set_]):
            return 0

        oldDistance = self.discrepancy[s * self.numOfStates + t]
        if abs(newDistance - oldDistance) < self.accuracy:
            return 0

        # update the coupling if
        # (\Gamma^{new_coupling}_\lambda(\gamma^{old_coupling}_\lambda)(s, t)
        # \ls \gamma^{old_coupling}_\lambda(s, t))
        if newDistance < oldDistance:
            newCoupling[pair] = newSet
            self.coupling = dict(newCoupling)
            self.discrepancy = self.calculate_discrepancy(self.coupling)
            if self.discrepancy is None:
                return -1
            return 1
        return 0

    def init_coupling(self, pair):
        newSet = set()
        s = pair.getRow()
        t = pair.getColumn()
        for mu in self.transitions[s]:
            for nu in self.transitions[t]:
                newSet.add(Distribution(self.solveTransportationProblem(mu, nu).getPoint()))
        self.coupling[pair] = newSet

    def solveTransportationProblem(self, mu, nu):
        size = self.numOfStates * self.numOfStates
        solver = pywraplp.Solver('LinearProgramming', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
        array = [solver.NumVar(0.0, 1, 'x' + str(i)) for i in range(size)]

        for i in range(self.numOfStates):
            c = solver.Constraint(mu.get_probability(i), mu.get_probability(i), 'c' + str(mu) + str(i))
            for j in range(self.numOfStates):
                c.SetCoefficient(array[i * self.numOfStates + j], 1)

        for j in range(self.numOfStates):
            c = solver.Constraint(nu.get_probability(j), nu.get_probability(j), 'c' + str(nu) + str(j))
            for i in range(self.numOfStates):
                c.SetCoefficient(array[i * self.numOfStates + j], 1)

        objective = solver.Objective()
        for i in range(size):
            objective.SetCoefficient(array[i], self.discrepancy[i])

        objective.SetMinimization()
        resultStatus = solver.Solve()
        # Check that the problem has an optimal solution.
        if resultStatus != pywraplp.Solver.OPTIMAL:
            print("The problem does not have an optimal solution!")
            return None

        newDistr = [array[i].solution_value() for i in range(size)]
        sp = SolutionPair(newDistr, objective.Value())
        return sp

    def policyIteration(self):
        # startTime = System.nanoTime();
        start_time = time.time()
        if (not (self.initialize())):
            return False

        # // System.out.println("toCompute: " + this.toCompute.toString());
        is_improved = True
        while (is_improved):
            is_improved = False
            for p in self.toCompute:
                tmp = self.update_coupling(p)
                if (tmp < 0):
                    return False
                is_improved |= (tmp > 0)
        # self.runningTime = System.nanoTime() - startTime;
        self.runningTime = time.time() - start_time
        return True


def main():
    input_file = file
    output_file = fileResult
    discount = -1
    accuracy = 0.1
    numOfStates = -1
    labels = None
    transitions = {}
    # sys.argv.append(file)
    # sys.argv.append(fileResult)
    # sys.argv.append(discount)
    # sys.argv.append(accuracy)
    # Parse command-line arguments
    # if len(sys.argv) != 4:
    #     print("Use python SimplePolicyIteration.py <inputFile> <outputDistanceFile> <discountFactor> <accuracy>")
    # else:
    # Process the command line arguments
    # input_file = open(input_file, 'r')
    output_file = open(output_file, 'w')

    with open(input_file, 'r') as f:
        while True:
            try:
                line = f.readline().strip()
                if not line:
                    break
                numOfStates = int(line)
                labels = list(map(int, f.readline().split()))
                for i in range(numOfStates):
                    nDistrS = f.readline()
                    nDistr = int(nDistrS)
                    set1 = set()
                    for iDistr in range(nDistr):
                        doubleArray = list(map(float, f.readline().split()))
                        distr = Distribution(doubleArray)
                        set1.add(distr)
                    transitions[i] = set1

                discount = float(0.8)
                assert discount < 1, f"Discount factor {discount} should be less than or equal to 1"
                assert discount > 0, f"Discount factor {discount} should be greater than 0"

                accuracy = float(0.000001)
                assert accuracy <= 1, f"Accuracy {accuracy} should be less than or equal to 1"
                assert accuracy > 0, f"Accuracy {accuracy} should be greater than 0"

                spi = SimplePolicyIteration(numOfStates, transitions, labels, discount, accuracy)
                if not spi.policyIteration():
                    continue

                # Output the distances
                # format_str = f"%.{int(-math.log10(spi.accuracy))}f "
                # for i in range(spi.numOfStates):
                #     for j in range(spi.numOfStates):
                #         output_file.write(format_str % spi.discrepancy[i * spi.numOfStates + j])
                #     output_file.write("\n")
                # output_file.write("\n")

                format_str = "{:.{}f} "
                precision = int(-math.log10(spi.accuracy))

                # output the distances
                # with open('output.txt', 'w') as output:
                for i in range(spi.numOfStates):
                    for j in range(spi.numOfStates):
                        output_file.write(format_str.format(spi.discrepancy[i * spi.numOfStates + j], precision))
                    output_file.write('\n')
                output_file.write('\n')
            except ValueError as e:
                print(f"Caught a ValueError: {e}")
                break

if __name__ == "__main__":
    main()
