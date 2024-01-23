import sys
import os
import time
from typing import List, Tuple, Set, Dict
import numpy as np


# from probabilisticBismilarityDistancesPython import *
from probabilisticBismilarityDistancesPython.ProbBisimilarity import Distribution
from probabilisticBismilarityDistancesPython.SimplePolicyIteration import SimplePolicyIteration
from probabilisticBismilarityDistancesPython.ValueIteration import ValueIteration
from utils.constant import *


def get_error(dist1: List[float], dist2: List[float]) -> float:
    error = 0
    for i in range(len(dist1)):
        diff = abs(dist1[i] - dist2[i])
        if error < diff:
            error = diff
    return error


def main():
    # if len(args) != 5:
    #     print(
    #         "Usage: python performance_compare.py <inputFile> <outputFile> <errorInputFile> <discountFactor> <accuracy>")
    #     sys.exit(1)

    transitions = {}
    discount: float = -1
    accuracy: float = 0.1
    input_file = file
    output_file = fileResult
    errorInputFile = errorResult
    discount = -1
    accuracy = 0.1
    numOfStates = -1
    labels = None
    transitions = {}
    # input_file = open(input_file, 'r')
    # output_file = open(output_file, 'w')

    try:
        with open(input_file, "r") as input_file:
            num_pa = -1
            while True:
                line = input_file.readline().strip()
                if not line:
                    break

                num_pa += 1
                numOfStates = int(line)
                labels = [int(label) for label in input_file.readline().strip().split()]

                for i in range(numOfStates):
                    n_distr = int(input_file.readline().strip())
                    set_distr = set()
                    for _ in range(n_distr):
                        double_array = [float(p) for p in input_file.readline().strip().split()]
                        distr = Distribution(double_array)
                        set_distr.add(distr)
                    transitions[i] = set_distr

                discount = float(0.8)
                assert discount <= 1, f"Discount factor {discount} should be less than or equal to 1"
                assert discount > 0, f"Discount factor {discount} should be greater than 0"

                accuracy = float(0.000001)
                assert accuracy <= 1, f"Accuracy {accuracy} should be less than or equal to 1"
                assert accuracy > 0, f"Accuracy {accuracy} should be greater than 0"

                start_time = time.time()
                spi = SimplePolicyIteration(numOfStates, transitions, labels, discount, accuracy)

                if not spi.policyIteration():
                    with open(errorInputFile, "a") as error_input_file:
                        error_input_file.write(f"{input_file}: {num_pa}\n")
                    continue

                vi = ValueIteration(numOfStates, transitions, labels, discount, accuracy, spi.runningTime)
                vi.valueIteration()

                error = get_error(spi.discrepancy, vi.discrepancy)
                end_time = time.time()

                with open(output_file, "a") as output_file:
                    output_file.write(
                        f"{spi.runningTime}, {spi.numTP}, {spi.numLP}, {vi.running_time}, {vi.num_tp}, {vi.num_tp}, {error}\n")

    except FileNotFoundError:
        print(f"File {input_file} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
