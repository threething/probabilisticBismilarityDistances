import sys
import math
import random
from typing import List, Tuple, Dict, Set

class Distribution:
    def __init__(self, array: List[float]):
        self.array = array

class SimplePolicyIteration:
    def __init__(self, num_states: int, transitions: Dict[int, Set[Distribution]], labels: List[int], accuracy: float):
        self.num_states = num_states
        self.transitions = transitions
        self.labels = labels
        self.accuracy = accuracy
        self.running_time = 0
        self.num_tp = 0
        self.num_lp = 0
        self.num_set_m = 0
        self.discrepancy = [0] * num_states
        self.policy = [random.randint(0, len(self.transitions[state])-1) for state in range(num_states)]

    def policy_iteration(self) -> bool:
        while True:
            delta = 0
            self.num_tp += 1
            for state in range(self.num_states):
                old_action = self.policy[state]
                old_value = self.get_value(state, old_action)
                best_value, best_action = self.get_best_value_and_action(state)
                self.policy[state] = best_action
                self.discrepancy[state] = abs(best_value - old_value)
                delta = max(delta, self.discrepancy[state])
            self.running_time += 1
            if delta < self.accuracy:
                return True
            self.num_lp += 1

    def get_value(self, state: int, action: int) -> float:
        value = 0
        for distribution in self.transitions[state]:
            value += distribution.array[action] * self.get_expected_utility(state, distribution)
        return value

    def get_expected_utility(self, state: int, distribution: Distribution) -> float:
        utility = 0
        for next_state in range(self.num_states):
            utility += distribution.array[next_state] * self.get_utility(next_state)
        return utility

    def get_utility(self, state: int) -> float:
        return self.labels[state]

    def get_best_value_and_action(self, state: int) -> Tuple[float, int]:
        best_value = float('-inf')
        best_action = None
        for action in range(len(self.transitions[state])):
            value = self.get_value(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        return best_value, best_action

class ValueIteration:
    def __init__(self, num_states: int, transitions: Dict[int, Set[Distribution]], labels: List[int], discount: float, accuracy: float, max_iterations: int):
        self.num_states = num_states
        self.transitions = transitions
        self.labels = labels
        self.discount = discount
        self.accuracy = accuracy
        self.max_iterations = max_iterations
        self.running_time = 0
        self.num_tp = 0
        self.num_iter = 0
        self.discrepancy = [0] * num_states
        self.values = [0] * num_states

    def value_iteration(self) -> None:
        while True:
            delta = 0
            self.num_tp += 1
            for state in range(self.num_states):
                old_value = self.values[state]
                best_value = float('-inf')
                for distribution in self.transitions[state]:
                    value = SimplePolicyIteration.get_expected_utility
