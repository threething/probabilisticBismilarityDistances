import collections
from typing import List, Tuple, Dict, Set


class Distribution:
    def __init__(self, distr: List[float]):
        self.distr = distr

    def __hash__(self):
        return hash(tuple(self.distr))

    def __eq__(self, other):
        return isinstance(other, Distribution) and self.distr == other.distr

    def get_number_states(self):
        return len(self.distr)

    def get_probability(self, index):
        return self.distr[index]


class StepClass:
    def __init__(self, label: int, set_of_distr: Set[Distribution] = None):
        self.label = label
        self.set_of_distr = set_of_distr or set()

    def add_all_distribution(self, set_of_distr: Set[Distribution]):
        self.set_of_distr |= set_of_distr

    def add_distribution(self, distr: Distribution):
        self.set_of_distr.add(distr)

    def get_label(self):
        return self.label

    def get_distributions(self):
        return self.set_of_distr

    def is_intersect(self, set_: Set[Distribution]):
        return bool(self.set_of_distr & set_)


class ProbBisimilarity:
    def __init__(self, num_of_states: int, labels: List[int], transitions: Dict[int, Set[Distribution]]):
        self.num_of_states = num_of_states
        self.labels = labels
        self.transitions = transitions
        self.NewBlocks = set()
        self.NewStepClasses = set()
        self.partitions = set()
        self.stepClasses = set()

    def initialize(self):
        # initialize the partitions by labels
        map_partitions = collections.defaultdict(set)
        for i, label in enumerate(self.labels):
            map_partitions[label].add(i)

        max_, max_set = 0, None
        for s in map_partitions.values():
            self.partitions.add(s)
            if len(s) > max_:
                max_set = s
                max_ = max_set

            self.NewBlocks.add(s)

        if max_set in self.NewBlocks:
            self.NewBlocks.remove(max_set)

        # initialize the StepClasses by labels
        map_step_classes = {}
        for i, label in enumerate(self.labels):
            distr_set = self.transitions.get(i, set())
            step_class = map_step_classes.setdefault(label, StepClass(label))
            step_class.add_all_distribution(distr_set)

        self.stepClasses |= set(map_step_classes.values())
        self.NewStepClasses |= self.stepClasses

    def split(self, step_classes: Set[StepClass], splitter: Set[int]):
        step_classes_old = set(step_classes)
        step_classes.clear()

        for step_class in step_classes_old:
            label = step_class.get_label()
            map_split_distributions = collections.defaultdict(set)
            for distr in step_class.get_distributions():
                sum_ = 0
                for s in splitter:
                    sum_ += distr.get_probability(s)

                map_split_distributions[sum_].add(distr)

            for distr_set in map_split_distributions.values():
                step_classes.add(StepClass(label, distr_set))

            if any(distr_set in self.NewStepClasses for distr_set in map_split_distributions.values()):
                self.NewStepClasses.remove(step_class)


    def split(self, stepClasses: Set[StepClass], splitter: Set[int]) -> None:
        stepClassesOld = stepClasses.copy()
        stepClasses.clear()
        for stepClass in stepClassesOld:
            # split it
            label = stepClass.getLabel()
            mapSplitDistributions = collections.defaultdict(set)
            # loop through all the distributions
            for distr in stepClass.getDistributions():
                s = sum(distr.getProbability(i) for i in splitter)
                mapSplitDistributions[s].add(distr)
            # update StepClasses
            for set_ in mapSplitDistributions.values():
                stepClasses.add(StepClass(label, set_))
            # update NewStepClasses
            if mapSplitDistributions:
                if not stepClass.isIntersect(mapSplitDistributions.values()):
                    self.NewStepClasses.remove

    def refine(self,partitions, stepClass):
        partitionsOld = set(partitions)
        partitions.clear()
        for part in partitionsOld:
            # possibly split each part
            label = self.labels[list(part)[0]]
            if label != stepClass.getLabel():
                partitions.add(part)
                continue
            newSet = set()
            for i in part:
                if stepClass.isIntersect(self.transitions[i]):
                    newSet.add(i)
            # the complement of the set
            compSet = part - newSet
            # update the partitions
            if newSet:
                partitions.add(newSet)
            if compSet:
                partitions.add(compSet)
            # update NewBlocks
            if not (newSet and compSet):
                continue
            if len(newSet) < len(compSet):
                self.NewBlocks.add(newSet)
            else:
                self.NewBlocks.add(compSet)

    def computeProbabilisticBisimilarity(self):
        self.initialize()
        while self.NewStepClasses or self.NewBlocks:
            # phase 1
            itrNewBlocks = iter(self.NewBlocks)
            for block in itrNewBlocks:
                self.NewBlocks.remove(block)
                self.split(self.stepClasses, block)
            # phase 2
            itrNewStepClasses = iter(self.NewStepClasses)
            for sc in itrNewStepClasses:
                self.NewStepClasses.remove(sc)
                self.refine(self.partitions, sc)
        bisimilationSet = set()
        for set_ in self.partitions:
            intArray = list(set_)
            for i in range(len(intArray)):
                s = intArray[i]
                for j in range(i + 1):
                    bisimilationSet.add((s, intArray[j]))
        return bisimilationSet


    # /*
    # public static void main(String[] args) {
    #
    #     int numOfStates = 3;
    #     int[] labels = {0, 0, 1};
    #     Map<Integer, Set<Distribution>> transitions = new HashMap<>();
    #     transitions.computeIfAbsent(0, k -> new HashSet<Distribution>())
    #             .add(new Distribution(new double[]{1, 0, 0}));
    #     transitions.computeIfAbsent(1, k -> new HashSet<Distribution>())
    #             .add(new Distribution(new double[]{0, 1, 0}));
    #     transitions.computeIfAbsent(1, k -> new HashSet<Distribution>())
    #             .add(new Distribution(new double[]{0.5, 0, 0.5}));
    #     transitions.computeIfAbsent(2, k -> new HashSet<Distribution>())
    #             .add(new Distribution(new double[]{0, 0, 1}));
    #
    #     ProbBisimilarity probBis = new ProbBisimilarity(numOfStates, labels, transitions);
    #     probBis.initialize();
    #     while (!probBis.NewStepClasses.isEmpty() || !probBis.NewBlocks.isEmpty()) {
    #         // phase 1
    #         Iterator<Set<Integer>> itrNewBlocks = probBis.NewBlocks.iterator();
    #         while (itrNewBlocks.hasNext()) {
    #             Set<Integer> block = (Set<Integer>) itrNewBlocks.next();
    #             itrNewBlocks.remove();
    #             probBis.split(probBis.stepClasses, block);
    #         }
    #         // phase 2
    #         Iterator<StepClass> itrNewStepClasses = probBis.NewStepClasses.iterator();
    #         while (itrNewStepClasses.hasNext()) {
    #             StepClass sc = (StepClass) itrNewStepClasses.next();
    #             itrNewStepClasses.remove();
    #             probBis.refine(probBis.partitions, sc);
    #         }
    #     }
    #     // print the results
    #     for (Set<Integer> s : probBis.partitions) {
    #         System.out.println(s.toString());
    #     }
    # }
    # */

