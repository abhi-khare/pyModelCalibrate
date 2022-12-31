from statistics import mean
from partition import *


class Bin:

    def __init__(self, bin_id: int) -> None:

        self.bin_id: int = bin_id  # A unique integer ID for the bin
        self.samples: list[tuple] = []  # A list of tuple containing the input (probability) and the output (label)
        self.calibrated_score = None  # Actual proportion of samples that belongs to class 1
        self.low_prob = None  # lowest probability score in the sample set.
        self.high_prob = None  # highest probability score in the sample set.
        self.avg_prob = None  # average of the probability scores of this sample set.

    def add_sample(self, prob: float, label: int) -> None:

        self.samples.append((prob, label))

    def compute_statistics(self) -> None:

        probs, labels = zip(*self.samples)

        self.low_prob, self.high_prob = min(probs), max(probs)

        self.avg_prob, self.calibrated_score = mean(probs), mean(labels)


class HistogramBinningCalibrator:

    def __init__(self, probs: list, labels: list, partition_scheme: str = 'mass'):

        # check for errors
        self.check_errors(probs, labels, partition_scheme)

        self.probs = probs
        self.labels = labels
        self.partition_scheme = partition_scheme

        self.samples = [(prob, label) for prob, label in zip(probs, labels)]
        self.bins = {}

    @staticmethod
    def check_errors(probs: list, labels: list, partition_scheme: str):

        if partition_scheme not in ("mass", "width"):
            raise ValueError(f"Incorrect parameter value provided.")

        if len(probs) != len(labels):
            raise ValueError(f'Size mismatch. prob contains {len(probs)} elements \
                label contains {len(labels)} element')

        if len(probs) == 0:
            raise ValueError(f"prob array must contain atleast 1 element")

        if len(labels) == 0:
            raise ValueError(f"label array must contain atleast 1 element")

        for prob in probs:
            if not isinstance(prob, (float, int)):
                raise ValueError(f'variable prob contains value of incorrect datatype.\
                    Expected float found {type(prob)}')

            if not 0 <= prob <= 1:
                raise ValueError(f'Value out of Bound. Expected value between 0 and 1.')

        for label in labels:
            if not isinstance(label, (int,)):
                raise ValueError(f'variable label contains value of incorrect datatype.\
                    Expected float found {type(label)}')

    def fit(self):

        # create partition
        probs, labels, part_ids = None, None, None
        if self.partition_scheme == 'mass':
            probs, labels, part_ids = get_uniform_mass_partitions(samples=self.samples)
        elif self.partition_scheme == 'width':
            probs, labels, part_ids = get_uniform_width_partitions(samples=self.samples)

        # learn bin wise statistics
        for part_id in set(part_ids):
            _bin = Bin(bin_id=part_id)
            self.bins[part_id] = _bin

        for idx, part_id in enumerate(part_ids):
            self.bins[part_id].add_sample(prob=probs[idx],
                                          label=labels[idx])

        # compute calibrated score for bins
        for _, bin in self.bins.items():
            bin.compute_calibrated_score()

    def predict(self):
        pass
