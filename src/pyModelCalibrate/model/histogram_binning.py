from partition import *


class Bin:

    def __init__(self, bin_id: int) -> None:

        self.bin_id: int = bin_id
        self.samples: list[tuple] = []
        self.calibrated_score = None

    @staticmethod
    def check_values(prob: float, label: int):

        if not isinstance(prob, (float, int)):
            raise ValueError(f'variable prob contains value of incorrect datatype.\
                Expected float found {isinstance(prob)}')

        if not 0 <= prob <= 1:
            raise ValueError(f'Value out of Bound. Expected value between 0 and 1.')

        if not isinstance(label, (int,)):
            raise ValueError(f'variable label contains value of incorrect datatype.\
                Expected float found {isinstance(label)}')

    def add_sample(self, prob: float, label: int) -> None:

        self.check_values(prob, label)

        self.samples.append((prob, label))

    def compute_calibrated_score(self) -> None:
        self.calibrated_score = sum(label for prob, label in self.samples) / len(self.samples)


class HistogramBinningCalibrator:

    def __init__(self, probs: list, labels: list, partition_scheme: str = 'mass'):
        self.probs = probs
        self.labels = labels
        self.partition_scheme = partition_scheme

        # check for errors
        self.check_values(probs, labels)
        self.samples = [(prob, label) for prob, label in zip(probs, labels)]
        self.bins = {}

    @staticmethod
    def check_values(prob: list, label: list):
        if len(prob) != len(label):
            raise ValueError(f'Size mismatch. prob contains {len(prob)} elements \
                label contains {len(label)} element')

    def fit(self):

        # create partition
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
