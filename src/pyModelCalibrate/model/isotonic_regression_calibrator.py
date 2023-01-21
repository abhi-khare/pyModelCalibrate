from statistics import mean
from partition import *
from utils import strict_lower_bound


class Bin:

    def __init__(self, bin_id: int) -> None:
        self.bin_id: int = bin_id  # A unique integer ID for the bin
        self.samples: list[tuple] = []  # A list of tuple containing the input (probability) and the output (label)
        self.calibrated_score = None  # Actual proportion of samples that belongs to class 1
        self.low_prob = None  # lowest probability score in the sample set.
        self.high_prob = None  # highest probability score in the sample set.
        self.avg_prob = None  # average of the probability scores of this sample set.

    def add_sample(self, prob: float, label: int) -> None:
        """
        This function takes input X and label y and add them to an existing list of
        samples.
        Parameters:
        - prob: A single instance of train set : uncalibrated probability score
        - label: A single instance of train set : label

        Returns: None
        """

        self.samples.append((prob, label))

    def compute_statistics(self) -> None:
        probs, labels = zip(*self.samples)

        self.low_prob, self.high_prob = min(probs), max(probs)

        self.avg_prob, self.calibrated_score = mean(probs), mean(labels)

    def merge(self, bin_a):
        self.samples += bin_a.samples
        self.compute_statistics()


class IsotonicRegressionCalibrator:

    def __init__(self, probs: list, labels: list, partition_scheme: str = 'mass'):

        # check for errors
        self.check_errors(probs, labels, partition_scheme)

        self.probs = probs
        self.labels = labels
        self.partition_scheme = partition_scheme

        self.samples = [(prob, label) for prob, label in zip(probs, labels)]
        self.bins = {}

        # model params
        self.low_array = []
        self.calib_array = []

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

    @staticmethod
    def monotonic_smoothing(bins: list) -> list:

        stack = []
        for bin in bins:

            if len(stack) == 0:
                stack.append(bin)
            else:
                top_bin = stack[-1]
                current_bin = bin

                while top_bin.calibrated_score > current_bin.calibrated_score:
                    current_bin.merge(top_bin)
                    stack.pop()

                stack.append(current_bin)

        return stack

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

        processed_bins = self.monotonic_smoothing(self.bins)

        # compute low_bin and calib_score array. This will be used during inference.
        for _, bin in processed_bins.items():
            self.low_array.append(bin.low_prob)
            self.calib_array.append(bin.calibrated_score)

    @staticmethod
    def find_bin_id(self, prob):

        idx = strict_lower_bound(prob, self.low_array)
        if idx == -1:
            return self.calib_array[0]
        else:
            return self.calib_array[idx]

    def predict(self, probs):

        calibrated_probs = [self.find_bin_id(prob) for prob in probs]

        return calibrated_probs
