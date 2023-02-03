from statistics import mean
from .partition import *
from .utils import strict_lower_bound, Bin


class HistogramBinningCalibrator:

    def __init__(self, probs: list, labels: list, partition_scheme: str = 'count', **kwargs):

        # check for errors
        self.check_errors(probs, labels, partition_scheme, kwargs)

        self.partition_scheme = partition_scheme
        self.params = {key: param for key, param in kwargs.items()}

        self.samples = [(prob, label) for prob, label in zip(probs, labels)]
        self.bins = {}

        # model params
        self.low_array = []
        self.calib_array = []

    @staticmethod
    def check_errors(probs: list, labels: list, partition_scheme: str, kwargs: dict):

        assert partition_scheme in ("mass", "width", "count"), \
            "valid partition schemes: mass, width and count"

        for keys in kwargs.keys():
            assert keys in ("partition_size", "width", "partition_num"), \
                "Invalid parameter"

        assert len(probs) == len(labels), \
            f'Size mismatch. probs array contains {len(probs)} elements \
                label array contains {len(labels)} element'

        assert len(probs) != 0, f"probs array must contain atleast 1 element"

        assert len(labels) != 0, f"label array must contain atleast 1 element"

        for prob in probs:
            if not isinstance(prob, (float,)):
                raise ValueError(f"variable prob contains value of incorrect datatype."
                                 f"Expected float found {type(prob)}")

            if not 0 <= prob <= 1:
                raise ValueError(f"Value out of Bound. Expected value between 0 and 1.")

        for label in labels:
            if not isinstance(label, (int,)):
                raise ValueError(f"variable label contains value of incorrect datatype."
                                 f"Expected float found {type(label)}")

    def fit(self):

        # create partition
        probs, labels, part_ids = None, None, None
        if self.partition_scheme == 'mass':
            probs, labels, part_ids = get_uniform_mass_partitions(samples=self.samples,
                                                                  partition_size=self.params['partition_size'])
        elif self.partition_scheme == 'width':
            probs, labels, part_ids = get_uniform_width_partitions(samples=self.samples,
                                                                   width=self.params['width'])
        else:
            probs, labels, part_ids = get_uniform_num_partitions(samples=self.samples,
                                                                 partition_num=self.params['partition_num'])

        # learn bin wise statistics
        for part_id in set(part_ids):
            _bin = Bin(bin_id=part_id)
            self.bins[part_id] = _bin

        for idx, part_id in enumerate(part_ids):
            self.bins[part_id].add_sample(prob=probs[idx],
                                          label=labels[idx])

        # compute calibrated score for bins
        for _, bin in self.bins.items():
            bin.compute_bin_statistics()

        # compute low_bin and calib_score array. This will be used during inference.
        for _, bin in self.bins.items():
            self.low_array.append(bin.low_prob)
            self.calib_array.append(bin.calibrated_score)

    def find_bin_id(self, prob):

        idx = strict_lower_bound(prob, self.low_array)
        if idx == -1:
            return self.calib_array[0]
        else:
            return self.calib_array[idx]

    def predict(self, probs):

        calibrated_probs = [self.find_bin_id(prob) for prob in probs]
        return calibrated_probs

    def get_model(self):
        return self.low_array, self.calib_array
