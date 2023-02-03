from statistics import mean


def strict_lower_bound(x: float, arr: list) -> int:
    start = 0
    end = len(arr) - 1

    while start <= end:
        mid = (start + end) // 2
        if arr[mid] < x:
            start = mid + 1
        else:
            end = mid - 1

    return end


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

    def compute_bin_statistics(self) -> None:
        probs, labels = zip(*self.samples)

        self.low_prob, self.high_prob = min(probs), max(probs)

        self.avg_prob, self.calibrated_score = mean(probs), mean(labels)

    def merge(self, bin_a):
        self.samples += bin_a.samples
        self.compute_bin_statistics()
