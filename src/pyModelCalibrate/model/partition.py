"""
This file contains implementation of following uniform partition scheme:
1. Uniform mass partition
2. Uniform width partition
3. equal bin count partition
"""


def get_uniform_mass_partitions(samples: list, partition_size: int, decreasing: bool = False) -> tuple:
    """
    Given a list of probabilities and the corresponding class labels, this function partition the probabilities
    into `partition_size` equal-mass partitions (i.e. each bin contains equal number of samples)
    and return a tuple containing the probabilities, labels, and partition IDs for each sample. 
    The samples are sorted in increasing order by default, but this can be changed by setting 
    the `decreasing` parameter to False.

    Parameters:
    - samples (list[tuple]): A list of tuple containing probability score and the label.
    - partition_size (int): Number of samples in each bin.
    - decreasing (bool): A flag indicating whether to sort the probabilities and labels in decreasing order. Default is
                        False.

    Returns:
    - tuple: A tuple containing the probabilities, labels and the partition IDs.
    """

    samples.sort(key=lambda x: x[0], reverse=decreasing)
    sorted_probs, sorted_labels = zip(*samples)

    partition_ids = [int(_iter / partition_size) for _iter in range(len(samples))]

    return sorted_probs, sorted_labels, partition_ids


def get_uniform_num_partitions(samples: list, partition_num: int, decreasing: bool = False) -> tuple:
    """
    Given a list of probabilities and the corresponding class labels, this function partition the probabilities
    into `partition_num` partitions and return a tuple containing the probabilities, labels,
    and partition IDs for each sample. This scheme also results in equal number of samples in each bin, just that
    we can control the sample size using the partition count. The samples are sorted in increasing order by default,
    but this can be changed by setting the `decreasing` parameter to False.

    Parameters:
    - samples (list[tuple]): A list of tuple containing probability and the label.
    - partition_num (int): Number of bins
    - decreasing (bool): A flag indicating whether to sort the probabilities and labels in decreasing order. Default is
                        False.

    Returns:
    - tuple: A tuple containing the probabilities, labels and the partition IDs.
    """

    samples.sort(key=lambda x: x[0], reverse=decreasing)
    sorted_probs, sorted_labels = zip(*samples)

    partition_ids = [int(_iter / len(samples))*partition_num for _iter in range(len(samples))]

    return sorted_probs, sorted_labels, partition_ids


def get_uniform_width_partitions(samples: list, width: float = None,
                                 decreasing: bool = False) -> tuple:
    """
    Given a list of probabilities and the corresponding class labels, this function partition the probabilities
    into equal-width partitions(i.e. each partition has equal width in terms of probability range) 
    and return a tuple containing the probabilities, labels, and partition IDs
    for each sample. The samples are sorted in increasing order by default, but this can be
    changed by setting the `decreasing` parameter to False.

    By default, it uses width parameter if both width and partition_num is provided, otherwise it computes width if only
    partition_num is provided.

    Parameters:
    - samples (list[tuple]): A list of tuple containing probability and the label.
    - width (float): width of each partition
    - decreasing (bool): A flag indicating whether to sort the probabilities and labels in decreasing order. Default is
                        False.

    Returns:
    - tuple: A tuple containing the sorted probabilities, sorted labels, and partition IDs for each probability.
    """

    samples.sort(key=lambda x: x[0], reverse=decreasing)

    sorted_probs, sorted_labels = zip(*samples)

    # Compute the minimum and maximum probability values
    min_prob = min(sorted_probs)
    partition_width = width

    partition_ids = [int((prob - min_prob) / partition_width) for prob in sorted_probs]

    return sorted_probs, sorted_labels, partition_ids
