"""_summary_
This file contains implementation of following uniform partition scheme:
1. Uniform mass partition
2. Uniform width partition
3. equal bin count partition
"""


def get_uniform_mass_partitions(samples: list, partition_num: int, decreasing: bool = False) -> tuple:
    """
    Given a list of probabilities and the corresponding class labels, this function partition the probabilities
    into `partition_num` equal-mass partitions and return a tuple containing the sorted probabilities, sorted labels,
    and partition IDs for each sample. The probabilities and labels are sorted in increasing order by default,
    but this can be changed by setting the `decreasing` parameter to False.

    Parameters:
    - samples (list[tuple]): A list of tuple containing probability and the label.
    - partition_num (int): The number of equal-mass partitions to create.
    - decreasing (bool): A flag indicating whether to sort the probabilities and labels in decreasing order. Default is
                        False.

    Returns:
    - tuple: A tuple containing the sorted probabilities, sorted labels, and partition IDs for each probability.
    """

    samples.sort(key=lambda x: x[0], reverse=decreasing)
    sorted_probs, sorted_labels = zip(*samples)

    partition_ids = [int(_iter / partition_num) for _iter in range(len(samples))]

    return sorted_probs, sorted_labels, partition_ids


def get_uniform_width_partitions(samples: list, width: float = None, partition_num: int = None,
                                 decreasing: bool = False) -> tuple:
    """
    Given a list of probabilities and the corresponding class labels, this function partition the probabilities
    into equal-width partitions and return a tuple containing the sorted probabilities, sorted labels, and partition IDs
    for each sample. The probabilities and labels are sorted in increasing order by default, but this can be
    changed by setting the `decreasing` parameter to False.

    By default, it uses width parameter if both width and partition_num is provided, otherwise it computes width if only
    partition_num is provided.

    Parameters:
    - samples (list[tuple]): A list of tuple containing probability and the label.
    - partition_num (int): The number of equal-width partitions to create.
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
    max_prob = max(sorted_probs)

    # Compute the width of each partition
    if width is None & partition_num is None:
        raise ValueError(
            "Either pass width of the partition or the number of partitions in which samples set needs to be / "
            "partitioned")
    elif width is None:
        partition_width = (max_prob - min_prob) / partition_num
    else:
        partition_width = width

    partition_ids = [int((prob - min_prob) / partition_width) for prob in sorted_probs]

    return sorted_probs, sorted_labels, partition_ids
