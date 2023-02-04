from src.pyModelCalibrate.model.utils import get_uniform_mass_partitions,\
                                            get_uniform_width_partitions,\
                                            get_uniform_num_partitions,\
                                            Bin


def ece(probs: list, labels: list, partition_scheme: str, **kwargs) -> float:
    samples = [(prob, label) for prob, label in zip(probs, labels)]
    sorted(samples, key=lambda x: x[0])

    probs, labels, part_ids = None, None, None
    if partition_scheme == 'mass':
        probs, labels, part_ids = get_uniform_mass_partitions(samples=samples,
                                                              partition_size=kwargs)
    elif partition_scheme == 'width':
        probs, labels, part_ids = get_uniform_width_partitions(samples=samples,
                                                               width=kwargs)
    else:
        probs, labels, part_ids = get_uniform_num_partitions(samples=samples,
                                                             partition_num=kwargs)

    bins = []
    # learn bin wise statistics
    for part_id in set(part_ids):
        _bin = Bin(bin_id=part_id)
        bins[part_id] = _bin

    for idx, part_id in enumerate(part_ids):
        bins[part_id].add_sample(prob=probs[idx],
                                      label=labels[idx])

    ece = 0

    for _, bin in bins.items():
        sum_clicks = sum(bin.labels)
        sum_probs = bin.avg_prob
        ece += abs(sum_clicks-sum_probs)**2

    return ece/len(probs)

