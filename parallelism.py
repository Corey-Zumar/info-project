import math
import itertools
import numpy as np

def get_worker_range_pairs(node_ids, num_workers):
    num_intervals = max(1, int(4 * math.sqrt(num_workers)))
    interval_length = int(len(node_ids) / num_intervals)
    idxs = np.cumsum([0] + [interval_length for _ in range(num_intervals - 1)])
    idxs = np.append(idxs, len(node_ids))
    ranges = []
    for i in range(len(idxs) - 1):
        r_low = idxs[i]
        r_high = idxs[i + 1]
        ranges.append((r_low, r_high))

    range_pairs = [(r1, r2) for r1, r2 in itertools.product(ranges, ranges)]
    unique_set = set()
    new_pairs = []
    for pair in range_pairs:
        unique_key = frozenset(pair)
        if unique_key not in unique_set:
            new_pairs.append(pair)
            unique_set.add(unique_key)

    range_pairs = new_pairs

    return range_pairs

