"""Collection of feature extraction functions."""
import awkwards as ak
import numpy as np


def get_features(det, hit_times, reference="median_global"):
    """Calculate standard features: Percentiles, Amplitude and module positions."""
    if reference == "median_global":

        sorted_times = np.sort(ak.flatten(hit_times).to_numpy())
        reference = sorted_times[int(0.5 * sorted_times.shape[0])]

    else:
        raise NotImplementedError()

    n_total = len(hit_times)
    hits = ak.count(hit_times, axis=1) > 0
    hit_times = hit_times[hits]
    hits = hits.to_numpy()

    percentiles = np.linspace(0.1, 0.9, num=9)
    features = np.empty((n_total, len(percentiles) + 6))
    # mean = ak.mean(hit_times, axis=1) - reference

    for i, p in enumerate(percentiles):
        features[hits, i] = (
            ak.flatten(
                hit_times[
                    ak.from_regular(
                        ak.values_astype(
                            ak.count(hit_times, axis=1) / (1 / p), np.int64
                        )[:, None],
                        1,
                    )
                ],
                axis=1,
            )
            - reference
        ) / 1e3

    features[hits, len(percentiles)] = (ak.firsts(hit_times) - reference) / 1e3
    features[hits, len(percentiles) + 1] = np.log10(ak.count(hit_times, axis=1))
    features[hits, len(percentiles) + 2] = (ak.std(hit_times, axis=1)) / 1e3
    features[hits, len(percentiles) + 3] = det.module_coords_ak[hits, 0] / 100
    features[hits, len(percentiles) + 4] = det.module_coords_ak[hits, 1] / 100
    features[hits, len(percentiles) + 5] = det.module_coords_ak[hits, 2] / 100

    features[~hits, :] = np.nan

    return features
