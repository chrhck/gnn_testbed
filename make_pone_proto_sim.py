import pickle
import os
import numpy as np
from gnn_testbed.event_generation import (
    make_line,
    Detector,
    proposal_setup,
    generate_cascades,
    generate_realistic_tracks,
    generate_realistic_starting_tracks,
)

outpath = "."

rng = np.random.RandomState(31338)
modules = make_line(0, 0, 30, 10, rng, 1e-6, 0)
modules += make_line(100, 0, 30, 10, rng, 1e-6, 0)
det = Detector(modules)

seed = 31337

prop = proposal_setup()
n_events = 100
cascades, cascade_records = generate_cascades(det, 900, 750, n_events, seed=seed)
pickle.dump(
    (cascades, cascade_records),
    open(os.path.join(outpath, "training_data_cascades_pone.pickle"), "wb"),
)
tracks, track_records = generate_realistic_tracks(
    det, 900, 750, n_events, seed=seed, propagator=prop
)
pickle.dump(
    (tracks, track_records),
    open(os.path.join(outpath, "training_data_tracks_pone.pickle"), "wb"),
)
stracks, strack_records = generate_realistic_starting_tracks(
    det, 900, 750, n_events, seed=seed, propagator=prop
)
pickle.dump(
    (stracks, strack_records),
    open(os.path.join(outpath, "training_data_stracks_pone.pickle"), "wb"),
)
