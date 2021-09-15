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
oms_per_line = 20
dist_z = 50

modules = make_line(-75, -65, dist_z, oms_per_line, rng, 16 * 1e-6, 0, efficiency=0.3)
modules += make_line(75, -65, dist_z, oms_per_line, rng, 16 * 1e-6, 0, efficiency=0.3)
modules += make_line(0, 65, dist_z, oms_per_line, rng, 16 * 1e-6, 0, efficiency=0.3)
det = Detector(modules)

height = 1500
radius = 200

seed = 31337

pprop_kwargs = dict(
    photocathode_area=16 * (7.62 / 2) ** 2 * np.pi * 1e-4, lambda_abs=30, lambda_sca=100
)


prop = proposal_setup()
"""
n_events = 1000
cascades, cascade_records = generate_cascades(
    det,
    height,
    radius,
    n_events,
    seed=seed,
    log_emin=1,
    log_emax=7,
    pprop_extras=pprop_kwargs,
)
pickle.dump(
    (cascades, cascade_records),
    open(os.path.join(outpath, "training_data_cascades_pone.pickle"), "wb"),
)
"""
n_events = 50000
tracks, track_records = generate_realistic_tracks(
    det,
    height,
    radius,
    n_events,
    seed=seed,
    propagator=prop,
    log_emin=1,
    log_emax=7,
    pprop_extras=pprop_kwargs,
)
pickle.dump(
    (tracks, track_records),
    open(os.path.join(outpath, "training_data_tracks_pone.pickle"), "wb"),
)
"""
n_events = 1000

stracks, strack_records = generate_realistic_starting_tracks(
    det,
    height,
    radius,
    n_events,
    seed=31337,
    propagator=prop,
    log_emin=1,
    log_emax=7,
    pprop_extras=pprop_kwargs,
)
pickle.dump(
    (stracks, strack_records),
    open(os.path.join(outpath, "training_data_stracks_pone.pickle"), "wb"),
)
"""
