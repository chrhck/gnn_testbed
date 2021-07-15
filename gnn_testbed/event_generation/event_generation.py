"""Event Generators."""
import logging

import awkward as ak
from numba.typed import List
import numpy as np
from tqdm.auto import tqdm

from .constants import Constants
from .detector import (
    sample_cylinder_volume,
    sample_cylinder_surface,
    sample_direction,
    generate_noise,
    trigger,
)

from .mc_record import MCRecord
from .photon_propagation import PhotonSource, dejit_sources, generate_photons

logger = logging.getLogger(__name__)


def generate_cascade(det, pos, t0, energy=None, n_photons=None, d0=33, seed=31337):
    """
    Generate a single cascade with given amplitude and position and return time of detected photons.

    Parameters:
      det: Detector
        Instance of Detector class
      pos: np.ndarray
        Position (x, y, z) of the cascade
      t0: float
        Time of the cascade
      energy: float
        Energy of the cascade
      d0: Decay constant for photon absorption [m]
      seed: int
    """

    if energy is None and n_photons is None:
        raise RuntimeError("Set either energy or n_photons")
    if energy is not None:
        n_photons = energy * Constants.photons_per_GeV

    source = PhotonSource(pos, n_photons, t0)
    source_list = [source]
    record = MCRecord(
        "cascade", dejit_sources(source_list), {"energy": energy, "position": pos}
    )
    hit_times = ak.sort(
        ak.Array(
            generate_photons(
                det.module_coords, det.module_efficiencies, List(source_list), seed=seed
            )
        )
    )
    return hit_times, record


def generate_cascades(
    det,
    height,
    radius,
    nsamples,
    d0=33,
    seed=31337,
):
    """Generate a sample of cascades, randomly sampling the positions in a cylinder of given radius and length."""
    rng = np.random.RandomState(seed)

    events = []
    records = []
    i = 0
    pbar = tqdm(total=nsamples)
    while i < nsamples:
        pos = sample_cylinder_volume(height, radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(2, 5))

        event, record = generate_cascade(
            det, pos, 0, energy=energy, d0=d0, seed=seed + i
        )
        if ak.count(event) == 0:
            continue
        time_range = [
            ak.min(ak.flatten(event)) - 1000,
            ak.max(ak.flatten(event)) + 5000,
        ]
        noise = generate_noise(det, time_range)
        event = ak.sort(ak.concatenate([event, noise], axis=1))

        if trigger(det, event):
            events.append(event)
            records.append(record)
            i += 1
            pbar.update()
    return events, records


def generate_realistic_track(
    det,
    pos,
    direc,
    track_len,
    energy,
    t0=0,
    res=10,
    seed=31337,
    rng=np.random.RandomState(31337),
    propagator=None,
):
    """
    Generate a realistic track using energy losses from PROPOSAL.

    Parameters:
      det: Detector
        Instance of Detector class
      pos: np.ndarray
        Position (x, y, z) of the track at t0
      direc: np.ndarray
        Direction (dx, dy, dz) of the track
      track_len: float
        Length of the track
      energy: float
        Initial energy of the track
      t0: float
        Time at position `pos`
      seed: int
      rng: RandomState
      propagator: Proposal propagator
    """
    try:
        import proposal as pp
    except ImportError as e:
        logger.critical("Could not import proposal!")
        raise e

    sources = []

    if propagator is None:
        raise RuntimeError()

    init_state = pp.particle.ParticleState()
    init_state.energy = energy * 1e3  # initial energy in MeV
    init_state.position = pp.Cartesian3D(0, 0, 0)
    init_state.direction = pp.Cartesian3D(0, 0, 1)
    track = propagator.propagate(init_state, track_len * 100)  # cm

    # harvest losses
    for loss in track.stochastic_losses():
        dist = loss.position.z / 100
        e_loss = loss.energy / 1e3
        p = pos + dist * direc
        t = dist / Constants.c_vac + t0

        if np.linalg.norm(p) > det.outer_radius + 3 * Constants.lambda_abs:
            continue
        sources.append(PhotonSource(p, e_loss * Constants.photons_per_GeV, t))

    if not sources:
        return None

    record = MCRecord(
        "realistic_track",
        dejit_sources(sources),
        {
            "position": pos,
            "energy": energy,
            "track_len": track.track_propagated_distances()[-1] / 100,
            "direction": direc,
        },
    )
    hit_times = ak.sort(
        ak.Array(
            generate_photons(
                det.module_coords, det.module_efficiencies, List(sources), seed=seed
            )
        )
    )
    return hit_times, record


def generate_realistic_tracks(
    det,
    height,
    radius,
    nsamples,
    seed=31337,
    propagator=None,
):
    """Generate realistic muon tracks."""
    rng = np.random.RandomState(seed)
    # Safe length to that tracks will appear infinite
    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    events = []
    records = []

    pbar = tqdm(total=nsamples)
    i = 0
    while i < nsamples:
        pos = sample_cylinder_surface(height, radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(2, 6, size=1))
        direc = sample_direction(1, rng).squeeze()

        # shift pos back by half the length:
        pos = pos - track_length / 2 * direc

        result = generate_realistic_track(
            det,
            pos,
            direc,
            track_length,
            energy=energy,
            t0=0,
            seed=seed + i,
            rng=rng,
            propagator=propagator,
        )
        if result is None:
            continue
        event, record = result
        if ak.count(event) == 0:
            continue
        time_range = [
            ak.min(ak.flatten(event)) - 1000,
            ak.max(ak.flatten(event)) + 5000,
        ]
        noise = generate_noise(det, time_range)
        event = ak.sort(ak.concatenate([event, noise], axis=1))

        if trigger(det, event):
            events.append(event)
            records.append(record)
            i += 1
            pbar.update()

    return events, records


def generate_realistic_starting_tracks(
    det,
    height,
    radius,
    nsamples,
    seed=31337,
    propagator=None,
):
    """Generate realistic starting tracks (cascade + track)."""
    rng = np.random.RandomState(seed)
    # Safe length to that tracks will appear infinite
    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    events = []
    records = []

    i = 0
    pbar = tqdm(total=nsamples)
    while i < nsamples:
        pos = sample_cylinder_volume(height, radius, 1, rng).squeeze()
        energy = np.power(10, rng.uniform(2, 6))
        direc = sample_direction(1, rng).squeeze()
        inelas = rng.uniform(1e-6, 1 - 1e-6)

        result = generate_realistic_track(
            det,
            pos,
            direc,
            track_length,
            energy=energy * inelas,
            t0=0,
            seed=seed + i,
            rng=rng,
            propagator=propagator,
        )
        if result is None:
            continue
        event, record = result

        # Generate the initial cascade
        event_c, record_c = generate_cascade(
            det, pos, 0, energy * (1 - inelas), seed=seed + i
        )
        event = ak.concatenate([event, event_c], axis=1)
        record = record + record_c
        if ak.count(event) == 0:
            continue

        time_range = [
            ak.min(ak.flatten(event)) - 1000,
            ak.max(ak.flatten(event)) + 5000,
        ]
        noise = generate_noise(det, time_range)
        event = ak.sort(ak.concatenate([event, noise], axis=1))

        if trigger(det, event):
            events.append(event)
            records.append(record)
            i += 1
            pbar.update()

    return events, records


def generate_uniform_track(
    det, pos, direc, track_len, energy, t0=0, res=10, seed=31337
):
    """
    Generate a track approximated by cascades at fixed intervals.

    Parameters:
      det: Detector
        Instance of Detector class
      pos: np.ndarray
        Position (x, y, z) of the track at t0
      direc: np.ndarray
        Direction (dx, dy, dz) of the track
      track_len: float
        Length of the track
      energy: float
        Energy of each individual cascade
      t0: float
        Time at position `pos`
      res: float
        Distance of cascades along the track [m]
      seed: float

    """
    sources = []

    for i in np.arange(0, track_len, res):
        p = pos + i * direc

        # Check if this position is way outside of the detector.
        # In that case: ignore

        if np.linalg.norm(p) > det.outer_radius + 3 * Constants.lambda_abs:
            continue

        t = i / Constants.c_vac + t0
        sources.append(PhotonSource(p, energy * Constants.photons_per_GeV, t))

    record = MCRecord(
        "uniform_track",
        dejit_sources(sources),
        {"position": pos, "energy": energy, "track_len": track_len, "direction": direc},
    )
    hit_times = ak.sort(
        ak.Array(
            generate_photons(
                det.module_coords, det.module_efficiencies, List(sources), seed=seed
            )
        )
    )
    return hit_times, record


def generate_uniform_tracks(
    det,
    height,
    radius,
    nsamples,
    seed=31337,
):
    rng = np.random.RandomState(seed)
    # Safe length to that tracks will appear infinite
    # TODO: Calculate intersection with generation cylinder
    track_length = 3000

    positions = sample_cylinder_surface(height, radius, nsamples, rng)
    directions = sample_direction(nsamples, rng)

    # Sample amplitude uniform in log
    amplitudes = np.power(10, rng.uniform(0, 4, size=nsamples))

    events = []
    records = []

    for i, (pos, amp, direc) in enumerate(zip(positions, amplitudes, directions)):

        # shift pos back by half the length:
        pos = pos - track_length / 2 * direc

        event, record = generate_uniform_track(
            det, pos, direc, track_length, amp, 0, 10, seed + i
        )
        if ak.count(event) == 0:
            continue
        time_range = [
            ak.min(ak.flatten(event)) - 1000,
            ak.max(ak.flatten(event)) + 5000,
        ]
        noise = generate_noise(det, time_range)
        event = ak.sort(ak.concatenate([event, noise], axis=1))
        events.append(event)
        records.append(record)
    return events, records
