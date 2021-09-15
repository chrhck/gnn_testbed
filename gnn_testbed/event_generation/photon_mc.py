import jax.numpy as jnp
from jax import jit, random, vmap
from jax.lax import cond, fori_loop
import numpy as np
import jax


def _photon_sphere_intersection(photon_x, photon_p, target_x, target_r, step_size):
    p_normed = photon_p  # assume normed / photon_p_n

    a = jnp.dot(p_normed, (photon_x - target_x))
    b = a ** 2 - (jnp.linalg.norm(photon_x - target_x) ** 2 - target_r ** 2)

    isected = (b >= 0) & ((-a - jnp.sqrt(b)) > 0) & ((-a - jnp.sqrt(b)) < step_size)

    result = cond(
        isected,
        lambda _: (True, photon_x + (-a - jnp.sqrt(b)) * p_normed),
        lambda _: (False, jnp.ones(3) * 1e8),
        0,
    )

    return result


photon_sphere_intersection = jit(_photon_sphere_intersection, static_argnums=[3])


def scattering_function(subkey, g=0.9):
    # henyey greenstein scattering in one plane
    eta = random.uniform(subkey)
    costheta = (
        1 / (2 * g) * (1 + g ** 2 - ((1 - g ** 2) / (1 + g * (2 * eta - 1))) ** 2)
    )
    return jnp.arccos(costheta)


def cherenkov_ang_dist(costheta):
    # https://arxiv.org/pdf/1210.5140.pdf
    # params for e-
    n = 1.35
    a = 4.27033
    b = -6.02527
    c = 0.29887
    cos_theta_c = 1 / n
    d = -0.00103
    return a * jnp.exp(b * jnp.abs(costheta - cos_theta_c) ** c) + d


def sample_cherenkov_ang_dist(nsamples, key):
    vals = []
    cnt = 0
    result = jnp.empty(nsamples, dtype=jnp.float32)
    a = 4.27033
    d = -0.00103
    max_val = a - d

    while cnt < nsamples:
        key, subkey = random.split(key)
        ct = 2 * random.uniform(subkey) - 1
        key, subkey = random.split(key)
        surv = max_val * random.uniform(subkey)

        if surv < cherenkov_ang_dist(ct):
            result = result.at[cnt].set(ct)
            cnt += 1
    return result


def make_step(sca, c_medium):
    def step(pos, dir, time, key):
        k1, k2, k3, k4 = random.split(key, 4)

        eta = random.uniform(k1)
        step_size = -jnp.log(eta) / sca

        new_pos = pos + step_size * dir
        new_time = time + step_size / c_medium

        theta = scattering_function(k2)
        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        phi = random.uniform(k3, minval=0, maxval=2 * np.pi)
        cos_phi = jnp.cos(phi)
        sin_phi = jnp.sin(phi)

        px, py, pz = dir

        is_para_z = jnp.abs(pz) == 1
        new_dir = cond(
            is_para_z,
            lambda _: jnp.array(
                [
                    sin_theta * cos_phi,
                    jnp.sign(pz) * sin_theta * sin_phi,
                    jnp.sign(pz) * cos_theta,
                ]
            ),
            lambda _: jnp.array(
                [
                    (px * cos_theta)
                    + (
                        (sin_theta * (px * pz * cos_phi - py * sin_phi))
                        / (jnp.sqrt(1.0 - pz ** 2))
                    ),
                    (py * cos_theta)
                    + (
                        (sin_theta * (py * pz * cos_phi + px * sin_phi))
                        / (jnp.sqrt(1.0 - pz ** 2))
                    ),
                    (pz * cos_theta) - (sin_theta * cos_phi * jnp.sqrt(1.0 - pz ** 2)),
                ]
            ),
            None,
        )

        return new_pos, new_dir, new_time, k4

    return step


def sph_to_cart(theta, phi=0, r=1):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return jnp.array([x, y, z])


abs_len = 30.0
sca_len = 90
key = random.PRNGKey(0)
c_medium = 0.3 / 1.35
step_fun = make_step(1 / sca_len, c_medium)


def step_and_record(i, args):
    pos_rec = args[0]
    dir = args[1]
    time_rec = args[2]
    key, subkey = random.split(args[3])

    new_pos, new_dir, new_time, key = step_fun(pos_rec[i], dir, time_rec[i], subkey)

    pos_rec = pos_rec.at[i + 1].set(new_pos)
    time_rec = time_rec.at[i + 1].set(new_time)

    return pos_rec, new_dir, time_rec, key


def make_n_steps(nsteps, key):
    positions = jnp.empty((nsteps + 1, 3))
    times = jnp.empty(nsteps + 1)

    k1, k2, k3 = random.split(key, 3)
    theta = jnp.arccos(random.uniform(k1, minval=-1, maxval=1))
    phi = random.uniform(k2, minval=0, maxval=2 * np.pi)
    direction = sph_to_cart(theta, phi, r=1)

    positions = positions.at[0].set(jnp.array([0.0, 0.0, 0.0]))
    # direction = jnp.array([0.0, 0.0, 1.0])
    times = times.at[0].set(0.0)

    positions, direction, times, key = fori_loop(
        0, nsteps, step_and_record, (positions, direction, times, k3)
    )
    return positions, times, direction


make_n_steps_v = jit(
    vmap(make_n_steps, in_axes=[None, 0], out_axes=0), static_argnums=[0]
)

psi_v = vmap(photon_sphere_intersection, in_axes=(0, 0, None, None, 0))
psi_vv = vmap(psi_v, in_axes=(0, 0, None, None, 0))


def calc_intersections(positions, times, target_x, target_r, abs_len, sca_scale):
    positions = positions * sca_scale
    times = times * sca_scale
    delta_x = jnp.diff(positions, axis=1)
    step_w = jnp.linalg.norm(delta_x, axis=-1)
    isec, isec_pos = psi_vv(
        positions[:, :-1, :],
        (delta_x / step_w[..., np.newaxis]),
        target_x,
        target_r,
        step_w,
    )

    dist_to_isec = jnp.linalg.norm(positions[:, :-1][isec] - isec_pos[isec], axis=1)
    total_dist = c_medium * times[:, :-1][isec] + dist_to_isec

    weight = jnp.exp(-total_dist / abs_len)

    return total_dist / c_medium, isec, weight


def get_time_amp_pos(key, target_x, target_r, abs_len, sca_scale, n_steps, n_ph):
    positions, times = make_n_steps_v(int(n_steps), random.split(key, num=int(n_ph)))
    isec_times, isec_pos, weights = calc_intersections(
        positions, times, target_x, target_r, abs_len, sca_scale
    )
    return isec_times, weights


key = random.PRNGKey(0)
isec_times = []
weights = []

from time import time

r = 10

thetas = np.linspace(0, np.pi, 50)

nph = int(1e6)


# for theta in thetas:

fp = np.memmap("photons.dat", dtype="float32", mode="w+", shape=(int(1e7), 15, 4))
fpd = np.memmap("photon_dirs.dat", dtype="float32", mode="w+", shape=(int(1e7), 3))
jax.profiler.start_trace("/tmp/tensorboard")

for i in range(10):
    key, subkey = random.split(key)
    start = time()
    positions, times, directions = make_n_steps_v(int(15), random.split(key, num=nph))

    fp[i * nph : (i + 1) * nph, :, :3] = np.asarray(positions[:, 1:, :])
    fp[i * nph : (i + 1) * nph, :, 3] = np.asarray(times[:, 1:])
    fpd[i * nph : (i + 1) * nph] = np.asarray(directions)
jax.profiler.stop_trace()
