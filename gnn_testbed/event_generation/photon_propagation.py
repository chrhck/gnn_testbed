"""Implements photon propagation."""
from numba import float64, jit
from numba.experimental import jitclass
import numpy as np

from .constants import Constants


class PhotonSource_(object):
    """
    A source of photons.

    This class is the pure python prototype for the jitclass `PhotonSource`.
    """

    pos: float64[:]
    amp: float
    t0: float

    def __init__(self, pos, amp, t0):
        """Initialize PhotonSource_."""
        self.pos = pos
        self.amp = amp
        self.t0 = t0


PhotonSource = jitclass(PhotonSource_)


def dejit_sources(sources):
    """Convert numba jitclass PhotonSources into pure python objects."""
    return [PhotonSource_(source.pos, source.amp, source.t0) for source in sources]


@jit(nopython=True)
def generate_photons(
    module_coords,
    module_efficiencies,
    sources,
    c_vac=Constants.c_vac,
    n_gr=Constants.n_gr,
    pandel_lambda=Constants.pandel_lambda,
    theta_cherenkov=Constants.theta_cherenkov,
    pandel_rho=Constants.pandel_rho,
    photocathode_area=Constants.photocathode_area,
    lambda_abs=Constants.lambda_abs,
    lambda_sca=Constants.lambda_sca,
    seed=31337,
):
    """
    Generate photons for a list of sources.

    The amplitude (== number of photon) at each detector module is modeled as exponential decay based on
    the distance to `pos` with decay constant `d0`. The detection process is modelled as poisson process.
    The photon arrival times are modeled using the
    `Pandel`-PDF (https://www.sciencedirect.com/science/article/pii/S0927650507001260), which is a gamma distribution
    with distance-dependent scaling of the shape parameters.
    """
    all_times_det = []
    np.random.seed(seed)

    lambda_p = np.sqrt(lambda_abs * lambda_sca / 3)
    xi = np.exp(-lambda_sca / lambda_abs)
    lambda_c = lambda_sca / (3 * xi)

    for idom in range(module_coords.shape[0]):

        this_times = []
        total_length = 0
        for source in sources:
            dist = np.linalg.norm(source.pos - module_coords[idom])

            """
      # model photon emission as point-like

      detected_flux = source.amp * np.exp(-dist/d0) / (4*np.pi* dist**2)
      detected_photons = detected_flux * photocathode_area

      # from https://arxiv.org/pdf/1311.4767.pdf
      """
            detected_photons = (
                source.amp
                * photocathode_area
                / (4 * np.pi)
                * np.exp(-dist / lambda_p)
                * 1
                / (lambda_c * dist * np.tanh(dist / lambda_c))
            )

            amp_det = np.random.poisson(detected_photons * module_efficiencies[idom])

            time_geo = dist / (c_vac / n_gr) + source.t0
            pandel_xi = dist / (pandel_lambda * np.sin(theta_cherenkov))

            times_det = (
                np.random.gamma(pandel_xi, scale=1 / pandel_rho, size=amp_det)
                + time_geo
            )
            this_times.append(times_det)
            total_length += amp_det

        this_times_arr = np.empty(total_length)
        i = 0
        for tt in this_times:
            this_times_arr[i : i + tt.shape[0]] = tt  # noqa: E203
            i += tt.shape[0]

        all_times_det.append(this_times_arr)

    return all_times_det
