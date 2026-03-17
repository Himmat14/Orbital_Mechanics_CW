# Orbital Mechanics Coursework - AERO70016
# Author: Himmat Kaul
# Date: 2026-03-06
# AURORA Mission - Special Perturbations (Cowell's Method)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MU      = 3.986004418e14   # Earth GM [m^3/s^2]
RE      = 6.378137e6       # Earth equatorial radius [m]
J2      = 1.08262668e-3    # J2 coefficient
OMEGA_E = 7.2921150e-5     # Earth rotation rate [rad/s]
AU      = 1.495978707e11   # Astronomical Unit [m]
P_SRP   = 4.56e-6          # Solar radiation pressure at 1 AU [N/m^2]
MU_SUN  = 1.32712440018e20 # Sun GM [m^3/s^2]
MU_MOON = 4.9048695e12     # Moon GM [m^3/s^2]
G0      = 9.80665          # Standard gravity [m/s^2]

# ─────────────────────────────────────────────
# AURORA SPACECRAFT INITIAL CONDITIONS (1 Jan 2026 12:00:00 UTC)
# ─────────────────────────────────────────────
A_SC  = 26600e3  # semi-major axis [m]
E_SC  = 0.7      # eccentricity
I_SC  = np.radians(116.6)  # inclination [rad]
OM_SC = np.radians(115.0)  # RAAN [rad]
W_SC  = np.radians(90.0)   # argument of perigee [rad]
NU_SC = np.radians(180.0)  # true anomaly [rad]

CD = 2.2    # drag coefficient
CR = 1.5    # reflectivity coefficient
A_m = 20.0  # cross-sectional area [m^2]
M_SC = 700.0  # spacecraft mass [kg]

# Epoch: 1 Jan 2026 12:00:00 UTC
EPOCH = datetime(2026, 1, 1, 12, 0, 0)

# ─────────────────────────────────────────────
# GROUND STATIONS
# ─────────────────────────────────────────────
# Malargüe, Argentina
MAL_LAT = np.radians(-35.0)
MAL_LON = np.radians(291.0 - 360.0)  # convert to [-180,180]: -69 deg
MAL_ALT = 0.0

# Kiruna, Sweden
KIR_LAT = np.radians(67.0)
KIR_LON = np.radians(20.0)
KIR_ALT = 0.0

ELEV_MASK = np.radians(0.0)  # elevation mask: visible when above horizon (0 deg per coursework brief)

# Output directory
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────

def kep2cart(a, e, i, Om, w, nu):
    """Convert Keplerian elements to ECI Cartesian state [m, m/s]."""
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))

    # Position in perifocal frame
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.sqrt(MU / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    # Rotation matrix: perifocal -> ECI
    cos_Om, sin_Om = np.cos(Om), np.sin(Om)
    cos_i,  sin_i  = np.cos(i),  np.sin(i)
    cos_w,  sin_w  = np.cos(w),  np.sin(w)

    R = np.array([
        [cos_Om*cos_w - sin_Om*sin_w*cos_i,  -cos_Om*sin_w - sin_Om*cos_w*cos_i,  sin_Om*sin_i],
        [sin_Om*cos_w + cos_Om*sin_w*cos_i,  -sin_Om*sin_w + cos_Om*cos_w*cos_i, -cos_Om*sin_i],
        [sin_w*sin_i,                          cos_w*sin_i,                         cos_i       ]
    ])

    r_eci = R @ r_pf
    v_eci = R @ v_pf
    return np.concatenate([r_eci, v_eci])


def cart2kep(state):
    """Convert ECI Cartesian state to Keplerian elements."""
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = ((v**2 - MU/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU
    e = np.linalg.norm(e_vec)

    energy = v**2/2 - MU/r
    a = -MU / (2 * energy)

    i = np.arccos(h_vec[2] / h)

    Om = np.arctan2(n_vec[1], n_vec[0])
    if Om < 0:
        Om += 2 * np.pi

    if n > 1e-10:
        w = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0:
            w = 2 * np.pi - w
    else:
        w = np.arctan2(e_vec[1], e_vec[0])
        if w < 0:
            w += 2 * np.pi

    nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
    if np.dot(r_vec, v_vec) < 0:
        nu = 2 * np.pi - nu

    return a, e, np.degrees(i), np.degrees(Om), np.degrees(w), np.degrees(nu)


def gmst(t_seconds):
    """Greenwich Mean Sidereal Time [rad] at epoch + t_seconds."""
    # GMST at 1 Jan 2026 12:00:00 UTC
    # Julian date: J2000 epoch = 1 Jan 2000 12:00:00 TT
    # Days from J2000 to epoch
    jd_epoch = 2451545.0  # J2000
    jd_utc = 2451545.0 + (9497.0)  # 1 Jan 2026 12:00 UTC relative to J2000
    # More precise: days from J2000.0 to 2026 Jan 1.5 = 26*365 + 6 (leap years 2000,4,8,12,16,20,24) + 0.5
    # years 2000-2025: 26 years, leap years: 2000,2004,2008,2012,2016,2020,2024 = 7 leaps
    # days = 26*365 + 7 = 9490 + 7 = 9497 days
    T0 = 9497.0 / 36525.0  # Julian centuries from J2000

    # GMST at 1 Jan 2026 12:00 UTC [degrees]
    gmst0_deg = (280.46061837 + 360.98564736629 * 9497.0
                 + 0.000387933 * T0**2 - T0**3/38710000.0) % 360.0

    gmst_rad = np.radians(gmst0_deg) + OMEGA_E * t_seconds
    return gmst_rad % (2 * np.pi)


def eci2ecef(r_eci, t_seconds):
    """Rotate ECI position vector to ECEF."""
    theta = gmst(t_seconds)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    return R @ r_eci


def ecef2lla(r_ecef):
    """Convert ECEF to geodetic lat [rad], lon [rad], alt [m]. (Bowring iteration)"""
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - 0.00669437999014))  # first approx
    for _ in range(5):
        N = RE / np.sqrt(1 - 0.00669437999014 * np.sin(lat)**2)
        lat = np.arctan2(z + 0.00669437999014 * N * np.sin(lat), p)
    N = RE / np.sqrt(1 - 0.00669437999014 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    return lat, lon, alt


# ─────────────────────────────────────────────
# PERTURBATION ACCELERATIONS
# ─────────────────────────────────────────────

def j2_accel(r_vec):
    """J2 perturbation acceleration in ECI [m/s^2]."""
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    factor = -3/2 * J2 * MU * RE**2 / r**5
    zr2 = (z/r)**2
    ax = factor * x * (1 - 5*zr2)
    ay = factor * y * (1 - 5*zr2)
    az = factor * z * (3 - 5*zr2)
    return np.array([ax, ay, az])


def atm_density(alt_m):
    """Exponential atmospheric density model [kg/m^3]. alt in meters."""
    alt_km = alt_m / 1000.0
    # Table: altitude [km], base density [kg/m^3], scale height [km]
    table = np.array([
        [0,    1.225,      8.44],
        [25,   3.899e-2,   6.49],
        [30,   1.774e-2,   6.75],
        [40,   3.972e-3,   7.54],
        [50,   1.057e-3,   8.05],
        [60,   3.206e-4,   7.80],
        [70,   8.770e-5,   7.45],
        [80,   1.905e-5,   7.72],
        [90,   3.396e-6,   7.65],
        [100,  5.297e-7,   7.30],
        [110,  9.661e-8,   7.50],
        [120,  2.438e-8,   8.17],
        [130,  8.484e-9,   9.97],
        [140,  3.845e-9,   12.03],
        [150,  2.070e-9,   14.33],
        [180,  5.464e-10,  21.75],
        [200,  2.789e-10,  27.74],
        [250,  7.248e-11,  44.68],
        [300,  2.418e-11,  64.62],
        [350,  9.518e-12,  85.97],
        [400,  3.725e-12,  107.61],
        [450,  1.585e-12,  133.09],
        [500,  6.967e-13,  164.23],
        [600,  1.454e-13,  225.27],
        [700,  3.614e-14,  290.08],
        [800,  1.170e-14,  376.53],
        [900,  5.245e-15,  495.39],
        [1000, 3.019e-15,  702.81],
        [1500, 2.076e-16, 1204.61],
        [2000, 3.845e-17, 1673.87],
    ])

    if alt_km < 0:
        return table[0, 1]
    if alt_km >= table[-1, 0]:
        # Exponential extrapolation from last layer
        rho0 = table[-1, 1]
        H = table[-1, 2]
        return rho0 * np.exp(-(alt_km - table[-1, 0]) / H)

    idx = np.searchsorted(table[:, 0], alt_km) - 1
    idx = np.clip(idx, 0, len(table) - 2)
    h0, rho0, H = table[idx]
    return rho0 * np.exp(-(alt_km - h0) / H)


def drag_accel(r_vec, v_vec):
    """Atmospheric drag acceleration [m/s^2]. ECI velocity."""
    r = np.linalg.norm(r_vec)
    alt = r - RE

    if alt > 2000e3:
        return np.zeros(3)

    rho = atm_density(alt)

    # Velocity relative to rotating atmosphere
    omega_vec = np.array([0, 0, OMEGA_E])
    v_atm = np.cross(omega_vec, r_vec)
    v_rel = v_vec - v_atm
    v_rel_mag = np.linalg.norm(v_rel)

    if v_rel_mag < 1e-10:
        return np.zeros(3)

    a_drag = -0.5 * CD * (A_m / M_SC) * rho * v_rel_mag**2 * (v_rel / v_rel_mag)
    return a_drag


def sun_position(t_seconds):
    """Approximate Sun position in ECI [m]."""
    # Days from J2000
    d = 9497.0 + t_seconds / 86400.0
    # Mean longitude and mean anomaly
    L = np.radians((280.460 + 0.9856474 * d) % 360)
    g = np.radians((357.528 + 0.9856003 * d) % 360)
    # Ecliptic longitude
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2*g)
    # Obliquity
    eps = np.radians(23.439 - 3.56e-7 * d)
    r_sun = AU * np.array([np.cos(lam), np.cos(eps)*np.sin(lam), np.sin(eps)*np.sin(lam)])
    return r_sun


def moon_position(t_seconds):
    """Approximate Moon position in ECI [m]."""
    d = 9497.0 + t_seconds / 86400.0
    # Simplified lunar position (Meeus simplified)
    L0 = np.radians((218.316 + 13.176396 * d) % 360)
    M  = np.radians((134.963 + 13.064993 * d) % 360)
    F  = np.radians((93.272  + 13.229350 * d) % 360)
    lon = L0 + np.radians(6.289) * np.sin(M)
    lat = np.radians(5.128) * np.sin(F)
    dist = 385001e3 - 20905e3 * np.cos(M)  # [m]
    eps = np.radians(23.439)
    r_moon = dist * np.array([
        np.cos(lat)*np.cos(lon),
        np.cos(eps)*np.cos(lat)*np.sin(lon) - np.sin(eps)*np.sin(lat),
        np.sin(eps)*np.cos(lat)*np.sin(lon) + np.cos(eps)*np.sin(lat)
    ])
    return r_moon


def srp_accel(r_vec, t_seconds):
    """Solar radiation pressure acceleration [m/s^2]. SRP pushes away from Sun."""
    r_sun = sun_position(t_seconds)

    # Vector from Sun to spacecraft (SRP force direction: away from Sun)
    r_sun_to_sc = r_vec - r_sun
    dist_sun = np.linalg.norm(r_sun_to_sc)
    r_sun_hat = r_sun_to_sc / dist_sun  # unit vector from Sun toward SC

    # Cylindrical shadow: project SC onto the Earth->Sun axis
    r_sun_dir = r_sun / np.linalg.norm(r_sun)  # unit vector from Earth toward Sun
    dot = np.dot(r_vec, r_sun_dir)
    perp = np.linalg.norm(r_vec - dot * r_sun_dir)
    in_shadow = (dot < 0) and (perp < RE)
    if in_shadow:
        return np.zeros(3)

    # SRP acceleration: away from Sun
    a_srp = CR * (A_m / M_SC) * P_SRP * (AU / dist_sun)**2 * r_sun_hat
    return a_srp


def lunisolar_accel(r_vec, t_seconds):
    """Third-body perturbation from Sun and Moon [m/s^2]."""
    r_sun  = sun_position(t_seconds)
    r_moon = moon_position(t_seconds)

    def third_body(r_sc, r_body, mu_body):
        d = r_body - r_sc
        dist = np.linalg.norm(d)
        r_b  = np.linalg.norm(r_body)
        return mu_body * (d / dist**3 - r_body / r_b**3)

    a_sun  = third_body(r_vec, r_sun,  MU_SUN)
    a_moon = third_body(r_vec, r_moon, MU_MOON)
    return a_sun + a_moon


# ─────────────────────────────────────────────
# EQUATIONS OF MOTION
# ─────────────────────────────────────────────

def eom(t, state):
    """Full equations of motion with perturbations."""
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)

    # Two-body
    a_2b = -MU / r**3 * r_vec

    # Perturbations
    a_j2   = j2_accel(r_vec)
    a_drag = drag_accel(r_vec, v_vec)
    a_srp  = srp_accel(r_vec, t)
    a_lsol = lunisolar_accel(r_vec, t)

    a_total = a_2b + a_j2 + a_drag + a_srp + a_lsol
    return np.concatenate([v_vec, a_total])


def eom_keplerian(t, state):
    """Two-body only equations of motion."""
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)
    a_2b = -MU / r**3 * r_vec
    return np.concatenate([v_vec, a_2b])


# ─────────────────────────────────────────────
# PART A: ONE-YEAR PROPAGATION
# ─────────────────────────────────────────────

print("="*60)
print("PART A: One-Year Orbit Propagation with Perturbations")
print("="*60)

# Initial state vector
state0 = kep2cart(A_SC, E_SC, I_SC, OM_SC, W_SC, NU_SC)
print(f"Initial position [km]: {state0[:3]/1e3}")
print(f"Initial velocity [km/s]: {state0[3:]/1e3}")
print(f"Perigee altitude [km]: {A_SC*(1-E_SC)/1e3 - RE/1e3:.1f}")
print(f"Apogee altitude [km]:  {A_SC*(1+E_SC)/1e3 - RE/1e3:.1f}")

# Integration span: 1 year
T_year = 365.25 * 86400  # [s]
t_span = (0, T_year)

# Output times: every 1 hour for 1 year
dt_out = 3600.0
t_eval = np.arange(0, T_year + dt_out, dt_out)

print(f"\nIntegrating for 1 year ({T_year/86400:.2f} days)...")
print("This may take a few minutes...")

sol = solve_ivp(
    eom,
    t_span,
    state0,
    method='DOP853',
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12,
    max_step=300.0  # max step 5 minutes
)

print(f"Integration complete. Status: {sol.status}, Message: {sol.message}")
print(f"Solution points: {len(sol.t)}")

# Convert all states to orbital elements
print("Converting to orbital elements...")
elements = np.array([cart2kep(sol.y[:, k]) for k in range(sol.y.shape[1])])
# columns: a[m], e, i[deg], Om[deg], w[deg], nu[deg]

a_hist  = elements[:, 0] / 1e3   # [km]
e_hist  = elements[:, 1]
i_hist  = elements[:, 2]         # [deg]
Om_hist = elements[:, 3]         # [deg]
w_hist  = elements[:, 4]         # [deg]
nu_hist = elements[:, 5]         # [deg]

t_days = sol.t / 86400.0         # [days]

# ── Plot orbital elements ──
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('AURORA Orbital Elements – 1-Year Propagation\n(Cowell\'s Method with J2, Drag, SRP, Lunisolar Perturbations)',
             fontsize=13, fontweight='bold')

ax = axes[0, 0]
ax.plot(t_days, a_hist, 'b', linewidth=0.8)
ax.set_ylabel('Semi-major axis $a$ [km]')
ax.set_xlabel('Time [days]')
ax.set_title('Semi-major axis')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(t_days, e_hist, 'r', linewidth=0.8)
ax.set_ylabel('Eccentricity $e$')
ax.set_xlabel('Time [days]')
ax.set_title('Eccentricity')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(t_days, i_hist, 'g', linewidth=0.8)
ax.set_ylabel('Inclination $i$ [deg]')
ax.set_xlabel('Time [days]')
ax.set_title('Inclination')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(t_days, Om_hist, 'm', linewidth=0.8)
ax.set_ylabel(r'RAAN $\Omega$ [deg]')
ax.set_xlabel('Time [days]')
ax.set_title('Right Ascension of Ascending Node')
ax.grid(True, alpha=0.3)

ax = axes[2, 0]
ax.plot(t_days, w_hist, 'c', linewidth=0.8)
ax.set_ylabel(r'Arg. of Perigee $\omega$ [deg]')
ax.set_xlabel('Time [days]')
ax.set_title('Argument of Perigee')
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
ax.plot(t_days, nu_hist, 'k', linewidth=0.4, alpha=0.6)
ax.set_ylabel(r'True Anomaly $\nu$ [deg]')
ax.set_xlabel('Time [days]')
ax.set_title('True Anomaly')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'partA_orbital_elements.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figs/partA_orbital_elements.png")

# ── Plot altitude history ──
alt_hist = a_hist * (1 - e_hist) - RE/1e3   # perigee altitude [km]
apo_hist = a_hist * (1 + e_hist) - RE/1e3   # apogee altitude [km]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t_days, alt_hist, 'b', linewidth=0.8, label='Perigee altitude')
ax.plot(t_days, apo_hist, 'r', linewidth=0.8, label='Apogee altitude')
ax.set_xlabel('Time [days]')
ax.set_ylabel('Altitude [km]')
ax.set_title('AURORA Perigee and Apogee Altitude – 1-Year Propagation')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'partA_altitude_history.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figs/partA_altitude_history.png")

# ── Perturbation magnitude comparison over one orbital period ──
print("Computing perturbation accelerations over one orbital period...")
T_orbit = 2 * np.pi * np.sqrt(A_SC**3 / MU)
t_one_orbit = np.linspace(0, T_orbit, 600)

sol_one = solve_ivp(
    eom,
    (0, T_orbit),
    state0,
    method='DOP853',
    t_eval=t_one_orbit,
    rtol=1e-10,
    atol=1e-12,
    max_step=60.0
)

a_j2_mag   = []
a_drag_mag = []
a_srp_mag  = []
a_ls_mag   = []

for k in range(sol_one.y.shape[1]):
    r_v = sol_one.y[:3, k]
    v_v = sol_one.y[3:, k]
    t_k = sol_one.t[k]
    a_j2_mag.append(np.linalg.norm(j2_accel(r_v)))
    a_drag_mag.append(np.linalg.norm(drag_accel(r_v, v_v)))
    a_srp_mag.append(np.linalg.norm(srp_accel(r_v, t_k)))
    a_ls_mag.append(np.linalg.norm(lunisolar_accel(r_v, t_k)))

t_orbit_hrs = t_one_orbit / 3600.0

fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(t_orbit_hrs, a_j2_mag, 'b', linewidth=1.5, label='J2 (Earth oblateness)')
ax.semilogy(t_orbit_hrs, a_drag_mag, 'r', linewidth=1.5, label='Atmospheric drag')
ax.semilogy(t_orbit_hrs, a_srp_mag, 'g', linewidth=1.5, label='Solar radiation pressure')
ax.semilogy(t_orbit_hrs, a_ls_mag, 'm', linewidth=1.5, label='Lunisolar (Sun + Moon)')
ax.set_xlabel('Time [hours from epoch]')
ax.set_ylabel(r'Acceleration magnitude [m s$^{-2}$]')
ax.set_title('Perturbation Acceleration Magnitudes over One Orbital Period (T $\\approx$ 12 h)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'partA_perturbation_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figs/partA_perturbation_comparison.png")

# Print perturbation order of magnitude at perigee and apogee
idx_per = np.argmin(np.linalg.norm(sol_one.y[:3, :], axis=0))
idx_apo = np.argmax(np.linalg.norm(sol_one.y[:3, :], axis=0))
print(f"\nPerturbation magnitudes at perigee (alt={np.linalg.norm(sol_one.y[:3,idx_per])/1e3-RE/1e3:.0f} km):")
print(f"  J2:         {a_j2_mag[idx_per]:.3e} m/s^2")
print(f"  Drag:       {a_drag_mag[idx_per]:.3e} m/s^2")
print(f"  SRP:        {a_srp_mag[idx_per]:.3e} m/s^2")
print(f"  Lunisolar:  {a_ls_mag[idx_per]:.3e} m/s^2")
print(f"\nPerturbation magnitudes at apogee (alt={np.linalg.norm(sol_one.y[:3,idx_apo])/1e3-RE/1e3:.0f} km):")
print(f"  J2:         {a_j2_mag[idx_apo]:.3e} m/s^2")
print(f"  Drag:       {a_drag_mag[idx_apo]:.3e} m/s^2")
print(f"  SRP:        {a_srp_mag[idx_apo]:.3e} m/s^2")
print(f"  Lunisolar:  {a_ls_mag[idx_apo]:.3e} m/s^2")

# ── Summary statistics ──
print("\n--- Part A Summary Statistics ---")
print(f"Semi-major axis: initial={a_hist[0]:.3f} km, final={a_hist[-1]:.3f} km, da={a_hist[-1]-a_hist[0]:.3f} km")
print(f"Eccentricity:   initial={e_hist[0]:.6f}, final={e_hist[-1]:.6f}, de={e_hist[-1]-e_hist[0]:.6f}")
print(f"Inclination:    initial={i_hist[0]:.4f} deg, final={i_hist[-1]:.4f} deg, di={i_hist[-1]-i_hist[0]:.4f} deg")
print(f"RAAN:           initial={Om_hist[0]:.3f} deg, final={Om_hist[-1]:.3f} deg, dOm={Om_hist[-1]-Om_hist[0]:.3f} deg")
print(f"Arg. perigee:   initial={w_hist[0]:.3f} deg, final={w_hist[-1]:.3f} deg, dw={w_hist[-1]-w_hist[0]:.3f} deg")


# ─────────────────────────────────────────────
# PART B: GROUND STATION PASSES (1-3 Jan 2026)
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("PART B: Ground Station Pass Analysis (1-3 Jan 2026 UTC)")
print("="*60)

# 2-day integration for both perturbed and Keplerian
T_pass = 3 * 86400.0  # 3 days
dt_pass = 30.0        # 30-second steps for pass resolution
t_pass = np.arange(0, T_pass + dt_pass, dt_pass)

print("Integrating perturbed trajectory for pass analysis...")
sol_pass = solve_ivp(
    eom,
    (0, T_pass),
    state0,
    method='DOP853',
    t_eval=t_pass,
    rtol=1e-10,
    atol=1e-12,
    max_step=60.0
)

print("Integrating Keplerian (unperturbed) trajectory...")
sol_kep = solve_ivp(
    eom_keplerian,
    (0, T_pass),
    state0,
    method='DOP853',
    t_eval=t_pass,
    rtol=1e-10,
    atol=1e-12,
    max_step=60.0
)


def compute_elevation(r_eci, t_sec, gs_lat, gs_lon, gs_alt=0.0):
    """Compute elevation angle of spacecraft from ground station [rad]."""
    # Convert SC to ECEF
    r_ecef = eci2ecef(r_eci, t_sec)

    # Ground station ECEF position
    N = RE / np.sqrt(1 - 0.00669437999014 * np.sin(gs_lat)**2)
    r_gs = np.array([
        (N + gs_alt) * np.cos(gs_lat) * np.cos(gs_lon),
        (N + gs_alt) * np.cos(gs_lat) * np.sin(gs_lon),
        (N * (1 - 0.00669437999014) + gs_alt) * np.sin(gs_lat)
    ])

    # Relative vector in ECEF
    dr = r_ecef - r_gs

    # Rotate to ENZ (topocentric) frame
    sin_lat, cos_lat = np.sin(gs_lat), np.cos(gs_lat)
    sin_lon, cos_lon = np.sin(gs_lon), np.cos(gs_lon)

    # ENZ transformation
    # E = [-sin_lon, cos_lon, 0]
    # N = [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat]
    # Z = [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    E_hat = np.array([-sin_lon, cos_lon, 0])
    N_hat = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat])
    Z_hat = np.array([cos_lat*cos_lon, cos_lat*sin_lon, sin_lat])

    dr_norm = np.linalg.norm(dr)
    if dr_norm < 1e-3:
        return 0.0

    # Elevation = angle between dr and the local horizontal plane
    Z_comp = np.dot(dr, Z_hat)
    elev = np.arcsin(Z_comp / dr_norm)
    return elev


def find_passes(sol_traj, t_vec, gs_lat, gs_lon, gs_name, elev_mask_rad=np.radians(5.0)):
    """Find all ground station passes and return pass info."""
    elevations = np.array([
        compute_elevation(sol_traj[:3, k], t_vec[k], gs_lat, gs_lon)
        for k in range(sol_traj.shape[1])
    ])

    # Find pass start/end indices
    above = elevations >= elev_mask_rad
    passes = []
    in_pass = False
    t_start = 0
    idx_start = 0
    max_elev = -np.inf

    for k in range(len(t_vec)):
        if above[k] and not in_pass:
            in_pass = True
            t_start = t_vec[k]
            idx_start = k
            max_elev = elevations[k]
        elif above[k] and in_pass:
            max_elev = max(max_elev, elevations[k])
        elif not above[k] and in_pass:
            in_pass = False
            t_end = t_vec[k-1]
            duration = t_end - t_start
            if duration > 30:  # ignore tiny glitches
                passes.append({
                    'start_s': t_start,
                    'end_s': t_end,
                    'duration_s': duration,
                    'max_elev_deg': np.degrees(max_elev),
                    'start_utc': EPOCH + timedelta(seconds=t_start),
                    'end_utc':   EPOCH + timedelta(seconds=t_end),
                })

    # Close final pass if still above at end
    if in_pass:
        t_end = t_vec[-1]
        duration = t_end - t_start
        if duration > 30:
            passes.append({
                'start_s': t_start,
                'end_s': t_end,
                'duration_s': duration,
                'max_elev_deg': np.degrees(max_elev),
                'start_utc': EPOCH + timedelta(seconds=t_start),
                'end_utc':   EPOCH + timedelta(seconds=t_end),
            })

    return passes, elevations


# Compute passes for PERTURBED trajectory
print("Computing perturbed passes - Malargüe...")
passes_mal_p, elev_mal_p = find_passes(sol_pass.y, sol_pass.t, MAL_LAT, MAL_LON, "Malargüe")
print("Computing perturbed passes - Kiruna...")
passes_kir_p, elev_kir_p = find_passes(sol_pass.y, sol_pass.t, KIR_LAT, KIR_LON, "Kiruna")

# Compute passes for KEPLERIAN trajectory
print("Computing Keplerian passes - Malargüe...")
passes_mal_k, elev_mal_k = find_passes(sol_kep.y, sol_kep.t, MAL_LAT, MAL_LON, "Malargüe")
print("Computing Keplerian passes - Kiruna...")
passes_kir_k, elev_kir_k = find_passes(sol_kep.y, sol_kep.t, KIR_LAT, KIR_LON, "Kiruna")


# 48-hour window: epoch to epoch+48h (coursework window: 1–3 Jan 2026 UTC)
T_WINDOW_48H = 2 * 86400.0  # 172800 s


def filter_passes_48h(passes):
    """Filter passes to the 48-hour coursework window."""
    return [p for p in passes if p['start_s'] < T_WINDOW_48H]


def print_pass_table(passes, gs_name, case):
    print(f"\n{'='*70}")
    print(f"  {gs_name} - {case}")
    print(f"{'='*70}")
    if not passes:
        print("  No passes detected.")
        return
    print(f"  {'#':<4} {'AOS (UTC)':<22} {'LOS (UTC)':<22} {'Duration':>10} {'Max Elev':>10}")
    print(f"  {'-'*64}")
    for i, p in enumerate(passes):
        print(f"  {i+1:<4} {str(p['start_utc']):<22} {str(p['end_utc']):<22} "
              f"{p['duration_s']/60:>8.2f} m  {p['max_elev_deg']:>8.2f} deg")
    total_contact = sum(p['duration_s'] for p in passes) / 60
    print(f"\n  Total contact time: {total_contact:.2f} minutes over {passes[-1]['end_s']/86400:.2f} days")


# Filter to 48-hour coursework window
passes_mal_p_48 = filter_passes_48h(passes_mal_p)
passes_kir_p_48 = filter_passes_48h(passes_kir_p)
passes_mal_k_48 = filter_passes_48h(passes_mal_k)
passes_kir_k_48 = filter_passes_48h(passes_kir_k)

print("\n--- 48-hour window (1 Jan 2026 12:00 UTC to 3 Jan 2026 12:00 UTC) ---")
print_pass_table(passes_mal_p_48, "Malargue (lon=291 deg, lat=-35 deg)", "PERTURBED")
print_pass_table(passes_kir_p_48, "Kiruna (lon=20 deg, lat=67 deg)", "PERTURBED")
print_pass_table(passes_mal_k_48, "Malargue (lon=291 deg, lat=-35 deg)", "KEPLERIAN (unperturbed)")
print_pass_table(passes_kir_k_48, "Kiruna (lon=20 deg, lat=67 deg)", "KEPLERIAN (unperturbed)")


# ── Elevation angle plots ──
t_hours = sol_pass.t / 3600.0

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Spacecraft Elevation Angle from Ground Stations\n(1–3 January 2026 UTC)', fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(t_hours, np.degrees(elev_mal_p), 'b', linewidth=0.8, label='Perturbed')
ax.plot(t_hours, np.degrees(elev_mal_k), 'r--', linewidth=0.8, alpha=0.7, label='Keplerian')
ax.axhline(5, color='k', linestyle=':', linewidth=1, label='Elevation mask (5°)')
ax.fill_between(t_hours, 5, np.degrees(elev_mal_p), where=np.degrees(elev_mal_p)>=5,
                alpha=0.2, color='blue')
ax.set_xlabel('Time [hours from epoch]')
ax.set_ylabel('Elevation [deg]')
ax.set_title('Malargüe, Argentina (λ=291°, φ=−35°)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 72)
ax.set_ylim(-10, 90)

ax = axes[1]
ax.plot(t_hours, np.degrees(elev_kir_p), 'b', linewidth=0.8, label='Perturbed')
ax.plot(t_hours, np.degrees(elev_kir_k), 'r--', linewidth=0.8, alpha=0.7, label='Keplerian')
ax.axhline(5, color='k', linestyle=':', linewidth=1, label='Elevation mask (5°)')
ax.fill_between(t_hours, 5, np.degrees(elev_kir_p), where=np.degrees(elev_kir_p)>=5,
                alpha=0.2, color='blue')
ax.set_xlabel('Time [hours from epoch]')
ax.set_ylabel('Elevation [deg]')
ax.set_title('Kiruna, Sweden (λ=20°, φ=67°)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 72)
ax.set_ylim(-10, 90)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'partB_elevation_angles.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: figs/partB_elevation_angles.png")

# ── Ground track plot ──
print("Computing ground track...")
lats, lons = [], []
for k in range(sol_pass.y.shape[1]):
    r_ecef = eci2ecef(sol_pass.y[:3, k], sol_pass.t[k])
    lat, lon, _ = ecef2lla(r_ecef)
    lats.append(np.degrees(lat))
    lons.append(np.degrees(lon))

lats = np.array(lats)
lons = np.array(lons)

fig, ax = plt.subplots(figsize=(14, 7))
# Handle longitude wrapping for plotting
lon_diff = np.abs(np.diff(lons))
split_idx = np.where(lon_diff > 180)[0] + 1

# Plot ground track in segments
prev = 0
for idx in split_idx:
    ax.plot(lons[prev:idx], lats[prev:idx], 'b-', linewidth=0.5, alpha=0.7)
    prev = idx
ax.plot(lons[prev:], lats[prev:], 'b-', linewidth=0.5, alpha=0.7, label='Ground track')

# Mark ground stations
ax.plot(np.degrees(MAL_LON), np.degrees(MAL_LAT), 'r^', markersize=12, label='Malargüe')
ax.plot(np.degrees(KIR_LON), np.degrees(KIR_LAT), 'gs', markersize=12, label='Kiruna')
ax.annotate('Malargüe', (np.degrees(MAL_LON), np.degrees(MAL_LAT)), textcoords="offset points",
            xytext=(5, 5), color='red', fontsize=9)
ax.annotate('Kiruna', (np.degrees(KIR_LON), np.degrees(KIR_LAT)), textcoords="offset points",
            xytext=(5, 5), color='green', fontsize=9)

ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
ax.set_xlabel('Longitude [deg]')
ax.set_ylabel('Latitude [deg]')
ax.set_title('AURORA Ground Track – 3 Days from 1 Jan 2026')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'partB_ground_track.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figs/partB_ground_track.png")

# ── Pass duration comparison bar chart ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Ground Station Pass Duration Comparison\n(Perturbed vs Keplerian)', fontsize=13, fontweight='bold')

for ax, passes_p, passes_k, name in zip(
    axes,
    [passes_mal_p_48, passes_kir_p_48],
    [passes_mal_k_48, passes_kir_k_48],
    ['Malargüe', 'Kiruna']
):
    n_p = len(passes_p)
    n_k = len(passes_k)

    dur_p = [p['duration_s']/60 for p in passes_p]
    dur_k = [p['duration_s']/60 for p in passes_k]
    max_n = max(n_p, n_k, 1)

    x_p = np.arange(n_p)
    x_k = np.arange(n_k)

    if n_p > 0:
        ax.bar(x_p - 0.2, dur_p, 0.35, label='Perturbed', color='steelblue', alpha=0.8)
    if n_k > 0:
        ax.bar(x_k + 0.2, dur_k, 0.35, label='Keplerian', color='tomato', alpha=0.8)

    ax.set_xlabel('Pass number')
    ax.set_ylabel('Duration [min]')
    ax.set_title(f'{name}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'partB_pass_durations.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figs/partB_pass_durations.png")

print("\n" + "="*60)
print("All computations complete.")
print(f"Figures saved to: {FIG_DIR}/")
print("  - partA_orbital_elements.png")
print("  - partA_altitude_history.png")
print("  - partB_elevation_angles.png")
print("  - partB_ground_track.png")
print("  - partB_pass_durations.png")
print("="*60)
