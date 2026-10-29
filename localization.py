"""Educational 2D pose EKF. Metres, radians and wheel-travel increments are used."""

from dataclasses import dataclass
import math
import numpy as np


WHEEL_BASE = 0.28
DT = 0.1
WHEEL_NOISE_STD = 0.002
ASSUMED_WHEEL_NOISE_STD = 0.003
WHEEL_SCALE_BIAS = np.array([0.012, -0.008])
RANGE_STD = 0.08
BEARING_STD = math.radians(2.0)
INITIAL_ESTIMATE = np.array([0.20, -0.15, 0.08])
INITIAL_COVARIANCE = np.diag([0.30**2, 0.30**2, 0.12**2])
LANDMARKS = np.array([[-2., -2.], [4., -2.], [10., -2.],
                      [-2., 5.], [4., 5.], [10., 5.],
                      [-2., 11.], [4., 11.], [10., 11.]])


def wrap_angle(angle):
    """Wrap a scalar or array to [-pi, pi)."""
    value = np.asarray(angle, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError("Angles must be finite.")
    return (value + np.pi) % (2 * np.pi) - np.pi


def vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {length}.")
    return result


def covariance(value, size, name, strictly_positive=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (size, size) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size} x {size} matrix.")
    if not np.allclose(result, result.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    values = np.linalg.eigvalsh(result)
    if values.min() < -1e-12 or (strictly_positive and values.min() <= 0):
        raise ValueError(f"{name} must be {'positive definite' if strictly_positive else 'positive semidefinite'}.")
    return result.copy()


def motion_and_jacobians(state, wheels, wheel_base=WHEEL_BASE):
    """Midpoint differential-drive motion and its exact analytical Jacobians.

    This is the chosen discrete model, not exact continuous arc integration.
    wheels contains left/right distances travelled during one time interval.
    """
    state = vector(state, 3, "state")
    left, right = vector(wheels, 2, "wheel increments")
    if not math.isfinite(wheel_base) or wheel_base <= 0:
        raise ValueError("wheel_base must be finite and positive.")
    distance = left / 2 + right / 2
    with np.errstate(over='ignore', invalid='ignore'):
        turn = (right - left) / wheel_base
    if not math.isfinite(distance) or not math.isfinite(turn):
        raise ValueError("Wheel increments exceed the numerical range of the motion model.")
    midpoint = state[2] + turn / 2
    c, s = math.cos(midpoint), math.sin(midpoint)
    predicted = state + np.array([distance * c, distance * s, turn])
    predicted[2] = wrap_angle(predicted[2])
    state_jacobian = np.array([[1., 0., -distance * s],
                               [0., 1., distance * c], [0., 0., 1.]])
    wheel_jacobian = np.array([
        [c / 2 + distance * s / (2 * wheel_base), c / 2 - distance * s / (2 * wheel_base)],
        [s / 2 - distance * c / (2 * wheel_base), s / 2 + distance * c / (2 * wheel_base)],
        [-1 / wheel_base, 1 / wheel_base],
    ])
    return predicted, state_jacobian, wheel_jacobian


def landmark_model(state, landmark):
    """Expected range/bearing and analytical state Jacobian for a known landmark."""
    state = vector(state, 3, "state")
    landmark = vector(landmark, 2, "landmark")
    dx, dy = landmark - state[:2]
    squared_range = dx * dx + dy * dy
    if squared_range <= 1e-12:
        raise ValueError("Landmark range is too small: its bearing is singular.")
    distance = math.sqrt(squared_range)
    predicted = np.array([distance, wrap_angle(math.atan2(dy, dx) - state[2])])
    jacobian = np.array([[-dx / distance, -dy / distance, 0.],
                         [dy / squared_range, -dx / squared_range, -1.]])
    return predicted, jacobian


class PoseEKF:
    """Pose-only EKF with known correspondences; wheel biases are not estimated."""

    def __init__(self, state=INITIAL_ESTIMATE, initial_covariance=INITIAL_COVARIANCE,
                 wheel_covariance=None, sensor_covariance=None, wheel_base=WHEEL_BASE):
        if not math.isfinite(wheel_base) or wheel_base <= 0:
            raise ValueError("wheel_base must be finite and positive.")
        self.state = vector(state, 3, "state").copy()
        self.state[2] = wrap_angle(self.state[2])
        self.P = covariance(initial_covariance, 3, "initial covariance")
        self.Q = covariance(np.eye(2) * ASSUMED_WHEEL_NOISE_STD**2 if wheel_covariance is None
                            else wheel_covariance, 2, "wheel covariance")
        self.R = covariance(np.diag([RANGE_STD**2, BEARING_STD**2]) if sensor_covariance is None
                            else sensor_covariance, 2, "sensor covariance", strictly_positive=True)
        self.wheel_base = wheel_base

    def predict(self, wheels):
        predicted, F, G = motion_and_jacobians(self.state, wheels, self.wheel_base)
        with np.errstate(over='ignore', invalid='ignore'):
            predicted_covariance = F @ self.P @ F.T + G @ self.Q @ G.T
        if not np.isfinite(predicted).all() or not np.isfinite(predicted_covariance).all():
            raise ValueError("Non-finite prediction; input scales are numerically invalid.")
        self.state = predicted
        self.P = (predicted_covariance + predicted_covariance.T) / 2

    def update(self, measurement, landmark):
        """Apply one observation, returning the wrapped innovation for inspection."""
        measurement = vector(measurement, 2, "range/bearing measurement")
        if measurement[0] <= 0:
            raise ValueError("Measured range must be positive.")
        expected, H = landmark_model(self.state, landmark)
        innovation = measurement - expected
        innovation[1] = wrap_angle(innovation[1])
        S = H @ self.P @ H.T + self.R
        try:
            # Positive-definite R should make S invertible; fail explicitly if
            # numerical corruption or invalid values have reached the filter.
            np.linalg.cholesky(S)
            gain = np.linalg.solve(S, (self.P @ H.T).T).T
        except np.linalg.LinAlgError as error:
            raise ValueError("Innovation covariance is singular or not positive definite.") from error
        corrected = self.state + gain @ innovation
        corrected[2] = wrap_angle(corrected[2])
        residual_matrix = np.eye(3) - gain @ H
        # Joseph form retains covariance symmetry/PSD better than (I-KH)P.
        corrected_covariance = residual_matrix @ self.P @ residual_matrix.T + gain @ self.R @ gain.T
        if not np.isfinite(corrected).all() or not np.isfinite(corrected_covariance).all():
            raise ValueError("Non-finite update; input scales are numerically invalid.")
        self.state = corrected
        self.P = (corrected_covariance + corrected_covariance.T) / 2
        return innovation.copy()


@dataclass(frozen=True)
class Observation:
    landmark_id: int
    measurement: np.ndarray


@dataclass(frozen=True)
class Dataset:
    truth: np.ndarray
    odometry: np.ndarray
    observations: tuple
    landmarks: np.ndarray
    dt: float


def generate_dataset(seed=0, steps=600, dt=DT):
    """Generate truth and measured sensors once; dropout is applied afterward.

    Truth is not an estimator input. Two independent RNG streams separate wheel
    noise from landmark noise. The path is identical across the ten noise seeds.
    """
    if not isinstance(seed, int) or seed < 0 or not isinstance(steps, int) or steps <= 0:
        raise ValueError("Seed must be a non-negative integer and steps a positive integer.")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    wheel_seed, sensor_seed = np.random.SeedSequence(seed).spawn(2)
    wheel_rng = np.random.default_rng(wheel_seed)
    sensor_rng = np.random.default_rng(sensor_seed)
    truth = np.zeros((steps + 1, 3))
    odometry = np.zeros((steps, 2))
    observations = []
    for step in range(steps):
        time = step * dt
        speed = 0.55 + 0.08 * math.sin(0.35 * time)
        yaw_rate = 0.14 + 0.14 * math.sin(0.12 * time) + 0.10 * math.sin(0.5 * time)
        true_wheels = np.array([speed - yaw_rate * WHEEL_BASE / 2,
                                speed + yaw_rate * WHEEL_BASE / 2]) * dt
        truth[step + 1] = motion_and_jacobians(truth[step], true_wheels)[0]
        odometry[step] = true_wheels * (1 + WHEEL_SCALE_BIAS) + wheel_rng.normal(0, WHEEL_NOISE_STD, 2)
        observed = []
        if (step + 1) % 5 == 0:  # every fifth control interval; 2 Hz at default dt
            for landmark_id, landmark in enumerate(LANDMARKS):
                expected, _ = landmark_model(truth[step + 1], landmark)
                if 0.5 <= expected[0] <= 7.0:  # assumed 360-degree sensor, finite range
                    measured = expected + sensor_rng.normal(0, [RANGE_STD, BEARING_STD])
                    measured[1] = wrap_angle(measured[1])
                    if measured[0] > 0:
                        observed.append(Observation(landmark_id, measured))
        observations.append(tuple(observed))
    return Dataset(truth, odometry, tuple(observations), LANDMARKS.copy(), dt)


@dataclass(frozen=True)
class Run:
    states: np.ndarray
    covariances: np.ndarray
    updates_per_step: np.ndarray


def estimate(dataset, use_landmarks=True, dropout=None):
    """Run prediction/update using measured sensors, never dataset.truth.

    dropout=(start_s, end_s) masks all observations in the half-open interval.
    Initial estimate/prior are identical for the odometry ablation and both EKFs.
    """
    if dropout is not None:
        start, end = vector(dropout, 2, "dropout interval")
        if start < 0 or end <= start:
            raise ValueError("Dropout needs 0 <= start < end.")
    ekf = PoseEKF()
    steps = len(dataset.odometry)
    states = np.empty((steps + 1, 3))
    covariances = np.empty((steps + 1, 3, 3))
    counts = np.zeros(steps + 1, dtype=int)
    states[0], covariances[0] = ekf.state, ekf.P
    for step, wheels in enumerate(dataset.odometry):
        ekf.predict(wheels)
        time = (step + 1) * dataset.dt
        missing = dropout is not None and dropout[0] <= time < dropout[1]
        if use_landmarks and not missing:
            for observation in dataset.observations[step]:
                ekf.update(observation.measurement, dataset.landmarks[observation.landmark_id])
                counts[step + 1] += 1
        states[step + 1], covariances[step + 1] = ekf.state, ekf.P
    return Run(states, covariances, counts)


def score(truth, run):
    """Score after estimation; the initial pose is included in trajectory RMSE."""
    if truth.shape != run.states.shape:
        raise ValueError("Truth and estimated trajectories must have the same shape.")
    position_errors = np.linalg.norm(run.states[:, :2] - truth[:, :2], axis=1)
    heading_errors = wrap_angle(run.states[:, 2] - truth[:, 2])
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_errors**2))),
        "final_position_error_m": float(position_errors[-1]),
        "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(heading_errors**2)))),
        "landmark_updates": int(run.updates_per_step.sum()),
        "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(run.covariances).min()),
    }


def run_study(seeds=range(10)):
    """Paired trials on one route with independent deterministic noise seeds."""
    records = {"odometry_only": [], "landmark_ekf": [], "dropout_ekf": []}
    for seed in seeds:
        dataset = generate_dataset(int(seed))
        runs = {"odometry_only": estimate(dataset, use_landmarks=False),
                "landmark_ekf": estimate(dataset),
                "dropout_ekf": estimate(dataset, dropout=(20., 40.))}
        for name, run in runs.items():
            records[name].append({"seed": int(seed), **score(dataset.truth, run)})
    if not records["odometry_only"]:
        raise ValueError("At least one seed is required.")
    aggregate = {}
    for name, trials in records.items():
        aggregate[name] = {}
        for metric in ("position_rmse_m", "final_position_error_m", "heading_rmse_deg"):
            values = np.array([trial[metric] for trial in trials])
            aggregate[name]["mean_" + metric] = float(values.mean())
            aggregate[name]["sample_std_" + metric] = float(values.std(ddof=1)) if len(values) > 1 else None
    return {
        "experiment": {
            "seeds": [trial["seed"] for trial in records["odometry_only"]],
            "steps": 600, "dt_s": DT, "duration_s": 600 * DT,
            "sensor_period_steps": 5, "landmark_count": len(LANDMARKS),
            "sensor_range_m": [0.5, 7.0], "sensor_field_of_view_deg": 360,
            "wheel_base_m": WHEEL_BASE, "wheel_scale_bias_left_right": WHEEL_SCALE_BIAS.tolist(),
            "true_encoder_noise_std_m_per_step": WHEEL_NOISE_STD,
            "assumed_encoder_noise_std_m_per_step": ASSUMED_WHEEL_NOISE_STD,
            "range_noise_std_m": RANGE_STD, "bearing_noise_std_deg": math.degrees(BEARING_STD),
            "initial_estimate_m_m_rad": INITIAL_ESTIMATE.tolist(),
            "initial_covariance": INITIAL_COVARIANCE.tolist(),
            "true_initial_pose_m_m_rad": [0., 0., 0.],
            "dropout_interval_s_half_open": [20., 40.],
            "motion_model": "Midpoint differential-drive discrete model, shared by truth and prediction.",
            "aggregation": "Mean and sample standard deviation of per-seed trajectory RMSE/final errors; not a confidence interval.",
            "scope": "One synthetic route with known landmarks and associations; no hardware or real sensor data.",
        },
        "aggregate": aggregate,
        "trials": records,
    }
