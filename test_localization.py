"""Numerical and behavioural checks independent of the generated figures."""

from dataclasses import replace
import unittest
import numpy as np

from localization import (PoseEKF, estimate, generate_dataset, landmark_model,
                          motion_and_jacobians, score, wrap_angle)


def numerical_jacobian(function, point, angle_rows=(), epsilon=1e-6):
    columns = []
    for index in range(len(point)):
        offset = np.zeros_like(point)
        offset[index] = epsilon
        difference = function(point + offset) - function(point - offset)
        for row in angle_rows:
            difference[row] = wrap_angle(difference[row])
        columns.append(difference / (2 * epsilon))
    return np.column_stack(columns)


class JacobianTests(unittest.TestCase):
    def test_motion_state_and_wheel_jacobians_match_finite_differences(self):
        for state, wheels in ((np.array([1., -2., 0.7]), np.array([0.04, 0.08])),
                              (np.array([0., 0., np.pi - 0.01]), np.array([-0.01, 0.06])),
                              (np.array([-3., 2., -2.]), np.array([0.05, 0.05]))):
            _, F, G = motion_and_jacobians(state, wheels)
            expected_F = numerical_jacobian(lambda x: motion_and_jacobians(x, wheels)[0], state, (2,))
            expected_G = numerical_jacobian(lambda u: motion_and_jacobians(state, u)[0], wheels, (2,))
            np.testing.assert_allclose(F, expected_F, atol=1e-8, rtol=1e-6)
            np.testing.assert_allclose(G, expected_G, atol=1e-8, rtol=1e-6)

    def test_observation_jacobian_including_bearing_branch_matches_numerically(self):
        for state in (np.array([1., 2., 0.3]), np.array([1., 2., np.pi])):
            landmark = np.array([5., 2.])
            _, H = landmark_model(state, landmark)
            expected = numerical_jacobian(lambda x: landmark_model(x, landmark)[0], state, (1,))
            np.testing.assert_allclose(H, expected, atol=1e-8, rtol=1e-6)


class FilterTests(unittest.TestCase):
    def test_wrapped_bearing_innovation_is_small_across_pi(self):
        ekf = PoseEKF(state=[0., 0., np.pi - 0.01])
        original_heading = ekf.state[2]
        innovation = ekf.update([5., np.pi - 0.01], [5., 0.])
        self.assertAlmostEqual(innovation[1], -0.02)
        self.assertLess(abs(wrap_angle(ekf.state[2] - original_heading)), 0.03)
        self.assertTrue(-np.pi <= ekf.state[2] < np.pi)

    def test_covariance_remains_symmetric_positive_semidefinite(self):
        data = generate_dataset(3)
        for run in (estimate(data), estimate(data, dropout=(20., 40.))):
            np.testing.assert_allclose(run.covariances, run.covariances.transpose(0, 2, 1), atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(run.covariances).min(), -1e-12)
            self.assertTrue(np.isfinite(run.states).all())

    def test_invalid_and_singular_inputs_are_rejected(self):
        invalid = [dict(state=[0, 0, np.nan]), dict(initial_covariance=-np.eye(3)),
                   dict(initial_covariance=np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])),
                   dict(sensor_covariance=np.zeros((2, 2))), dict(wheel_covariance=-np.eye(2)),
                   dict(wheel_base=0)]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                PoseEKF(**arguments)
        ekf = PoseEKF(state=[0., 0., 0.])
        for measurement, landmark in (([0, 0], [1, 2]), ([2, np.inf], [1, 2]),
                                       ([2, 0], [0, 0]), ([2, 0], [np.nan, 1])):
            with self.assertRaises(ValueError):
                ekf.update(measurement, landmark)
        with self.assertRaises(ValueError):
            ekf.predict([np.nan, 0])
        with self.assertRaises(ValueError):
            motion_and_jacobians([0, 0, 0], [-1e308, 1e308])
        with self.assertRaises(ValueError):
            wrap_angle(np.inf)

    def test_stationary_exact_observation_preserves_pose_and_reduces_uncertainty(self):
        ekf = PoseEKF(state=[1., 2., 0.4])
        before = ekf.state.copy()
        prior_trace = np.trace(ekf.P)
        measurement, _ = landmark_model(ekf.state, [4., 5.])
        ekf.update(measurement, [4., 5.])
        np.testing.assert_allclose(ekf.state, before, atol=1e-12)
        self.assertLess(np.trace(ekf.P), prior_trace)


class ExperimentTests(unittest.TestCase):
    def test_no_landmarks_is_exactly_the_same_odometry_ablation(self):
        data = generate_dataset(1, steps=100)
        removed = replace(data, observations=tuple(() for _ in data.observations))
        baseline = estimate(data, use_landmarks=False)
        without_measurements = estimate(removed)
        np.testing.assert_array_equal(baseline.states, without_measurements.states)
        np.testing.assert_array_equal(baseline.covariances, without_measurements.covariances)

    def test_ground_truth_changes_cannot_change_estimator_output(self):
        data = generate_dataset(2, steps=100)
        altered = replace(data, truth=data.truth + np.array([100., -50., 1.]))
        np.testing.assert_array_equal(estimate(data).states, estimate(altered).states)

    def test_landmarks_improve_paired_position_rmse_across_three_seeds(self):
        for seed in (0, 1, 2):
            data = generate_dataset(seed)
            odometry = score(data.truth, estimate(data, use_landmarks=False))
            ekf = score(data.truth, estimate(data))
            self.assertLess(ekf['position_rmse_m'], odometry['position_rmse_m'] * 0.25)

    def test_dropout_masks_only_the_interval_and_observations_recover(self):
        data = generate_dataset(0)
        normal, dropout = estimate(data), estimate(data, dropout=(20., 40.))
        times = np.arange(len(dropout.states)) * data.dt
        np.testing.assert_array_equal(normal.states[times < 20], dropout.states[times < 20])
        self.assertEqual(dropout.updates_per_step[(times >= 20) & (times < 40)].sum(), 0)
        self.assertGreater(dropout.updates_per_step[times >= 40].sum(), 0)
        trace = np.trace(dropout.covariances[:, :2, :2], axis1=1, axis2=2)
        self.assertGreater(trace[399], trace[199])
        self.assertLess(trace[400], trace[399])
        self.assertLess(score(data.truth, dropout)['final_position_error_m'], 0.2)

    def test_data_and_filter_are_reproducible_with_independent_noise_seeds(self):
        first, second = generate_dataset(4, steps=100), generate_dataset(4, steps=100)
        np.testing.assert_array_equal(first.truth, second.truth)
        np.testing.assert_array_equal(first.odometry, second.odometry)
        np.testing.assert_array_equal(estimate(first).states, estimate(second).states)
        third = generate_dataset(5, steps=100)
        np.testing.assert_array_equal(first.truth, third.truth)
        self.assertFalse(np.array_equal(first.odometry, third.odometry))
        for a, b in zip(first.observations, second.observations):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assertEqual(x.landmark_id, y.landmark_id)
                np.testing.assert_array_equal(x.measurement, y.measurement)


if __name__ == '__main__':
    unittest.main()
