#!/usr/bin/env python

"""
Closes the loop on a monocular visual odometry trajectory whose scale drifted.

A single camera can't observe absolute scale, so visual odometry estimates each
keyframe's map at a scale that drifts over time. Here the drift is 3% per
keyframe. When the camera returns to its starting point, recognizing the place
gives a loop closure measurement between the last and first keyframes.

A pose graph in SE(3) can only spread the loop closure's error over positions
and rotations, which bends the trajectory. A pose graph in Sim(3) gives each
keyframe a scale as well, so the scale drift is spread over the trajectory and
its shape is recovered (Strasdat et al., "Scale Drift-Aware Large Scale
Monocular SLAM", RSS 2010).

The camera also has an IMU, which measures the direction of gravity in each
keyframe's frame. That's a constraint on each keyframe's rotation that the
solver takes directly: the measured up direction, rotated into the world frame,
must be within the IMU's accuracy of the world's up axis.
"""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from sleipnir.autodiff import (
    Pose3,
    Real,
    RealMatrix,
    Rotation3,
    Similarity3,
    SimilarityTwist3,
    UnitVector,
)
from sleipnir.optimization import Problem


def squared_norm(v: RealMatrix) -> Real:
    """Squared norm of a column vector."""
    return (v.T @ v)[0]


def main() -> None:
    # Number of keyframes. Keyframe N − 1 is back at keyframe 0's place.
    N = 25

    # Scale drift per keyframe
    drift = 1.03

    # Accuracy of the IMU's gravity direction
    imu_accuracy = math.radians(1.0)

    # The world's up axis
    up = np.array([0.0, 0.0, 1.0])

    # The true trajectory is a 5 m radius loop of road with hills 0.5 m high.
    # The camera faces along the road and pitches with its slope. Keyframe
    # frames are x forward, y left, and z up. world2keyframe maps a point in a keyframe's
    # frame to the world frame. Its scale is how many meters one unit
    # of that keyframe's map is, which drifts by 3% per keyframe.
    true_world2keyframe = []
    for i in range(N):
        θ = 2.0 * math.pi * i / (N - 1)
        translation = np.array(
            [5.0 * math.sin(θ), 5.0 * (1.0 - math.cos(θ)), 0.5 * math.sin(2.0 * θ)]
        )
        # Yaw θ, then pitch the nose up by the road's slope angle φ
        φ = math.atan(math.cos(2.0 * θ) / 5.0)
        rotation = Rotation3(
            math.cos(θ / 2.0) * math.cos(φ / 2.0),
            math.sin(θ / 2.0) * math.sin(φ / 2.0),
            -math.cos(θ / 2.0) * math.sin(φ / 2.0),
            math.sin(θ / 2.0) * math.cos(φ / 2.0),
        )
        true_world2keyframe.append(
            Similarity3(translation, rotation, i * math.log(drift))
        )

    # Visual odometry measures the relative position and rotation between
    # consecutive keyframes in the earlier keyframe's map units. It assumes the
    # scale stays the same, so it measures a relative scale of 1.
    sim3_edges = []
    for i in range(N - 1):
        relative = true_world2keyframe[i + 1].relative_to(true_world2keyframe[i])
        sim3_edges.append(
            (i, i + 1, Similarity3(relative.translation(), relative.rotation(), 0.0))
        )

    # The loop closure aligns the last keyframe's map with the first keyframe's
    # map, which also measures their scale ratio.
    sim3_edges.append(
        (N - 1, 0, true_world2keyframe[0].relative_to(true_world2keyframe[N - 1]))
    )

    # Each measurement has Gaussian noise in its own frame, with standard
    # deviations of 2 cm, 0.01 rad, and 1% scale. The seed is fixed so the
    # results are repeatable.
    σ_linear = 0.02
    σ_angular = 0.01
    σ_log_scale = 0.01
    rng = np.random.RandomState(42)
    for k, (i, j, measurement) in enumerate(sim3_edges):
        linear = rng.normal(0.0, σ_linear, 3)
        angular = rng.normal(0.0, σ_angular, 3)
        noise = SimilarityTwist3(linear, angular, rng.normal(0.0, σ_log_scale))
        sim3_edges[k] = (i, j, measurement * noise.exp())

    # The IMU measures the world's up axis in each keyframe's frame, with noise
    # of 0.2° per axis before normalization
    measured_up = [np.zeros(3)]
    for i in range(1, N):
        noise = rng.normal(0.0, 0.0035, 3)
        true_up = true_world2keyframe[i].rotation().to_matrix().value().T @ up
        v = true_up + noise
        measured_up.append(v / np.linalg.norm(v))

    # The SE(3) pose graph uses the same measurements without the scale
    se3_edges = [
        (i, j, Pose3(measurement.translation(), measurement.rotation()))
        for i, j, measurement in sim3_edges
    ]

    # Chains the odometry for the initial guess. The first keyframe defines the
    # world frame.
    odometry = [true_world2keyframe[0]]
    for i in range(N - 1):
        odometry.append(odometry[i] * sim3_edges[i][2])

    def rms_position_error(position: Callable[[int], npt.NDArray[np.float64]]) -> float:
        """Root mean square position error of a trajectory against the truth."""
        total = 0.0
        for i in range(N):
            e = position(i) - true_world2keyframe[i].translation().value()
            total += e @ e
        return math.sqrt(total / N)

    def max_tilt_error_degrees(rotation: Callable[[int], Rotation3]) -> float:
        """
        Largest angle between the world's up axis in a trajectory's keyframe frames
        and in the true keyframe frames.
        """
        max_error = 0.0
        for i in range(N):
            true_up = true_world2keyframe[i].rotation().to_matrix().value().T @ up
            estimated_up = rotation(i).to_matrix().value().T @ up
            cos_error = min(estimated_up @ true_up, 1.0)
            max_error = max(max_error, math.degrees(math.acos(cos_error)))
        return max_error

    print(
        "Odometry RMS position error: "
        f"{rms_position_error(lambda i: odometry[i].translation().value()):.4f} m"
    )
    odometry_tilt_error = max_tilt_error_degrees(lambda i: odometry[i].rotation())
    print(f"Odometry max tilt error: {odometry_tilt_error:.2f}°")

    # SE(3) pose graph
    problem = Problem()

    # The first keyframe is fixed because it defines the world frame
    world2keyframe = [Pose3(odometry[0].translation(), odometry[0].rotation())]
    for i in range(1, N):
        pose = problem.pose3_decision_variable()
        pose.translation().set_value(odometry[i].translation().value())
        pose.rotation().set_value(*odometry[i].rotation().value())
        world2keyframe.append(pose)

    # The error of each edge is the twist from the measurement to the estimated
    # relative pose. Each component is divided by its noise's standard deviation,
    # so the cost is the measurements' negative log-likelihood up to a constant.
    J = 0.0
    for i, j, measurement in se3_edges:
        error = (
            world2keyframe[j].relative_to(world2keyframe[i]).relative_to(measurement)
        ).log()
        J += (
            squared_norm(error.linear()) / σ_linear**2
            + squared_norm(error.angular()) / σ_angular**2
        )
    problem.minimize(J)

    # The IMU's measured up direction, rotated into the world frame, is within the
    # IMU's accuracy of the world's up axis
    for i in range(1, N):
        keyframe_up = UnitVector(measured_up[i]).rotate_by(world2keyframe[i].rotation())
        problem.subject_to(keyframe_up.dot(up) >= math.cos(imu_accuracy))

    problem.solve(diagnostics=True)

    se3_error = rms_position_error(lambda i: world2keyframe[i].translation().value())
    print(f"SE(3) RMS position error: {se3_error:.4f} m")
    se3_tilt_error = max_tilt_error_degrees(lambda i: world2keyframe[i].rotation())
    print(f"SE(3) max tilt error: {se3_tilt_error:.2f}°")

    # Sim(3) pose graph
    problem = Problem()

    # The first keyframe is fixed because it defines the world frame and its
    # scale
    world2keyframe = [odometry[0]]
    for i in range(1, N):
        similarity = problem.similarity3_decision_variable()
        similarity.translation().set_value(odometry[i].translation().value())
        similarity.rotation().set_value(*odometry[i].rotation().value())
        similarity.log_scale().set_value(odometry[i].log_scale().value())
        world2keyframe.append(similarity)

    # The error of each edge is the similarity twist from the measurement to the
    # estimated relative similarity transform. Each component is divided by its
    # noise's standard deviation, so the cost is the measurements' negative
    # log-likelihood up to a constant.
    J = 0.0
    for i, j, measurement in sim3_edges:
        error = (
            world2keyframe[j].relative_to(world2keyframe[i]).relative_to(measurement)
        ).log()
        J += (
            squared_norm(error.linear()) / σ_linear**2
            + squared_norm(error.angular()) / σ_angular**2
            + error.log_scale() ** 2 / σ_log_scale**2
        )
    problem.minimize(J)

    # The IMU's measured up direction, rotated into the world frame, is within the
    # IMU's accuracy of the world's up axis
    for i in range(1, N):
        keyframe_up = UnitVector(measured_up[i]).rotate_by(world2keyframe[i].rotation())
        problem.subject_to(keyframe_up.dot(up) >= math.cos(imu_accuracy))

    problem.solve(diagnostics=True)

    sim3_error = rms_position_error(lambda i: world2keyframe[i].translation().value())
    print(f"Sim(3) RMS position error: {sim3_error:.4f} m")
    sim3_tilt_error = max_tilt_error_degrees(lambda i: world2keyframe[i].rotation())
    print(f"Sim(3) max tilt error: {sim3_tilt_error:.2f}°")
    print(
        f"Sim(3) last keyframe scale: {world2keyframe[N - 1].scale().value():.4f} "
        f"(true {drift ** (N - 1):.4f})"
    )


if __name__ == "__main__":
    main()
