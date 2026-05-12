#!/usr/bin/env python3

import argparse
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_poly_fit(ax, x, y, func, bases, color):
    """
    Plots a polynomial curve fit for the given x-y function.

    Parameter ``ax``:
        The axis on which to plot.

    Parameter ``x``:
        The list of x values.

    Parameter ``y``:
        The list of y values.

    Parameter ``func``:
        The function to fit.

    Parameter ``bases``:
        A list of strings to print for the basis terms.

    Parameter ``color``:
        The color to use for the plot.
    """
    coeffs = curve_fit(
        func, x, y, p0=(1,) * len(bases), bounds=([0] * len(bases), np.inf)
    )[0]

    label = ""
    for i in range(len(coeffs)):
        if abs(coeffs[i]) > 1e-10:
            if label:
                label += " + " if coeffs[i] > 0 else " - "
            label += f"{coeffs[i]:.4g}{bases[i]}"
    label = "Fit: y = " + label

    resampled_x = np.arange(x[0], x[-1] + 10, 10)
    ax.plot(
        resampled_x,
        func(resampled_x, *coeffs),
        color=color,
        label=label,
        linestyle="--",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Runs all formatting tasks on the code base. This should be invoked from a directory within the project."
    )
    parser.add_argument(
        "--filenames",
        dest="filenames",
        type=str,
        nargs="+",
        required=True,
        help="filenames of CSVs to plot",
    )
    parser.add_argument(
        "--labels",
        dest="labels",
        type=str,
        nargs="+",
        required=True,
        help="plot labels for each filename's data",
    )
    parser.add_argument(
        "--title",
        dest="title",
        type=str,
        required=True,
        help="plot title",
    )
    parser.add_argument(
        "--noninteractive",
        dest="noninteractive",
        action="store_true",
        help="if present, saves the figure to a file instead of displaying it",
    )
    args = parser.parse_args()

    samples = []
    setup_times = []
    solve_times = []

    for filename in args.filenames:
        (
            sample,
            setup_time,
            solve_time,
        ) = np.genfromtxt(
            filename,
            delimiter=",",
            skip_header=1,
            unpack=True,
            invalid_raise=False,
            ndmin=2,
        )
        if math.isnan(setup_time[-1]) or math.isnan(solve_time[-1]):
            sample = sample[:-1]
            setup_time = setup_time[:-1]
            solve_time = solve_time[:-1]
        if len(sample) == 0:
            print("No data to plot.")
            sys.exit(1)

        samples.append(sample)
        setup_times.append(setup_time)
        solve_times.append(solve_time)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title(f"{args.title} problem runtime vs samples")
    ax1.set_ylabel("Setup time (ms)")
    ax1.grid(visible=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i in range(len(samples)):
        ax1.plot(samples[i], setup_times[i], label=args.labels[i])
        plot_poly_fit(
            ax1,
            samples[i],
            setup_times[i],
            lambda x, a, b, c: a * x**2 + b * x + c,
            ["x²", "x", ""],
            color=colors[i],
        )

    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Solve time (ms)")
    ax2.grid(visible=True)

    for i in range(len(samples)):
        ax2.plot(samples[i], solve_times[i], label=args.labels[i])
        plot_poly_fit(
            ax2,
            samples[i],
            solve_times[i],
            lambda x, a, b, c: a * x**2 + b * x + c,
            ["x²", "x", ""],
            color=colors[i],
        )

    ax2.legend()

    if args.noninteractive:
        plt.savefig(f"{args.title.lower()}-scalability-results.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
