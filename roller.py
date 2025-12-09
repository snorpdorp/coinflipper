#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy.optimize import curve_fit
from math import sqrt

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simulate biased die rolls and plot posterior distributions."
    )

    parser.add_argument(
        "-N", "--num_rolls",
        type=int,
        default=2000,
        help="Number of die rolls (default: 2000)"
    )

    parser.add_argument(
        "-p", "--probs",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3],
        help="Probabilities for the first die faces; last face = 1 - sum(given)."
    )

    parser.add_argument(
        "-J", "--num_jiggles",
        type=int,
        default=200,
        help="Number of Poisson-jiggled resamples per round (default: 200)"
    )

    parser.add_argument(
        "-JR", "--jiggle_rounds",
        type=int,
        default=1000,
        help="Number of jiggle rounds (default: 50)"
    )

    return parser.parse_args()

# Gaussian function for fitting
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

def main():
    args = parse_arguments()

    N = args.num_rolls
    J = args.num_jiggles
    JR = args.jiggle_rounds

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------
    given = np.array(args.probs, dtype=float)
    s = given.sum()

    if s > 1.0:
        raise ValueError(f"Sum of provided probabilities = {s} must be <= 1.")

    p_last = 1.0 - s
    p_true = np.append(given, p_last)

    K = len(p_true)
    print(f"Number of die faces = {K}")
    print(f"True probabilities = {p_true}")

    # ------------------------------------------------------------------
    # Simulate multinomial rolls
    # ------------------------------------------------------------------
    print(f"Dummy round:")
    counts = np.random.multinomial(N, p_true)
    print("Observed counts:", counts)

    # Dirichlet(1...1) prior
    alpha_prior = np.ones(K)
    alpha_post = alpha_prior + counts
    posterior_mean = alpha_post / np.sum(alpha_post)
    print("Posterior mean:", posterior_mean)

    # Consistent colors
    colors = plt.cm.tab10(np.linspace(0, 1, K))

    # ------------------------------------------------------------------
    # SAVE FIRST ROUND jiggle_means for the "original bottom subplot"
    # ------------------------------------------------------------------
    first_round_jiggle_means = None
    zscores_by_face = [[] for _ in range(K)]

    print(f"Beginning rounds of jiggling.")
    for r in range(JR):
        counts = np.random.multinomial(N, p_true)
        jiggle_counts = np.random.poisson(counts, size=(J, K))
        jiggle_means = (jiggle_counts + 1) / np.sum(
            jiggle_counts + 1, axis=1
        )[:, None]

        if r == 0:
            first_round_jiggle_means = jiggle_means.copy()

        means = jiggle_means.mean(axis=0)
        stdevs = jiggle_means.std(axis=0, ddof=1)
        scales = stdevs * sqrt(J/(J-1))  # Use student scale param, not gaussian sigma param

        last_means = means
        last_stdevs = stdevs

        z = (means - p_true) / scales
        for face in range(K):
            zscores_by_face[face].append(z[face])

    zscores_by_face = np.array(zscores_by_face)

    # ------------------------------------------------------------------
    # FIGURE 1 — Top: true vs mean, Bottom: first-round jiggle histograms
    # ------------------------------------------------------------------
    plt.figure(figsize=(13, 11))

    # TOP SUBPLOT
    plt.subplot(2, 1, 1)
    x = np.arange(K)

    plt.bar(
        x, p_true,
        width=0.7,
        color=colors,
        alpha=0.65,
        edgecolor="black",
        label="True probability"
    )

    plt.errorbar(
        x, last_means,
        yerr=last_stdevs,
        fmt='o',
        color="black",
        capsize=4,
        markersize=8,
        label="Last jiggle mean ± stdev"
    )

    plt.xticks(x)
    plt.xlabel("Die face index")
    plt.ylabel("Probability")
    plt.title("True Probability vs. Jiggle-Estimated Probability (Last Round)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    # BOTTOM SUBPLOT — FIRST ROUND HISTOGRAMS
    plt.subplot(2, 1, 2)
    for face in range(K):
        plt.hist(
            first_round_jiggle_means[:, face],
            bins=30,
            density=True,
            alpha=0.30,
            color=colors[face],
            label=f"Face {face}"
        )

    plt.xlabel("Posterior mean probability (first jiggle round)")
    plt.ylabel("Density")
    plt.title("Posterior Mean Distributions (First Jiggle Round)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # FIGURE 2 — Z-score histograms across JR rounds with Gaussian and t-Student Fit
    # ------------------------------------------------------------------
    plt.figure(figsize=(14, 10))

    for face in range(K):
        plt.subplot(K, 1, face + 1)
        
        # Compute the mean and standard deviation of the z-scores
        z_mean = np.mean(zscores_by_face[face])
        z_std = np.std(zscores_by_face[face], ddof=1)

        # Create a histogram of the z-scores
        counts, bin_edges = np.histogram(zscores_by_face[face], bins=100, range=(-6, 6), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of the bins
        
        # Estimate the amplitude (integral of the histogram, assuming the area under the curve should be 1)
        amplitude = np.sum(counts) * (bin_edges[1] - bin_edges[0])
        
        # Fit a Gaussian function to the data and get the covariance matrix
        popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=[amplitude, z_mean, z_std])

        # Calculate the standard errors of the parameters from the covariance matrix
        perr = np.sqrt(np.diag(pcov))
        amplitude_err, mean_err, stddev_err = perr

        # Fit a t-distribution to the z-scores
        df, loc, scale = t.fit(zscores_by_face[face])  # Unpack the four returned values

        # Generate the t-distribution PDF
        t_pdf = t.pdf(bin_centers, df, loc, scale)
        t2_pdf = t.pdf(bin_centers, df=J-1, loc=0.0, scale=1.0)

        # Plot the histogram and Gaussian fit
        plt.hist(zscores_by_face[face], bins=100, range=(-6, 6), color=colors[face], alpha=0.6, density=True, label=f"Face {face} z-scores")
        
        # Plot the Gaussian fit
        x_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
        y_fit = gaussian(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='black', linestyle='--', label='Gaussian fit')

        # Plot the t-distribution fit
        plt.plot(bin_centers, t_pdf, color='red', linestyle='-', label='t-Student fit')

        plt.plot(bin_centers, t2_pdf, color='green', linestyle='-.', label=f't-Student precomputed df/scale/loc = {J-1=}/1.0/0.0')

        # Add error bands around the Gaussian fit
        y_err_upper = gaussian(x_fit, popt[0] + amplitude_err, popt[1] + mean_err, popt[2] + stddev_err)
        y_err_lower = gaussian(x_fit, popt[0] - amplitude_err, popt[1] - mean_err, popt[2] - stddev_err)
        plt.fill_between(x_fit, y_err_lower, y_err_upper, color='gray', alpha=0.3, label='Fit error range')

        # Add a vertical line at 0 to indicate where the true value is
        plt.axvline(0, color="black", linestyle="--")
        
        # Labels and legend
        plt.ylabel("Density")
        plt.legend()

        # Display mu and sigma with their uncertainties
        mu_text = f"$\\mu = {popt[1]:.2f} \\pm {mean_err:.2f}$"
        sigma_text = f"$\\sigma = {popt[2]:.2f} \\pm {stddev_err:.2f}$"

        plt.text(0.05, 0.85, mu_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.75, sigma_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Display t-Student parameters
        df_text = f"$df = {df:.2f}$"
        scale_text = f"$\\sigma = {scale:.2f}$"
        plt.text(0.05, 0.55, df_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.45, scale_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        if face == 0:
            params_text = (
                f"N={N}, "
                f"J={J}, "
                f"JR={JR}, "
                f"probs={p_true.tolist()}"
            )
            plt.title(f"Z-score Distributions with Gaussian and t-Student Fit\n{params_text}")
        # plt.yscale('log')  # Logarithmic scale on y-axis

    # Change y-axis to log scale
    plt.xlabel("z = (mean_jiggle - p_true) / scale_jiggle")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
