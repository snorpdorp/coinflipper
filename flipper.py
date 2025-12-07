#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from scipy.optimize import curve_fit
from math import comb
from uncertainties import ufloat
from uncertainties import unumpy as unp
import uncertainties.umath as umath

def binomial_likelihood(p, N, k):
    return comb(N, k) * (p**k) * ((1 - p)**(N - k))

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def u_gaussian(x, A, mu, sigma):
    return A * umath.exp(-(x - mu)**2 / (2 * sigma**2))

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Simulate biased coin flips and plot distributions."
    )
    parser.add_argument(
        "-N", "--num_flips",
        type=int,
        default=10**3,
        help="Number of coin flips (default: 10^3)"
    )
    parser.add_argument(
        "-p", "--prob",
        type=float,
        default=0.3,
        help="True probability of heads (default: 0.3)"
    )
    parser.add_argument(
        "-J", "--num_jiggles",
        type=int,
        default=100,
        help="Number of Poisson jiggled resamples (default: 100)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    N = args.num_flips
    p_true = args.prob
    J = args.num_jiggles

    # Simulate original flips
    k = np.random.binomial(N, p_true)

    # Simulate N flips as 0/1 array
    # Can be done for higher N at expense of O(n) runtime and memory
    # flips = np.random.binomial(1, p_true, size=N)
    # k = flips.sum()

    print(f"Flipped the coin {N} times with p={p_true}")
    print(f"Observed heads: {k}, tails: {N-k}")

    # Prior Beta(1,1)
    alpha_prior = 1
    beta_prior  = 1

    # Posterior for the original sample
    alpha_post = alpha_prior + k
    beta_post  = beta_prior  + (N - k)

    # Grid for plotting PDFs
    p = np.linspace(0, 1, 500)

    # Original posterior curve
    post_pdf = beta.pdf(p, alpha_post, beta_post)

    # ----------------------------------------------------------
    #  Poisson Jiggling of Counts
    # ----------------------------------------------------------

    k_jiggles = np.random.poisson(k, size=J)
    f_jiggles = np.random.poisson(N - k, size=J)

    # Compute posterior MEANS from jiggles
    jiggle_p_means = []
    for kj, fj in zip(k_jiggles, f_jiggles):
        a = alpha_prior + kj
        b = beta_prior + fj
        jiggle_p_means.append(a / (a + b))

    jiggle_p_means = np.array(jiggle_p_means)

    # Fit a gaussian:
    # Histogram
    counts, bin_edges = np.histogram(jiggle_p_means, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess
    mu_init = np.mean(jiggle_p_means)
    sigma_init = np.std(jiggle_p_means, ddof=1)
    A_init = max(counts)
    p0 = [A_init, mu_init, sigma_init]

    # Fit Gaussian
    popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=p0)

    # Convert fitted parameters to ufloat with uncertainties
    A_fit = ufloat(popt[0], np.sqrt(pcov[0,0]))
    mu_fit = ufloat(popt[1], np.sqrt(pcov[1,1]))
    sigma_fit = ufloat(popt[2], np.sqrt(pcov[2,2]))

    print(f"Fitted parameters: A={A_fit}, mu={mu_fit}, sigma={sigma_fit}")

    # For later plotting
    gauss_y_fit = np.array([u_gaussian(xi, A_fit, mu_fit, sigma_fit) for xi in p])
    gauss_y_nom = unp.nominal_values(gauss_y_fit)
    gauss_y_std = unp.std_devs(gauss_y_fit)

    # ----------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------
    plt.figure(figsize=(11, 7))

    # Histogram of jiggled posterior means
    plt.hist(
        jiggle_p_means,
        bins=30,
        color="orange",
        alpha=0.5,
        density=True,
        label=f"Histogram of jiggled posterior means (J={J})"
    )

    # True posterior curve
    plt.plot(p, post_pdf, color="blue", linewidth=2,
             label=f"Posterior Beta({alpha_post},{beta_post}) (Analytical solution with Laplace Smoothing)")

    # True p
    plt.axvline(p_true, color="red", linestyle=":",
                label=f"True p = {p_true}")

    plt.plot(p, gauss_y_nom, 'r-', label='Gaussian fit of jiggles')
    plt.fill_between(p, gauss_y_nom - gauss_y_std, gauss_y_nom + gauss_y_std, color='r', alpha=0.3, label='Gaussian Fit uncertainty (1sigma)')

    title_str = (
    f"Posterior With Poisson-Jiggled Posterior Mean Histogram\n"
    f"N={N}, p_true={p_true}, J={J} | "
    )
    plt.title(title_str, fontsize=12)

    plt.xlabel("p")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
