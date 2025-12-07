#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from math import comb

def binomial_likelihood(p, N, k):
    return comb(N, k) * (p**k) * ((1 - p)**(N - k))

def main():
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

    args = parser.parse_args()
    N = args.num_flips
    p_true = args.prob
    J = args.num_jiggles

    # Simulate original flips
    k = np.random.binomial(N, p_true)
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

    # Likelihood (scaled)
    likelihood = np.array([binomial_likelihood(pi, N, k) for pi in p])
    likelihood /= np.mean(likelihood)

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
             label=f"Posterior Beta({alpha_post},{beta_post})")

    # Likelihood
    plt.plot(p, likelihood, color="gray", linestyle="--",
             label="Scaled likelihood")

    # True p
    plt.axvline(p_true, color="red", linestyle=":",
                label=f"True p = {p_true}")

    plt.title("Posterior With Poisson-Jiggled Posterior Mean Histogram")
    plt.xlabel("p")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
