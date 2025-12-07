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
        help="Number of coin flips (default: 1,000,000)"
    )
    parser.add_argument(
        "-p", "--prob",
        type=float,
        default=0.3,
        help="True probability of heads (default: 0.3)"
    )

    args = parser.parse_args()
    N = args.num_flips
    p_true = args.prob

    # Simulate N flips
    k = np.random.binomial(N, p_true)

    print(f"Flipped the coin {N} times with p={p_true}")
    print(f"Number of heads observed: {k}")

    # Prior: Beta(1,1)
    alpha_prior = 1
    beta_prior = 1

    # Posterior parameters
    alpha_post = alpha_prior + k
    beta_post  = beta_prior  + (N - k)

    # Grid of p values
    p = np.linspace(0, 1, 500)

    # Likelihood (scaled)
    likelihood = np.array([binomial_likelihood(pi, N, k) for pi in p])
    likelihood /= np.mean(likelihood)  # Because X is [0, 1] mean works

    # PDFs
    prior_pdf = beta.pdf(p, alpha_prior, beta_prior)
    post_pdf  = beta.pdf(p, alpha_post, beta_post)

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(p, likelihood, label=f"Likelihood (k={k}, N={N})", color="gray", linestyle="--")
    plt.axvline(p_true, color="red", linestyle=":", label=f"True p = {p_true}")

    plt.title("Bayesian Inference of a Coin's Bias")
    plt.xlabel("p (probability of heads)")
    plt.ylabel("pdf")
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
