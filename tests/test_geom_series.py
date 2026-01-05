import numpy as np
import matplotlib.pyplot as plt

noise = 5000
dc = 10

betas = [0.1, 0.5, 0.9, 0.99, 0.999]
series = {}

fig = plt.figure()

for beta in betas:
    S = 0.0
    series[beta] = [S]
    NUM_ITERS = 5000
    # Sample +-1 
    zs = np.random.choice([1, -1], size=NUM_ITERS)

    for i, z in enumerate(zs):
        S = beta * S + (1 - beta) * z
        series[beta].append(S / (1 - beta ** (i + 1)))

    plt.plot(np.arange(len(series[beta])), series[beta], label=f"beta = {beta}")

plt.legend()
plt.savefig("figures/geom_series.png")


betas = [0.1, 0.5, 0.9, 0.99, 0.999]
series = {}

fig = plt.figure()

for beta in betas:
    S = 0.0
    series[beta] = [S]
    NUM_ITERS = 5000
    # Sample normal
    zs = np.random.randn(NUM_ITERS) * 2

    for i, z in enumerate(zs):
        S = beta * S + (1 - beta) * z
        series[beta].append(S / (1 - beta ** (i + 1)))

    plt.plot(np.arange(len(series[beta])), series[beta], label=f"beta = {beta}")

plt.legend()
plt.savefig("figures/geom_series_normal.png")

fig = plt.figure()

for beta in betas:
    S = 0.0
    series[beta] = [S]
    NUM_ITERS = 5000
    # Sample normal
    zs = np.random.choice([1, -1], size=NUM_ITERS)

    start_const = 100
    end_const = 5

    for i, z in enumerate(zs):
        S = beta * S + (1 - beta) * (z * (end_const + (start_const - end_const) * (1 - i / NUM_ITERS)) * noise + dc)
        series[beta].append(S / (1 - beta ** (i + 1)))

    plt.plot(np.arange(len(series[beta])), series[beta], label=f"beta = {beta}")

plt.legend()
plt.savefig("figures/geom_series_changing.png")
