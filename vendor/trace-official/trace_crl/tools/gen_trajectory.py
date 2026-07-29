"""
Generate diverse transition trajectory data.

Output format matches infer.py expectations:
- Saved to: {data_path}/transition/traj_{K}domain_{d0}_{d1}_{d2}_{type}.npz
- Contains: xt, yt, alpha, domains

Usage:
    python -m trace.tools.gen_trajectory -k 3 -s 42
    python -m trace.tools.gen_trajectory -k 3 -s 42 -t all
    python -m trace.tools.gen_trajectory -k 3 -s 42 -t simple medium complex
"""

import os
import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform


# ============================================================
# Basic Functions
# ============================================================

def leaky_ReLU(D, negSlope):
    return np.where(D > 0, D, D * negSlope)


def mixing_function(y, mixingList, negSlope):
    mixedDat = np.copy(y)
    for A in mixingList:
        mixedDat = leaky_ReLU(mixedDat, negSlope)
        mixedDat = np.dot(mixedDat, A)
    return mixedDat


def load_generation_params(path):
    """Load saved generation parameters."""
    params_dir = os.path.join(path, "params")

    params = np.load(os.path.join(params_dir, "params.npy"), allow_pickle=True).item()

    W_bases = [
        np.load(os.path.join(params_dir, "W_base_lag1.npy")),
        np.load(os.path.join(params_dir, "W_base_lag2.npy"))
    ]

    delta_Ws = []
    for k in range(params['NClass']):
        delta_W = np.load(os.path.join(params_dir, f"delta_W_{k}.npy"))
        delta_Ws.append(delta_W)

    mixingList = []
    for l in range(params['Nlayer'] - 1):
        A = np.load(os.path.join(params_dir, f"mixing_{l}.npy"))
        mixingList.append(A)

    return {
        'params': params,
        'W_bases': W_bases,
        'delta_Ws': delta_Ws,
        'mixingList': mixingList,
    }


# ============================================================
# Domain Selection (Max Separation)
# ============================================================

def compute_domain_means(data_path):
    """Compute domain means from training data."""
    data = np.load(os.path.join(data_path, 'data.npz'))
    xt = data['xt']
    ct = data['ct'].flatten()

    K_total = int(ct.max()) + 1
    domain_means = {}

    for d in range(K_total):
        mask = ct == d
        xt_domain = xt[mask]
        z_t = xt_domain[:, -1, :]  # Last timestep
        domain_means[d] = z_t.mean(axis=0)

    return domain_means, K_total


def select_max_separation_domains(domain_means, K_active, K_total):
    """
    Greedy selection of domains with maximum separation.
    """
    if K_active >= K_total:
        return list(range(K_total))

    # Compute pairwise distances
    means_array = np.array([domain_means[d] for d in range(K_total)])
    distances = squareform(pdist(means_array))

    # Find the pair with maximum distance
    max_dist = 0
    best_pair = (0, 1)
    for i in range(K_total):
        for j in range(i + 1, K_total):
            if distances[i, j] > max_dist:
                max_dist = distances[i, j]
                best_pair = (i, j)

    selected = list(best_pair)

    # Greedily add domains that maximize minimum distance to selected
    while len(selected) < K_active:
        best_domain = None
        best_min_dist = -1

        for d in range(K_total):
            if d in selected:
                continue
            min_dist = min(distances[d, s] for s in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_domain = d

        if best_domain is not None:
            selected.append(best_domain)

    return sorted(selected)


# ============================================================
# Trajectory Functions
# ============================================================

def traj_simple(T, active_domains, K_total):
    """
    Simple (Sequential): D_a -> D_b -> D_c
    At most 2 domains active at any time.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))

    if K == 1:
        alpha[:, active_domains[0]] = 1.0
        return alpha

    seg_len = T / (K - 1)
    for t in range(T):
        seg_idx = min(int(t / seg_len), K - 2)
        progress = (t - seg_idx * seg_len) / seg_len
        progress = np.clip(progress, 0, 1)

        alpha[t, active_domains[seg_idx]] = 1.0 - progress
        alpha[t, active_domains[seg_idx + 1]] = progress

    return alpha


def traj_medium(T, active_domains, K_total):
    """
    Medium (Gaussian Overlapping): Smooth Gaussian transitions.
    Multiple domains can be active simultaneously.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_axis = np.arange(T)

    for i, d in enumerate(active_domains):
        peak_t = i * (T - 1) / (K - 1) if K > 1 else T / 2
        sigma = T / (K + 1)
        alpha[:, d] = np.exp(-0.5 * ((t_axis - peak_t) / sigma) ** 2)

    # Normalize
    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def traj_complex(T, active_domains, K_total):
    """
    Complex (Cosine Oscillating): Non-monotonic oscillations.
    All domains always contribute.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_norm = np.linspace(0, 1, T)

    for i, d in enumerate(active_domains):
        freq = 1 + i * 0.5
        phase = i * np.pi / K
        alpha[:, d] = 0.5 * (1 + np.cos(freq * 2 * np.pi * t_norm + phase))

    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def traj_step(T, active_domains, K_total):
    """
    Step (Piecewise Constant): Abrupt transitions.
    Only one domain active at a time.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    seg_len = T // K

    for t in range(T):
        seg_idx = min(t // seg_len, K - 1)
        alpha[t, active_domains[seg_idx]] = 1.0

    return alpha


def traj_sawtooth(T, active_domains, K_total):
    """
    Sawtooth: Linear ramp, sharp reset.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_norm = np.linspace(0, 1, T)

    for i, d in enumerate(active_domains):
        phase = i / K
        freq = 2
        wave = (freq * t_norm + phase) % 1.0
        alpha[:, d] = wave

    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def traj_exponential(T, active_domains, K_total):
    """
    Exponential: First decays, last grows, middle bell-shaped.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_norm = np.linspace(0, 1, T)

    for i, d in enumerate(active_domains):
        if i == 0:
            alpha[:, d] = np.exp(-5 * t_norm)
        elif i == K - 1:
            alpha[:, d] = 1 - np.exp(-5 * t_norm)
        else:
            center = i / (K - 1)
            alpha[:, d] = np.exp(-20 * (t_norm - center) ** 2)

    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def traj_sigmoid(T, active_domains, K_total):
    """
    Sigmoid: Smooth S-curve transitions.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_norm = np.linspace(0, 1, T)

    for i, d in enumerate(active_domains):
        center = i / (K - 1) if K > 1 else 0.5
        steepness = 10
        alpha[:, d] = 1 / (1 + np.exp(-steepness * (t_norm - center)))

    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def traj_pulse(T, active_domains, K_total):
    """
    Pulse: Sharp Gaussian pulses, domains take turns.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_axis = np.arange(T)

    for i, d in enumerate(active_domains):
        peak_t = i * (T - 1) / (K - 1) if K > 1 else T / 2
        sigma = T / (K * 3)  # Sharper than medium
        alpha[:, d] = np.exp(-0.5 * ((t_axis - peak_t) / sigma) ** 2)

    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def traj_triangle(T, active_domains, K_total):
    """
    Triangle wave: Linear rise and fall.
    """
    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_norm = np.linspace(0, 1, T)

    for i, d in enumerate(active_domains):
        phase = i / K
        wave = 1 - 2 * np.abs((t_norm + phase) % 1.0 - 0.5)
        alpha[:, d] = wave

    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def traj_random_walk(T, active_domains, K_total, seed=None):
    """
    Random walk: Brownian motion style.
    """
    if seed is not None:
        np.random.seed(seed)

    K = len(active_domains)
    alpha = np.zeros((T, K_total))
    weights = np.ones(K) / K

    for t in range(T):
        step = np.random.randn(K) * 0.05
        weights = np.clip(weights + step, 0.01, None)
        weights = weights / weights.sum()

        for i, d in enumerate(active_domains):
            alpha[t, d] = weights[i]

    return alpha


# Registry
TRAJECTORY_FUNCTIONS = {
    'simple': traj_simple,
    'medium': traj_medium,
    'complex': traj_complex,
}


# ============================================================
# Data Generation
# ============================================================

def generate_trajectory_data(gen_params, alpha_sequence, batch_size=500, seed=42):
    """Generate observation data for a given alpha trajectory."""
    np.random.seed(seed)

    params = gen_params['params']
    W_bases = gen_params['W_bases']
    delta_Ws = gen_params['delta_Ws']
    mixingList = gen_params['mixingList']

    negSlope = params['negSlope']
    noise_scale = params['noise_scale']
    latent_size = params['latent_size']
    lags = params['lags']
    NClass = params['NClass']

    T = alpha_sequence.shape[0]

    yt_all = []
    xt_all = []

    for t in range(T):
        alpha = alpha_sequence[t]

        # Fresh initial state each timestep
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0, keepdims=True)

        # Mixed transition matrix
        W_mixed = W_bases[0].copy()
        for k in range(NClass):
            W_mixed += alpha[k] * delta_Ws[k]

        # Generate one step
        y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
        y_t += leaky_ReLU(np.dot(y_l[:, -1, :], W_mixed), negSlope)
        y_t += leaky_ReLU(np.dot(y_l[:, -2, :], W_bases[1]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)

        y_seq = np.concatenate([y_l, y_t[:, np.newaxis, :]], axis=1)
        x_seq = mixing_function(y_seq, mixingList, negSlope)

        yt_all.append(y_seq)
        xt_all.append(x_seq)

    yt_all = np.array(yt_all).transpose(1, 0, 2, 3)
    xt_all = np.array(xt_all).transpose(1, 0, 2, 3)

    return {'yt': yt_all, 'xt': xt_all}


def generate_all_trajectories(
    data_path,
    K_active,
    traj_types,
    T=50,
    batch_size=500,
    n_samples=5,
    seed=42
):
    """Generate and save all trajectory types."""
    np.random.seed(seed)

    # Load params
    gen_params = load_generation_params(data_path)
    K_total = gen_params['params']['NClass']

    # Select domains with max separation
    domain_means, _ = compute_domain_means(data_path)
    active_domains = select_max_separation_domains(domain_means, K_active, K_total)

    print(f"\nK_total = {K_total}, K_active = {K_active}")
    print(f"Selected domains (max separation): {active_domains}")

    # Output directory - use 'transition' to match infer.py expectations
    output_dir = os.path.join(data_path, "transition")
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata = {
        'K_total': K_total,
        'K_active': K_active,
        'domains': active_domains,  # Key name matches infer.py expectation
        'T': T,
        'batch_size': batch_size,
        'n_samples': n_samples,
        'seed': seed,
        'traj_types': traj_types,
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)

    # Generate each type
    for traj_type in traj_types:
        print(f"\n  Generating '{traj_type}' trajectories...")

        traj_func = TRAJECTORY_FUNCTIONS[traj_type]

        for sample_idx in range(n_samples):
            sample_seed = seed + sample_idx * 100

            # Generate alpha
            if traj_type == 'random_walk':
                alpha = traj_func(T, active_domains, K_total, seed=sample_seed)
            else:
                alpha = traj_func(T, active_domains, K_total)

            # Generate observations
            data = generate_trajectory_data(
                gen_params, alpha, batch_size=batch_size, seed=sample_seed
            )

            # Save with naming format compatible with infer.py
            # Format: traj_{K}domain_{d0}_{d1}_{...}.npz
            domain_str = '_'.join(map(str, active_domains))
            if n_samples > 1:
                save_path = os.path.join(output_dir, f'traj_{K_active}domain_{domain_str}_{traj_type}_{sample_idx}.npz')
            else:
                save_path = os.path.join(output_dir, f'traj_{K_active}domain_{domain_str}_{traj_type}.npz')

            np.savez(
                save_path,
                xt=data['xt'],
                yt=data['yt'],
                alpha=alpha,
                domains=np.array(active_domains),  # Key name matches infer.py expectation
            )

            print(f"    Saved: {os.path.basename(save_path)}")

    print(f"\nAll saved to: {output_dir}/")
    return output_dir


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate trajectory data')
    parser.add_argument('-d', '--domains', type=int, default=5,
                        help='Number of total domains in dataset (default: 5)')
    parser.add_argument('-k', '--k_active', type=int, default=3,
                        help='Number of active domains in trajectory (default: 3)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('-t', '--traj_types', nargs='+', default=['simple', 'medium', 'complex'],
                        help='Trajectory types (default: simple medium complex)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to training data (overrides -d)')
    parser.add_argument('-T', '--timesteps', type=int, default=50,
                        help='Number of timesteps (default: 50)')
    parser.add_argument('-n', '--n_samples', type=int, default=1,
                        help='Samples per type (default: 1)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size (default: 500)')

    args = parser.parse_args()

    # Find data path
    if args.data_path is None:
        args.data_path = f'datasets/{args.domains}_domains'
        if not os.path.exists(args.data_path):
            print(f"Error: Dataset not found: {args.data_path}")
            print("Run gen_dataset first:")
            print(f"  python -m trace.tools.gen_dataset -d {args.domains}")
            exit(1)

    # Get K_total
    gen_params = load_generation_params(args.data_path)
    K_total = gen_params['params']['NClass']

    if args.k_active > K_total:
        print(f"Warning: K_active ({args.k_active}) > K_total ({K_total}), using {K_total}")
        args.k_active = K_total

    # Trajectory types
    if 'all' in args.traj_types:
        traj_types = list(TRAJECTORY_FUNCTIONS.keys())
    else:
        traj_types = args.traj_types
        for t in traj_types:
            if t not in TRAJECTORY_FUNCTIONS:
                print(f"Error: Unknown type '{t}'")
                print(f"Available: {list(TRAJECTORY_FUNCTIONS.keys())}")
                exit(1)

    print("=" * 60)
    print("Trajectory Generation")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"K_total: {K_total}, K_active: {args.k_active}")
    print(f"Types: {traj_types}")
    print(f"T={args.timesteps}, samples={args.n_samples}, seed={args.seed}")

    generate_all_trajectories(
        data_path=args.data_path,
        K_active=args.k_active,
        traj_types=traj_types,
        T=args.timesteps,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
