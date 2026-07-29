"""
Ablation Study: K_active and Trajectory Complexity

Experiments:
1. Fixed K_total=10, vary K_active and trajectory type
2. Compare K_total=5 vs K_total=10
3. K_total=10, K_active from 2 to 10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from scipy.spatial.distance import euclidean

from trace_crl.tools.utils import load_yaml

def load_inference_config():
    """Load inference config from trace/configs/inference.yaml"""
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '../trace_crl/configs/inference.yaml')
    return load_yaml(config_path)


# ============================================================
# Trajectory Generation Functions
# ============================================================

def generate_simple_trajectory(T, active_domains, K_total):
    """
    Simple (Sequential) trajectory:
    - Linear sequential transition: D_a -> D_b -> D_c -> ...
    - At most 2 domains have non-zero alpha at any time
    - Each active domain reaches alpha=1.0 sequentially
    """
    K_active = len(active_domains)
    alpha = np.zeros((T, K_total))

    # Divide time into K_active-1 segments
    segment_length = T / (K_active - 1) if K_active > 1 else T

    for t in range(T):
        if K_active == 1:
            alpha[t, active_domains[0]] = 1.0
        else:
            # Find which segment we're in
            segment_idx = min(int(t / segment_length), K_active - 2)
            progress = (t - segment_idx * segment_length) / segment_length
            progress = min(max(progress, 0), 1)

            # Linear interpolation between two domains
            d_from = active_domains[segment_idx]
            d_to = active_domains[segment_idx + 1]

            alpha[t, d_from] = 1.0 - progress
            alpha[t, d_to] = progress

    return alpha


def generate_medium_trajectory(T, active_domains, K_total):
    """
    Medium (Overlapping) trajectory:
    - Gaussian-shaped smooth transitions
    - Multiple domains can be active simultaneously (overlapping)
    - Each domain has a clear peak moment, peak value ~0.6-0.8
    - Low frequency changes, no oscillation
    """
    K_active = len(active_domains)
    alpha = np.zeros((T, K_total))

    t_axis = np.arange(T)

    # Each domain peaks at evenly spaced time points
    for i, d in enumerate(active_domains):
        if K_active == 1:
            peak_t = T / 2
        else:
            peak_t = i * (T - 1) / (K_active - 1)

        # Gaussian with sigma proportional to spacing
        sigma = T / (K_active + 1)
        alpha[:, d] = np.exp(-0.5 * ((t_axis - peak_t) / sigma) ** 2)

    # Normalize to simplex
    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    alpha = alpha / alpha_sum

    return alpha


def generate_complex_trajectory(T, active_domains, K_total):
    """
    Complex (Oscillating) trajectory:
    - Superposition of cosine waves with different frequencies
    - All active domains always contribute (no alpha=0 moment)
    - Has oscillation, non-monotonic
    - Max alpha ~0.4-0.6, no dominant domain
    """
    K_active = len(active_domains)
    alpha = np.zeros((T, K_total))

    t_normalized = np.linspace(0, 1, T)

    for i, d in enumerate(active_domains):
        freq = 1 + i * 0.5
        phase = i * np.pi / K_active
        alpha[:, d] = 0.5 * (1 + np.cos(freq * 2 * np.pi * t_normalized + phase))

    # Normalize to simplex
    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    alpha = alpha / alpha_sum

    return alpha


# ============================================================
# Domain Selection (Max Separation Strategy)
# ============================================================

def select_domains_for_K5(K_active):
    """Domain selection for K_total=5"""
    selection_map = {
        2: [0, 4],
        3: [0, 2, 4],
        4: [0, 1, 3, 4],
        5: [0, 1, 2, 3, 4],
    }
    return selection_map.get(K_active, list(range(K_active)))


def select_domains_for_K10(K_active):
    """Domain selection for K_total=10 (max separation)"""
    selection_map = {
        2: [0, 6],
        3: [0, 3, 6],
        4: [0, 3, 4, 6],
        5: [0, 1, 3, 4, 6],
        6: [0, 1, 2, 3, 4, 6],
        7: [0, 1, 2, 3, 4, 6, 8],
        8: [0, 1, 2, 3, 4, 6, 8, 9],
        9: [0, 1, 2, 3, 4, 5, 6, 8, 9],
        10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    return selection_map.get(K_active, list(range(K_active)))


# ============================================================
# Data Generation
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

        # Fresh y_l at each time step
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0, keepdims=True)

        # Mixed transition matrix
        W_mixed = W_bases[0].copy()
        for k in range(NClass):
            W_mixed += alpha[k] * delta_Ws[k]

        # Generate ONE step
        y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
        y_t += leaky_ReLU(np.dot(y_l[:, -1, :], W_mixed), negSlope)
        y_t += leaky_ReLU(np.dot(y_l[:, -2, :], W_bases[1]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)

        # Build sequence
        y_seq = np.concatenate([y_l, y_t[:, np.newaxis, :]], axis=1)
        x_seq = mixing_function(y_seq, mixingList, negSlope)

        yt_all.append(y_seq)
        xt_all.append(x_seq)

    yt_all = np.array(yt_all).transpose(1, 0, 2, 3)
    xt_all = np.array(xt_all).transpose(1, 0, 2, 3)

    return {'yt': yt_all, 'xt': xt_all}


# ============================================================
# Model Loading and Inference
# ============================================================

def load_model(ckpt_path, nclass, embedding_dim=8):
    from trace_crl.modules.change import TimeVaryingProcess

    ckpt = torch.load(ckpt_path)
    model = TimeVaryingProcess(
        input_dim=8, length=1, z_dim=8, lag=2, nclass=nclass,
        hidden_dim=128, embedding_dim=embedding_dim, trans_prior='NP',
        decoder_dist='gaussian', correlation='Pearson'
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.cuda()
    return model


def encode(model, x):
    B, T, x_dim = x.shape
    x_flat = x.reshape(-1, x_dim)
    with torch.no_grad():
        _, mus, _, _ = model.net(x_flat)
    return mus.reshape(B, T, -1)


def compute_domain_means(model, training_data_path, domains):
    data = np.load(training_data_path)
    xt = torch.FloatTensor(data['xt']).cuda()
    ct = data['ct'].flatten()

    domain_means = {}
    for d in domains:
        mask = ct == d
        xt_domain = xt[mask]
        with torch.no_grad():
            z_encoded = encode(model, xt_domain)
        z_t = z_encoded[:, -1, :]
        domain_means[d] = z_t.mean(dim=0).cpu().numpy()

    return domain_means


def temporal_smoothing(alpha_sequence, window_size=5):
    T, K = alpha_sequence.shape
    alpha_smooth = np.zeros_like(alpha_sequence)
    for t in range(T):
        start = max(0, t - window_size)
        end = min(T, t + window_size + 1)
        alpha_smooth[t] = alpha_sequence[start:end].mean(axis=0)
    return alpha_smooth


def calibrate_alpha(alpha_pred, alpha_true):
    K = alpha_pred.shape[1]
    alpha_calibrated = np.zeros_like(alpha_pred)
    for k in range(K):
        pred_k = alpha_pred[:, k]
        true_k = alpha_true[:, k]
        min_val, max_val = pred_k.min(), pred_k.max()
        if max_val - min_val > 1e-8:
            true_min, true_max = true_k.min(), true_k.max()
            alpha_calibrated[:, k] = (pred_k - min_val) / (max_val - min_val) * (true_max - true_min) + true_min
        else:
            alpha_calibrated[:, k] = pred_k
    return alpha_calibrated


def run_inference(model, domain_means, xt, alpha_true, active_domains, window_size=5):
    """Run inference and compute metrics for active domains only."""
    xt_tensor = torch.FloatTensor(xt).cuda()
    B, T, seq_len, z_dim = xt_tensor.shape
    K_active = len(active_domains)

    # Construct differential basis
    baseline = active_domains[0]
    mu_0 = domain_means[baseline]

    delta_mus = []
    for d in active_domains[1:]:
        delta_mus.append(domain_means[d] - mu_0)
    B_matrix = np.stack(delta_mus, axis=1) if delta_mus else np.zeros((len(mu_0), 0))

    # Encode
    xt_flat = xt_tensor.reshape(B * T, seq_len, z_dim)
    z_encoded_flat = encode(model, xt_flat)
    z_encoded = z_encoded_flat.reshape(B, T, seq_len, -1)
    z_t_all = z_encoded[:, :, -1, :].cpu().numpy()
    z_t_mean = z_t_all.mean(axis=0)

    # Solve for alpha
    alpha_pred = []
    B_pinv = np.linalg.pinv(B_matrix) if B_matrix.size > 0 else None

    for t in range(T):
        z_t = z_t_mean[t]
        residual = z_t - mu_0

        if B_pinv is not None:
            alpha_raw = B_pinv @ residual
            alpha_0 = 1.0 - alpha_raw.sum()
            alpha_full = np.concatenate([[alpha_0], alpha_raw])
        else:
            alpha_full = np.array([1.0])

        # Project to simplex
        alpha_full = np.clip(alpha_full, 0, None)
        if alpha_full.sum() < 1e-10:
            alpha_full = np.ones(K_active) / K_active
        else:
            alpha_full = alpha_full / alpha_full.sum()

        alpha_pred.append(alpha_full)

    alpha_pred = np.array(alpha_pred)

    # Smooth
    alpha_smooth = temporal_smoothing(alpha_pred, window_size)

    # Get ground truth for active domains only
    alpha_true_active = alpha_true[:, active_domains]

    # Calibrate
    alpha_calibrated = calibrate_alpha(alpha_smooth, alpha_true_active)

    # Compute metrics (only for active domains)
    mae = np.mean(np.abs(alpha_calibrated - alpha_true_active))

    corrs = []
    for i in range(K_active):
        corr = np.corrcoef(alpha_smooth[:, i], alpha_true_active[:, i])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    corr_mean = np.mean(corrs) if corrs else 0.0

    return {
        'correlation': corr_mean,
        'mae': mae,
        'alpha_pred': alpha_smooth,
        'alpha_true': alpha_true_active,
    }


# ============================================================
# Experiment Functions
# ============================================================

def run_single_experiment(model, gen_params, training_data_path, K_total,
                          K_active, traj_type, n_runs=10, seed=42):
    """Run a single experiment configuration."""

    # Select domains
    if K_total == 5:
        active_domains = select_domains_for_K5(K_active)
    else:
        active_domains = select_domains_for_K10(K_active)

    # Generate trajectory function
    traj_funcs = {
        'simple': generate_simple_trajectory,
        'medium': generate_medium_trajectory,
        'complex': generate_complex_trajectory,
    }

    T = 50
    alpha_sequence = traj_funcs[traj_type](T, active_domains, K_total)

    # Compute domain means
    domain_means = compute_domain_means(model, training_data_path, active_domains)

    correlations = []
    maes = []

    for run_idx in range(n_runs):
        run_seed = seed + run_idx * 100

        # Generate data
        data = generate_trajectory_data(gen_params, alpha_sequence, batch_size=500, seed=run_seed)

        # Run inference
        result = run_inference(model, domain_means, data['xt'], alpha_sequence, active_domains)

        correlations.append(result['correlation'])
        maes.append(result['mae'])

    return {
        'K_total': K_total,
        'K_active': K_active,
        'traj_type': traj_type,
        'active_domains': active_domains,
        'corr_mean': np.mean(correlations),
        'corr_std': np.std(correlations),
        'mae_mean': np.mean(maes),
        'mae_std': np.std(maes),
    }


def experiment1_K_active_and_trajectory():
    """Experiment 1: Fixed K_total=10, vary K_active and trajectory type"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: K_total=10, varying K_active and Trajectory Type")
    print("="*80)

    cfg = load_inference_config()
    ckpt_path = cfg['CHECKPOINTS']['10_DOMAINS']
    data_path = cfg['DATASETS']['10_DOMAINS']
    training_data_path = os.path.join(data_path, 'data.npz')
    model_cfg = cfg['MODELS']['10_DOMAINS']

    model = load_model(ckpt_path, nclass=model_cfg['NCLASS'], embedding_dim=model_cfg['EMBED_DIM'])
    gen_params = load_generation_params(data_path)

    results = []

    for K_active in [2, 3, 4, 5]:
        for traj_type in ['simple', 'medium', 'complex']:
            print(f"  Running: K_active={K_active}, traj={traj_type}...")
            result = run_single_experiment(
                model, gen_params, training_data_path,
                K_total=10, K_active=K_active, traj_type=traj_type
            )
            results.append(result)
            print(f"    Corr={result['corr_mean']:.4f}±{result['corr_std']:.4f}, "
                  f"MAE={result['mae_mean']:.4f}±{result['mae_std']:.4f}")

    # Print summary table
    print("\n" + "-"*80)
    print(f"{'K_active':<10} {'Trajectory':<12} {'Correlation':<20} {'MAE':<20}")
    print("-"*80)
    for r in results:
        print(f"{r['K_active']:<10} {r['traj_type']:<12} "
              f"{r['corr_mean']:.4f}±{r['corr_std']:.4f}      "
              f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}")

    return results


def experiment2_K_total_comparison():
    """Experiment 2: Compare K_total=5 vs K_total=10"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: K_total=5 vs K_total=10 Comparison")
    print("="*80)

    cfg = load_inference_config()
    configs = [
        (5, cfg['CHECKPOINTS']['5_DOMAINS'],
         cfg['DATASETS']['5_DOMAINS'], cfg['MODELS']['5_DOMAINS']['EMBED_DIM']),
        (10, cfg['CHECKPOINTS']['10_DOMAINS'],
         cfg['DATASETS']['10_DOMAINS'], cfg['MODELS']['10_DOMAINS']['EMBED_DIM']),
    ]

    results = []

    for K_total, ckpt_path, data_path, embed_dim in configs:
        print(f"\n  Loading K_total={K_total} model...")
        model = load_model(ckpt_path, nclass=K_total, embedding_dim=embed_dim)
        gen_params = load_generation_params(data_path)
        training_data_path = os.path.join(data_path, 'data.npz')

        max_K_active = min(5, K_total)
        for K_active in range(2, max_K_active + 1):
            for traj_type in ['simple', 'medium', 'complex']:
                print(f"    Running: K_total={K_total}, K_active={K_active}, traj={traj_type}...")
                result = run_single_experiment(
                    model, gen_params, training_data_path,
                    K_total=K_total, K_active=K_active, traj_type=traj_type
                )
                results.append(result)
                print(f"      Corr={result['corr_mean']:.4f}±{result['corr_std']:.4f}")

    # Print summary table
    print("\n" + "-"*80)
    print(f"{'K_total':<10} {'K_active':<10} {'Trajectory':<12} {'Correlation':<20} {'MAE':<20}")
    print("-"*80)
    for r in results:
        print(f"{r['K_total']:<10} {r['K_active']:<10} {r['traj_type']:<12} "
              f"{r['corr_mean']:.4f}±{r['corr_std']:.4f}      "
              f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}")

    return results


def experiment3_full_K_scaling():
    """Experiment 3: K_total=10, K_active from 2 to 10"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: K_total=10, K_active from 2 to 10 (Full Scaling)")
    print("="*80)

    cfg = load_inference_config()
    ckpt_path = cfg['CHECKPOINTS']['10_DOMAINS']
    data_path = cfg['DATASETS']['10_DOMAINS']
    training_data_path = os.path.join(data_path, 'data.npz')
    model_cfg = cfg['MODELS']['10_DOMAINS']

    model = load_model(ckpt_path, nclass=model_cfg['NCLASS'], embedding_dim=model_cfg['EMBED_DIM'])
    gen_params = load_generation_params(data_path)

    results = []

    for K_active in range(2, 11):
        for traj_type in ['simple', 'medium', 'complex']:
            print(f"  Running: K_active={K_active}, traj={traj_type}...")
            result = run_single_experiment(
                model, gen_params, training_data_path,
                K_total=10, K_active=K_active, traj_type=traj_type
            )
            results.append(result)
            print(f"    Corr={result['corr_mean']:.4f}±{result['corr_std']:.4f}, "
                  f"MAE={result['mae_mean']:.4f}±{result['mae_std']:.4f}")

    # Print summary table
    print("\n" + "-"*80)
    print(f"{'K_active':<10} {'Trajectory':<12} {'Correlation':<20} {'MAE':<20} {'Domains'}")
    print("-"*80)
    for r in results:
        print(f"{r['K_active']:<10} {r['traj_type']:<12} "
              f"{r['corr_mean']:.4f}±{r['corr_std']:.4f}      "
              f"{r['mae_mean']:.4f}±{r['mae_std']:.4f}   {r['active_domains']}")

    return results


def main():
    print("="*80)
    print("ABLATION STUDY: K_active and Trajectory Complexity")
    print("="*80)

    all_results = {}

    # Run all experiments
    all_results['experiment1'] = experiment1_K_active_and_trajectory()
    all_results['experiment2'] = experiment2_K_total_comparison()
    all_results['experiment3'] = experiment3_full_K_scaling()

    # Save results
    save_path = 'results/ablation_K_trajectory.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert to JSON-serializable format
    def convert_to_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json(v) for v in obj]
        else:
            return obj

    with open(save_path, 'w') as f:
        json.dump(convert_to_json(all_results), f, indent=2)
    print(f"\nResults saved to: {save_path}")

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)

    return all_results


if __name__ == "__main__":
    main()
