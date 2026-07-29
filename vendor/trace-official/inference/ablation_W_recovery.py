"""
Ablation Study: W Matrix Recovery vs Alpha Recovery

Key insight from TRACE paper:
Even when individual alpha_k(t) recovery fails at high K_active,
the combined effect W(t) = W_base + sum_k alpha_k(t) * delta_W^(k)
might still be accurately recoverable.

This script compares:
1. Alpha correlation: corr(alpha_true, alpha_pred)
2. W correlation: corr(W_true(t), W_pred(t)) for each time step
3. W relative error: ||W_true - W_pred||_F / ||W_true - W_base||_F
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from trace_crl.tools.utils import load_yaml

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================
# Trajectory Generation (same as ablation_K_trajectory.py)
# ============================================================

def generate_simple_trajectory(T, active_domains, K_total):
    """Sequential trajectory: D_a -> D_b -> D_c -> ..."""
    K_active = len(active_domains)
    alpha = np.zeros((T, K_total))
    segment_length = T / (K_active - 1) if K_active > 1 else T

    for t in range(T):
        if K_active == 1:
            alpha[t, active_domains[0]] = 1.0
        else:
            segment_idx = min(int(t / segment_length), K_active - 2)
            progress = (t - segment_idx * segment_length) / segment_length
            progress = min(max(progress, 0), 1)
            d_from = active_domains[segment_idx]
            d_to = active_domains[segment_idx + 1]
            alpha[t, d_from] = 1.0 - progress
            alpha[t, d_to] = progress
    return alpha


def generate_medium_trajectory(T, active_domains, K_total):
    """Overlapping Gaussian trajectory."""
    K_active = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_axis = np.arange(T)

    for i, d in enumerate(active_domains):
        peak_t = T / 2 if K_active == 1 else i * (T - 1) / (K_active - 1)
        sigma = T / (K_active + 1)
        alpha[:, d] = np.exp(-0.5 * ((t_axis - peak_t) / sigma) ** 2)

    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def generate_complex_trajectory(T, active_domains, K_total):
    """Oscillating cosine trajectory."""
    K_active = len(active_domains)
    alpha = np.zeros((T, K_total))
    t_normalized = np.linspace(0, 1, T)

    for i, d in enumerate(active_domains):
        freq = 1 + i * 0.5
        phase = i * np.pi / K_active
        alpha[:, d] = 0.5 * (1 + np.cos(freq * 2 * np.pi * t_normalized + phase))

    alpha_sum = alpha.sum(axis=1, keepdims=True)
    alpha_sum[alpha_sum < 1e-10] = 1.0
    return alpha / alpha_sum


def select_domains_for_K10(K_active):
    """Domain selection for K_total=10 (max separation)."""
    selection_map = {
        2: [0, 6], 3: [0, 3, 6], 4: [0, 3, 4, 6], 5: [0, 1, 3, 4, 6],
        6: [0, 1, 2, 3, 4, 6], 7: [0, 1, 2, 3, 4, 6, 8],
        8: [0, 1, 2, 3, 4, 6, 8, 9], 9: [0, 1, 2, 3, 4, 5, 6, 8, 9],
        10: list(range(10)),
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
    yt_all, xt_all = [], []

    for t in range(T):
        alpha = alpha_sequence[t]
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0, keepdims=True)

        W_mixed = W_bases[0].copy()
        for k in range(NClass):
            W_mixed += alpha[k] * delta_Ws[k]

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


# ============================================================
# Model and Inference
# ============================================================

def load_model(ckpt_path, nclass, embedding_dim=8):
    from trace_crl.modules.change import TimeVaryingProcess

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = TimeVaryingProcess(
        input_dim=8, length=1, z_dim=8, lag=2, nclass=nclass,
        hidden_dim=128, embedding_dim=embedding_dim, trans_prior='NP',
        decoder_dist='gaussian', correlation='Pearson'
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(DEVICE)
    return model


def encode(model, x):
    B, T, x_dim = x.shape
    x_flat = x.reshape(-1, x_dim)
    with torch.no_grad():
        _, mus, _, _ = model.net(x_flat)
    return mus.reshape(B, T, -1)


def compute_domain_means(model, training_data_path, domains):
    data = np.load(training_data_path)
    xt = torch.FloatTensor(data['xt']).to(DEVICE)
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


def run_inference(model, domain_means, xt, alpha_true, active_domains, window_size=5):
    """Run inference and return both alpha predictions and raw data for W computation."""
    xt_tensor = torch.FloatTensor(xt).to(DEVICE)
    B, T, seq_len, z_dim = xt_tensor.shape
    K_active = len(active_domains)

    baseline = active_domains[0]
    mu_0 = domain_means[baseline]

    delta_mus = []
    for d in active_domains[1:]:
        delta_mus.append(domain_means[d] - mu_0)
    B_matrix = np.stack(delta_mus, axis=1) if delta_mus else np.zeros((len(mu_0), 0))

    xt_flat = xt_tensor.reshape(B * T, seq_len, z_dim)
    z_encoded_flat = encode(model, xt_flat)
    z_encoded = z_encoded_flat.reshape(B, T, seq_len, -1)
    z_t_all = z_encoded[:, :, -1, :].cpu().numpy()
    z_t_mean = z_t_all.mean(axis=0)

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

        alpha_full = np.clip(alpha_full, 0, None)
        if alpha_full.sum() < 1e-10:
            alpha_full = np.ones(K_active) / K_active
        else:
            alpha_full = alpha_full / alpha_full.sum()

        alpha_pred.append(alpha_full)

    alpha_pred = np.array(alpha_pred)
    alpha_smooth = temporal_smoothing(alpha_pred, window_size)
    alpha_true_active = alpha_true[:, active_domains]

    # Compute alpha correlation
    corrs = []
    for i in range(K_active):
        corr = np.corrcoef(alpha_smooth[:, i], alpha_true_active[:, i])[0, 1]
        if not np.isnan(corr):
            corrs.append(corr)
    alpha_corr = np.mean(corrs) if corrs else 0.0

    return {
        'alpha_pred': alpha_smooth,
        'alpha_true': alpha_true_active,
        'alpha_corr': alpha_corr,
        'active_domains': active_domains,
    }


# ============================================================
# W Recovery Computation
# ============================================================

def compute_W_matrices(alpha_sequence, active_domains, gen_params):
    """
    Compute W(t) = W_base + sum_k alpha_k(t) * delta_W^(k)

    Args:
        alpha_sequence: (T, K_active) alpha values for active domains
        active_domains: list of active domain indices
        gen_params: generation parameters containing W_base and delta_Ws

    Returns:
        W_sequence: (T, d, d) transition matrices
    """
    W_base = gen_params['W_bases'][0]  # Use lag-1 base matrix
    delta_Ws = gen_params['delta_Ws']

    T, K_active = alpha_sequence.shape
    d = W_base.shape[0]

    W_sequence = np.zeros((T, d, d))

    for t in range(T):
        W_t = W_base.copy()
        for i, domain_idx in enumerate(active_domains):
            W_t = W_t + alpha_sequence[t, i] * delta_Ws[domain_idx]
        W_sequence[t] = W_t

    return W_sequence


def compute_W_metrics(W_true_seq, W_pred_seq, W_base):
    """
    Compute metrics comparing true and predicted W matrices.

    Returns:
        - W_corr: Average correlation between flattened W matrices
        - W_rel_error: Relative Frobenius error ||W_pred - W_true||_F / ||W_true - W_base||_F
        - W_abs_error: Absolute Frobenius error ||W_pred - W_true||_F
    """
    T = W_true_seq.shape[0]

    # Flatten for correlation
    W_true_flat = W_true_seq.reshape(T, -1)
    W_pred_flat = W_pred_seq.reshape(T, -1)

    # Per-timestep correlation
    corrs = []
    for t in range(T):
        corr, _ = pearsonr(W_true_flat[t], W_pred_flat[t])
        if not np.isnan(corr):
            corrs.append(corr)
    W_corr = np.mean(corrs) if corrs else 0.0

    # Also compute correlation across entire trajectory (flatten all)
    W_corr_global, _ = pearsonr(W_true_flat.flatten(), W_pred_flat.flatten())

    # Relative error
    diff_norms = []
    signal_norms = []
    for t in range(T):
        diff_norm = np.linalg.norm(W_pred_seq[t] - W_true_seq[t], 'fro')
        signal_norm = np.linalg.norm(W_true_seq[t] - W_base, 'fro')
        diff_norms.append(diff_norm)
        signal_norms.append(signal_norm)

    W_abs_error = np.mean(diff_norms)
    W_rel_error = np.mean(diff_norms) / (np.mean(signal_norms) + 1e-10)

    return {
        'W_corr_per_t': W_corr,
        'W_corr_global': W_corr_global,
        'W_rel_error': W_rel_error,
        'W_abs_error': W_abs_error,
    }


# ============================================================
# Main Experiment
# ============================================================

def run_single_experiment_with_W(model, gen_params, training_data_path, K_total,
                                  K_active, traj_type, n_runs=10, seed=42):
    """Run experiment and compute both alpha and W recovery metrics."""

    active_domains = select_domains_for_K10(K_active)

    traj_funcs = {
        'simple': generate_simple_trajectory,
        'medium': generate_medium_trajectory,
        'complex': generate_complex_trajectory,
    }

    T = 50
    alpha_sequence = traj_funcs[traj_type](T, active_domains, K_total)
    domain_means = compute_domain_means(model, training_data_path, active_domains)

    W_base = gen_params['W_bases'][0]

    alpha_corrs = []
    W_corrs_per_t = []
    W_corrs_global = []
    W_rel_errors = []

    for run_idx in range(n_runs):
        run_seed = seed + run_idx * 100

        # Generate data
        data = generate_trajectory_data(gen_params, alpha_sequence, batch_size=500, seed=run_seed)

        # Run inference
        result = run_inference(model, domain_means, data['xt'], alpha_sequence, active_domains)

        alpha_corrs.append(result['alpha_corr'])

        # Compute W matrices
        W_true_seq = compute_W_matrices(result['alpha_true'], active_domains, gen_params)
        W_pred_seq = compute_W_matrices(result['alpha_pred'], active_domains, gen_params)

        # Compute W metrics
        W_metrics = compute_W_metrics(W_true_seq, W_pred_seq, W_base)

        W_corrs_per_t.append(W_metrics['W_corr_per_t'])
        W_corrs_global.append(W_metrics['W_corr_global'])
        W_rel_errors.append(W_metrics['W_rel_error'])

    return {
        'K_active': K_active,
        'traj_type': traj_type,
        'active_domains': active_domains,
        # Alpha metrics
        'alpha_corr_mean': np.mean(alpha_corrs),
        'alpha_corr_std': np.std(alpha_corrs),
        # W metrics
        'W_corr_per_t_mean': np.mean(W_corrs_per_t),
        'W_corr_per_t_std': np.std(W_corrs_per_t),
        'W_corr_global_mean': np.mean(W_corrs_global),
        'W_corr_global_std': np.std(W_corrs_global),
        'W_rel_error_mean': np.mean(W_rel_errors),
        'W_rel_error_std': np.std(W_rel_errors),
    }


def main():
    print("="*80)
    print("ABLATION STUDY: W Matrix Recovery vs Alpha Recovery")
    print("="*80)

    # Load config from inference.yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'trace_crl', 'configs', 'inference.yaml')
    cfg = load_yaml(config_path)

    # Get paths from config
    ckpt_path = cfg['CHECKPOINTS']['10_DOMAINS']
    data_path = cfg['DATASETS']['10_DOMAINS']
    results_dir = cfg['OUTPUT']['RESULTS']

    # Make paths absolute if relative
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(project_root, ckpt_path)
    if not os.path.isabs(data_path):
        data_path = os.path.join(project_root, data_path)
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(project_root, results_dir)

    training_data_path = os.path.join(data_path, 'data.npz')

    # Get model config
    model_cfg = cfg['MODELS']['10_DOMAINS']
    nclass = model_cfg['NCLASS']
    embedding_dim = model_cfg['EMBED_DIM']

    print(f"\nConfig loaded from: {config_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Data path: {data_path}")
    print(f"  Results dir: {results_dir}")

    print("\nLoading model and parameters...")
    model = load_model(ckpt_path, nclass=nclass, embedding_dim=embedding_dim)
    gen_params = load_generation_params(data_path)

    results = []

    # Test K_active from 2 to 10
    for K_active in range(2, 11):
        for traj_type in ['simple', 'medium', 'complex']:
            print(f"\nRunning: K_active={K_active}, traj={traj_type}...")
            result = run_single_experiment_with_W(
                model, gen_params, training_data_path,
                K_total=10, K_active=K_active, traj_type=traj_type,
                n_runs=10
            )
            results.append(result)
            print(f"  Alpha Corr: {result['alpha_corr_mean']:.4f}±{result['alpha_corr_std']:.4f}")
            print(f"  W Corr (global): {result['W_corr_global_mean']:.4f}±{result['W_corr_global_std']:.4f}")
            print(f"  W Rel Error: {result['W_rel_error_mean']:.4f}±{result['W_rel_error_std']:.4f}")

    # Print summary tables
    print("\n" + "="*100)
    print("SUMMARY: Alpha Correlation vs W Correlation")
    print("="*100)

    # Table 1: Alpha correlation (same as original ablation)
    print("\n--- Alpha Correlation ---")
    print(f"{'K_active':<10} {'Simple':<20} {'Medium':<20} {'Complex':<20}")
    print("-"*70)
    for K_active in range(2, 11):
        row = f"{K_active:<10}"
        for traj_type in ['simple', 'medium', 'complex']:
            r = next(r for r in results if r['K_active'] == K_active and r['traj_type'] == traj_type)
            row += f"{r['alpha_corr_mean']:.3f}±{r['alpha_corr_std']:.3f}      "
        print(row)

    # Table 2: W correlation (global)
    print("\n--- W Correlation (Global) ---")
    print(f"{'K_active':<10} {'Simple':<20} {'Medium':<20} {'Complex':<20}")
    print("-"*70)
    for K_active in range(2, 11):
        row = f"{K_active:<10}"
        for traj_type in ['simple', 'medium', 'complex']:
            r = next(r for r in results if r['K_active'] == K_active and r['traj_type'] == traj_type)
            row += f"{r['W_corr_global_mean']:.3f}±{r['W_corr_global_std']:.3f}      "
        print(row)

    # Table 3: W relative error
    print("\n--- W Relative Error ---")
    print(f"{'K_active':<10} {'Simple':<20} {'Medium':<20} {'Complex':<20}")
    print("-"*70)
    for K_active in range(2, 11):
        row = f"{K_active:<10}"
        for traj_type in ['simple', 'medium', 'complex']:
            r = next(r for r in results if r['K_active'] == K_active and r['traj_type'] == traj_type)
            row += f"{r['W_rel_error_mean']:.3f}±{r['W_rel_error_std']:.3f}      "
        print(row)

    # Table 4: Comparison - show improvement of W over alpha
    print("\n--- Improvement: W Corr - Alpha Corr ---")
    print(f"{'K_active':<10} {'Simple':<20} {'Medium':<20} {'Complex':<20}")
    print("-"*70)
    for K_active in range(2, 11):
        row = f"{K_active:<10}"
        for traj_type in ['simple', 'medium', 'complex']:
            r = next(r for r in results if r['K_active'] == K_active and r['traj_type'] == traj_type)
            improvement = r['W_corr_global_mean'] - r['alpha_corr_mean']
            row += f"{improvement:+.3f}                "
        print(row)

    # Save results
    save_path = os.path.join(results_dir, 'ablation_W_recovery.json')
    os.makedirs(results_dir, exist_ok=True)

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
        json.dump(convert_to_json(results), f, indent=2)
    print(f"\nResults saved to: {save_path}")

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)

    return results


if __name__ == "__main__":
    main()
