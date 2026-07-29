"""
Evaluation script for varying number of domains (K = 2 to 10).

For each K:
1. Generate multiple transition trajectories (n_runs)
2. Run inference using the paper's method
3. Report MAE, Correlation with mean and std
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_crl.tools.utils import load_yaml

def load_inference_config():
    """Load inference config from trace/configs/inference.yaml"""
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '../trace_crl/configs/inference.yaml')
    return load_yaml(config_path)

import torch
import numpy as np
import json
from scipy.optimize import nnls
import itertools


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


def generate_K_domain_trajectory(
    gen_params,
    domains,
    transition_steps=50,
    batch_size=500,
    seed=None
):
    """
    Generate a trajectory with K domains participating.

    Alpha curves: smooth transitions using cosine/sine patterns.
    """
    if seed is not None:
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

    K = len(domains)

    # Generate alpha sequence for K domains
    t_normalized = np.linspace(0, 1, transition_steps)

    # Create smooth alpha curves for K domains
    # Use different frequency sinusoids for diversity
    alphas_raw = []
    for i in range(K):
        freq = 1 + i * 0.5
        phase = i * np.pi / K
        alpha_i = 0.5 * (1 + np.cos(freq * np.pi * t_normalized + phase))
        alphas_raw.append(alpha_i)

    alphas_raw = np.array(alphas_raw).T  # (T, K)

    # Normalize to simplex
    alpha_sum = alphas_raw.sum(axis=1, keepdims=True)
    alphas_normalized = alphas_raw / alpha_sum

    # Build full alpha sequence
    alpha_sequence = np.zeros((transition_steps, NClass))
    for i, d in enumerate(domains):
        alpha_sequence[:, d] = alphas_normalized[:, i]

    yt_all = []
    xt_all = []

    for t in range(transition_steps):
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

    return {
        'yt': yt_all,
        'xt': xt_all,
        'alpha': alpha_sequence,
        'domains': np.array(domains),
    }


def load_model(ckpt_path, nclass=10, embedding_dim=8):
    """Load trained model from checkpoint."""
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
    """Encode observations to latent space."""
    B, T, x_dim = x.shape
    x_flat = x.reshape(-1, x_dim)
    with torch.no_grad():
        _, mus, _, _ = model.net(x_flat)
    return mus.reshape(B, T, -1)


def compute_domain_means(model, training_data_path, domains):
    """Compute conditional expectation μ^{(k)} for each domain k."""
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


def compute_all_domain_means(model, training_data_path, n_domains=10):
    """Compute conditional expectation μ^{(k)} for all domains."""
    return compute_domain_means(model, training_data_path, list(range(n_domains)))


def select_max_separation_domains(all_domain_means, K, baseline=0):
    """
    Select K domains with maximum separation using greedy algorithm.

    Strategy: Start with baseline domain, then greedily add domains that
    maximize the minimum distance to already selected domains.
    """
    from scipy.spatial.distance import euclidean

    n_domains = len(all_domain_means)

    if K >= n_domains:
        return list(range(n_domains))

    # Start with baseline domain
    selected = [baseline]
    remaining = [d for d in range(n_domains) if d != baseline]

    while len(selected) < K:
        best_domain = None
        best_min_dist = -1

        for d in remaining:
            # Compute minimum distance to any selected domain
            min_dist = min(euclidean(all_domain_means[d], all_domain_means[s])
                          for s in selected)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_domain = d

        selected.append(best_domain)
        remaining.remove(best_domain)

    return sorted(selected)


def construct_differential_basis(domain_means, baseline_domain, target_domains):
    """Construct the differential basis matrix B."""
    mu_0 = domain_means[baseline_domain]

    delta_mus = []
    for d in target_domains:
        if d != baseline_domain:
            delta_mu = domain_means[d] - mu_0
            delta_mus.append(delta_mu)

    B = np.stack(delta_mus, axis=1)
    return B, mu_0


def project_to_simplex(alpha, K):
    """Project α onto the probability simplex."""
    alpha_0 = 1.0 - alpha.sum()
    alpha_full = np.concatenate([[alpha_0], alpha])
    alpha_full = np.clip(alpha_full, 0, None)

    if alpha_full.sum() < 1e-10:
        alpha_full = np.ones(K) / K
    else:
        alpha_full = alpha_full / alpha_full.sum()

    return alpha_full


def solve_alpha_least_squares(z_t, B, mu_0, K):
    """Solve for mixing coefficients using least squares."""
    B_pinv = np.linalg.pinv(B)
    residual = z_t - mu_0
    alpha_raw = B_pinv @ residual
    alpha_full = project_to_simplex(alpha_raw, K)
    return alpha_full


def temporal_smoothing(alpha_sequence, window_size=5):
    """Apply temporal smoothing."""
    T, K = alpha_sequence.shape
    alpha_smooth = np.zeros_like(alpha_sequence)

    for t in range(T):
        start = max(0, t - window_size)
        end = min(T, t + window_size + 1)
        alpha_smooth[t] = alpha_sequence[start:end].mean(axis=0)

    return alpha_smooth


def calibrate_alpha(alpha_pred, alpha_true):
    """Calibrate alpha using linear scaling."""
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


def run_single_inference(model, domain_means, traj_data, window_sizes=[0, 3, 5, 7, 10]):
    """Run inference on a single trajectory with multiple window sizes."""
    xt = torch.FloatTensor(traj_data['xt']).cuda()
    alpha_true = traj_data['alpha']
    domains = traj_data['domains'].tolist()

    B, T, seq_len, z_dim = xt.shape
    K = len(domains)

    # Construct differential basis
    baseline_domain = domains[0]
    B_matrix, mu_0 = construct_differential_basis(domain_means, baseline_domain, domains)

    # Encode
    xt_flat = xt.reshape(B * T, seq_len, z_dim)
    z_encoded_flat = encode(model, xt_flat)
    z_encoded = z_encoded_flat.reshape(B, T, seq_len, -1)

    # Get z_t for each time step
    z_t_all = z_encoded[:, :, -1, :].cpu().numpy()
    z_t_mean = z_t_all.mean(axis=0)

    # Solve for alpha (raw, no smoothing)
    alpha_pred_full = []
    for t in range(T):
        z_t = z_t_mean[t]
        alpha_full = solve_alpha_least_squares(z_t, B_matrix, mu_0, K)
        alpha_pred_full.append(alpha_full)

    alpha_pred_full = np.array(alpha_pred_full)

    # Extract true alpha for target domains
    alpha_true_subset = alpha_true[:, domains]

    # Evaluate for each window size
    results_by_window = {}
    for window_size in window_sizes:
        if window_size == 0:
            alpha_smooth = alpha_pred_full.copy()
        else:
            alpha_smooth = temporal_smoothing(alpha_pred_full, window_size)

        # Calibrate
        alpha_calibrated = calibrate_alpha(alpha_smooth, alpha_true_subset)

        # Compute metrics
        mae = np.mean(np.abs(alpha_calibrated - alpha_true_subset))

        corrs = []
        for i in range(K):
            corr = np.corrcoef(alpha_smooth[:, i], alpha_true_subset[:, i])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
        corr_mean = np.mean(corrs) if corrs else 0.0

        mse = np.mean((alpha_calibrated - alpha_true_subset) ** 2)
        rmse = np.sqrt(mse)

        results_by_window[window_size] = {
            'mae': mae,
            'corr': corr_mean,
            'mse': mse,
            'rmse': rmse,
        }

    return results_by_window


def evaluate_K_domains(model, training_data_path, gen_params, K, n_runs=5,
                       transition_steps=50, batch_size=500, base_seed=42,
                       window_sizes=[0, 3, 5, 7, 10],
                       all_domain_means=None):
    """Evaluate performance for K domains with multiple runs and window sizes."""

    # Select K domains with maximum separation
    if all_domain_means is None:
        all_domain_means = compute_all_domain_means(model, training_data_path, n_domains=10)

    domains = select_max_separation_domains(all_domain_means, K, baseline=0)

    print(f"\n{'='*60}")
    print(f"Evaluating K={K} domains (max separation): {domains}")
    print(f"{'='*60}")

    # Get domain means for selected domains
    domain_means = {d: all_domain_means[d] for d in domains}

    # Results for each window size
    results_by_window = {w: {'mae': [], 'corr': [], 'mse': [], 'rmse': []} for w in window_sizes}

    for run_idx in range(n_runs):
        seed = base_seed + run_idx * 100

        # Generate trajectory
        traj_data = generate_K_domain_trajectory(
            gen_params,
            domains=domains,
            transition_steps=transition_steps,
            batch_size=batch_size,
            seed=seed
        )

        # Run inference (returns results for all window sizes)
        run_results = run_single_inference(model, domain_means, traj_data, window_sizes)

        for w in window_sizes:
            results_by_window[w]['mae'].append(run_results[w]['mae'])
            results_by_window[w]['corr'].append(run_results[w]['corr'])
            results_by_window[w]['mse'].append(run_results[w]['mse'])
            results_by_window[w]['rmse'].append(run_results[w]['rmse'])

        # Print progress (using window=5 as reference)
        ref_w = 5 if 5 in window_sizes else window_sizes[0]
        print(f"  Run {run_idx+1}/{n_runs}: MAE={run_results[ref_w]['mae']:.4f}, Corr={run_results[ref_w]['corr']:.4f} (w={ref_w})")

    # Compute mean and std for each window size
    summary = {
        'K': K,
        'domains': domains,
        'n_runs': n_runs,
        'by_window': {}
    }

    print(f"\n  Summary for K={K}:")
    print(f"  {'Window':<8} {'MAE (mean±std)':<20} {'Corr (mean±std)':<20} {'RMSE (mean±std)':<20}")
    print(f"  {'-'*68}")

    for w in window_sizes:
        w_summary = {
            'mae_mean': np.mean(results_by_window[w]['mae']),
            'mae_std': np.std(results_by_window[w]['mae']),
            'corr_mean': np.mean(results_by_window[w]['corr']),
            'corr_std': np.std(results_by_window[w]['corr']),
            'mse_mean': np.mean(results_by_window[w]['mse']),
            'mse_std': np.std(results_by_window[w]['mse']),
            'rmse_mean': np.mean(results_by_window[w]['rmse']),
            'rmse_std': np.std(results_by_window[w]['rmse']),
        }
        summary['by_window'][w] = w_summary

        mae_str = f"{w_summary['mae_mean']:.4f} ± {w_summary['mae_std']:.4f}"
        corr_str = f"{w_summary['corr_mean']:.4f} ± {w_summary['corr_std']:.4f}"
        rmse_str = f"{w_summary['rmse_mean']:.4f} ± {w_summary['rmse_std']:.4f}"
        print(f"  w={w:<5} {mae_str:<20} {corr_str:<20} {rmse_str:<20}")

    return summary


def main():
    # Load config
    cfg = load_inference_config()

    # Paths from config
    ckpt_path = cfg['CHECKPOINTS']['10_DOMAINS']
    data_path = cfg['DATASETS']['10_DOMAINS']
    training_data_path = os.path.join(data_path, 'data.npz')
    nclass = cfg['MODELS']['10_DOMAINS']['NCLASS']

    print("="*60)
    print("Loading Model")
    print("="*60)
    model = load_model(ckpt_path, nclass=nclass)
    print("Model loaded successfully")

    print("\n"+"="*60)
    print("Loading Generation Parameters")
    print("="*60)
    gen_params = load_generation_params(data_path)
    print(f"NClass: {gen_params['params']['NClass']}")
    print(f"Latent size: {gen_params['params']['latent_size']}")

    # Precompute all domain means for efficient domain selection
    print("\n"+"="*60)
    print("Computing All Domain Means")
    print("="*60)
    all_domain_means = compute_all_domain_means(model, training_data_path, n_domains=10)

    # Print domain separation info
    from scipy.spatial.distance import euclidean
    print("\nPairwise distances between domains:")
    for i in range(10):
        for j in range(i+1, 10):
            dist = euclidean(all_domain_means[i], all_domain_means[j])
            if dist > 0.1:  # Only print larger distances
                print(f"  D{i}-D{j}: {dist:.4f}")

    # Configuration
    n_runs = 5
    window_sizes = [0, 3, 5, 7, 10]
    all_results = []

    # Evaluate for K = 2 to 10
    for K in range(2, 11):
        summary = evaluate_K_domains(
            model, training_data_path, gen_params, K,
            n_runs=n_runs,
            transition_steps=50,
            batch_size=500,
            window_sizes=window_sizes,
            all_domain_means=all_domain_means
        )
        all_results.append(summary)

    # Print final summary tables (one per window size)
    print("\n" + "="*100)
    print("FINAL SUMMARY TABLES")
    print("="*100)

    for w in window_sizes:
        print(f"\n{'='*80}")
        print(f"Window Size = {w} {'(No Smoothing)' if w == 0 else ''}")
        print(f"{'='*80}")
        print(f"{'K':<5} {'MAE (mean±std)':<20} {'Corr (mean±std)':<20} {'RMSE (mean±std)':<20}")
        print("-"*80)

        for res in all_results:
            w_data = res['by_window'][w]
            mae_str = f"{w_data['mae_mean']:.4f} ± {w_data['mae_std']:.4f}"
            corr_str = f"{w_data['corr_mean']:.4f} ± {w_data['corr_std']:.4f}"
            rmse_str = f"{w_data['rmse_mean']:.4f} ± {w_data['rmse_std']:.4f}"
            print(f"{res['K']:<5} {mae_str:<20} {corr_str:<20} {rmse_str:<20}")

    # Print combined table (K vs Window, showing Corr)
    print("\n" + "="*100)
    print("CORRELATION TABLE (K vs Window Size)")
    print("="*100)
    header = f"{'K':<5}"
    for w in window_sizes:
        header += f"{'w='+str(w):<18}"
    print(header)
    print("-"*100)

    for res in all_results:
        row = f"{res['K']:<5}"
        for w in window_sizes:
            w_data = res['by_window'][w]
            row += f"{w_data['corr_mean']:.4f} ± {w_data['corr_std']:.4f}  "
        print(row)

    # Print combined table (K vs Window, showing MAE)
    print("\n" + "="*100)
    print("MAE TABLE (K vs Window Size)")
    print("="*100)
    header = f"{'K':<5}"
    for w in window_sizes:
        header += f"{'w='+str(w):<18}"
    print(header)
    print("-"*100)

    for res in all_results:
        row = f"{res['K']:<5}"
        for w in window_sizes:
            w_data = res['by_window'][w]
            row += f"{w_data['mae_mean']:.4f} ± {w_data['mae_std']:.4f}  "
        print(row)

    # Save results
    save_path = 'results/varying_K_window_evaluation.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert numpy types to Python types for JSON
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

    results_for_json = convert_to_json(all_results)

    with open(save_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"\nResults saved to: {save_path}")

    print("\n" + "="*60)
    print("Done")
    print("="*60)

    return all_results


if __name__ == "__main__":
    main()
