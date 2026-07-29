"""
Inference script following the exact method described in the TRACE paper.

Paper Method (Section 5.3 - Stage 3: Mechanism Trajectory Inference):
1. Construct Differential Basis:
   - Extract conditional expectation μ^{(k)} for each domain
   - Compute δμ^{(k)} = μ^{(k)} - μ^{(0)} (shift relative to baseline)
   - Form basis matrix B = [δμ^{(1)}, ..., δμ^{(K-1)}]

2. Linear Least-Squares Solver (Theorem 2):
   - α(t) = Proj_Δ^{K-1}(B^† (z_t - μ^{(0)}))

3. Temporal Smoothing (Theorem 3):
   - Apply window averaging with window size w
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
import matplotlib.pyplot as plt
def load_model(ckpt_path, model_cfg=None):
    """Load trained model from checkpoint."""
    from trace_crl.modules.change import TimeVaryingProcess

    # Default config (5 domains)
    if model_cfg is None:
        model_cfg = {
            'INPUT_DIM': 8, 'Z_DIM': 8, 'LAG': 2,
            'NCLASS': 5, 'HIDDEN_DIM': 128, 'EMBED_DIM': 2
        }

    ckpt = torch.load(ckpt_path)
    model = TimeVaryingProcess(
        input_dim=model_cfg['INPUT_DIM'],
        length=1,
        z_dim=model_cfg['Z_DIM'],
        lag=model_cfg['LAG'],
        nclass=model_cfg['NCLASS'],
        hidden_dim=model_cfg['HIDDEN_DIM'],
        embedding_dim=model_cfg['EMBED_DIM'],
        trans_prior='NP',
        decoder_dist='gaussian',
        correlation='Pearson'
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
    """
    Compute conditional expectation μ^{(k)} for each domain k.

    According to the paper: "we extract the conditional expectation μ^{(k)}
    by finding where the expert's flow output equals zero."

    In practice, this is the mean of the latent z for samples from domain k.
    """
    data = np.load(training_data_path)
    xt = torch.FloatTensor(data['xt']).cuda()  # (N, seq_len, dim)
    ct = data['ct'].flatten()  # (N,)

    domain_means = {}

    for d in domains:
        # Get samples from domain d
        mask = ct == d
        xt_domain = xt[mask]

        # Encode to latent space
        with torch.no_grad():
            z_encoded = encode(model, xt_domain)

        # Use the last time step (z_t) for computing mean
        # z_encoded shape: (N_d, seq_len, z_dim)
        z_t = z_encoded[:, -1, :]  # (N_d, z_dim)

        # Compute mean
        domain_means[d] = z_t.mean(dim=0).cpu().numpy()

        print(f"  Domain {d}: {mask.sum()} samples, mean shape: {domain_means[d].shape}")

    return domain_means


def construct_differential_basis(domain_means, baseline_domain, target_domains):
    """
    Construct the differential basis matrix B.

    B = [δμ^{(k1)}, δμ^{(k2)}, ...] where δμ^{(k)} = μ^{(k)} - μ^{(baseline)}

    For K domains (including baseline), B has shape (z_dim, K-1)
    """
    mu_0 = domain_means[baseline_domain]

    # Build basis from non-baseline domains
    delta_mus = []
    for d in target_domains:
        if d != baseline_domain:
            delta_mu = domain_means[d] - mu_0
            delta_mus.append(delta_mu)

    B = np.stack(delta_mus, axis=1)  # (z_dim, K-1)

    print(f"  Basis matrix B shape: {B.shape}")
    print(f"  B singular values: {np.linalg.svd(B, compute_uv=False)}")

    return B, mu_0


def project_to_simplex(alpha, domains):
    """
    Project α onto the probability simplex Δ^{K-1}.

    For three domains [d0, d1, d2] where d0 is baseline:
    - α has 2 components: [α_d1, α_d2]
    - α_d0 = 1 - α_d1 - α_d2
    - All components must be >= 0 and sum to 1
    """
    K = len(domains)

    # α has K-1 components (excluding baseline)
    # Extend to full K components
    alpha_0 = 1.0 - alpha.sum()
    alpha_full = np.concatenate([[alpha_0], alpha])

    # Project to simplex: clip to [0, 1] and normalize
    alpha_full = np.clip(alpha_full, 0, None)

    # If all zeros, set uniform
    if alpha_full.sum() < 1e-10:
        alpha_full = np.ones(K) / K
    else:
        alpha_full = alpha_full / alpha_full.sum()

    return alpha_full


def solve_alpha_least_squares(z_t, B, mu_0, domains):
    """
    Solve for mixing coefficients using least squares (Theorem 2).

    α(t) = Proj_Δ^{K-1}(B^† (z_t - μ^{(0)}))

    where B^† = (B^T B)^{-1} B^T is the pseudoinverse.
    """
    # Compute pseudoinverse
    B_pinv = np.linalg.pinv(B)  # (K-1, z_dim)

    # Solve: α = B^† (z_t - μ_0)
    residual = z_t - mu_0
    alpha_raw = B_pinv @ residual  # (K-1,)

    # Project to simplex
    alpha_full = project_to_simplex(alpha_raw, domains)

    return alpha_full, alpha_raw


def temporal_smoothing(alpha_sequence, window_size=5):
    """
    Apply temporal smoothing (Theorem 3).

    α_smooth(t) = (1/(2w+1)) * Σ_{s=t-w}^{t+w} α(s)
    """
    T, K = alpha_sequence.shape
    alpha_smooth = np.zeros_like(alpha_sequence)

    for t in range(T):
        start = max(0, t - window_size)
        end = min(T, t + window_size + 1)
        alpha_smooth[t] = alpha_sequence[start:end].mean(axis=0)

    return alpha_smooth


def calibrate_alpha(alpha_pred, alpha_true):
    """
    Calibrate alpha using linear scaling (Remark 1 in paper).
    """
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


def run_inference(model, training_data_path, transition_data_path, window_size=5):
    """
    Run full inference pipeline following the paper's method.
    """
    print("\n" + "=" * 60)
    print("TRACE Paper Method Inference")
    print("=" * 60)

    # Load transition data
    traj = np.load(transition_data_path)
    xt = torch.FloatTensor(traj['xt']).cuda()
    alpha_true = traj['alpha']
    domains = traj['domains'].tolist()

    B, T, seq_len, z_dim = xt.shape
    print(f"\nTransition data: xt={xt.shape}, alpha={alpha_true.shape}")
    print(f"Domains: {domains}")

    # ================================================================
    # Step 1: Construct Differential Basis
    # ================================================================
    print("\n[Step 1] Constructing Differential Basis...")
    domain_means = compute_domain_means(model, training_data_path, domains)

    baseline_domain = domains[0]  # Use first domain as baseline
    B_matrix, mu_0 = construct_differential_basis(domain_means, baseline_domain, domains)

    # ================================================================
    # Step 2: Encode test observations
    # ================================================================
    print("\n[Step 2] Encoding test observations...")
    xt_flat = xt.reshape(B * T, seq_len, z_dim)
    z_encoded_flat = encode(model, xt_flat)
    z_encoded = z_encoded_flat.reshape(B, T, seq_len, -1)
    print(f"  Encoded shape: {z_encoded.shape}")

    # ================================================================
    # Step 3: Solve for α(t) using least squares
    # ================================================================
    print("\n[Step 3] Solving for mixing coefficients (Theorem 2)...")

    # Aggregate across batch dimension
    # Use the last time step z_t from each sequence
    z_t_all = z_encoded[:, :, -1, :].cpu().numpy()  # (B, T, z_dim)
    z_t_mean = z_t_all.mean(axis=0)  # (T, z_dim)

    alpha_pred_raw = []
    alpha_pred_full = []

    for t in range(T):
        z_t = z_t_mean[t]
        alpha_full, alpha_raw = solve_alpha_least_squares(z_t, B_matrix, mu_0, domains)
        alpha_pred_raw.append(alpha_raw)
        alpha_pred_full.append(alpha_full)

    alpha_pred_raw = np.array(alpha_pred_raw)    # (T, K-1)
    alpha_pred_full = np.array(alpha_pred_full)  # (T, K)

    print(f"  Raw α shape: {alpha_pred_raw.shape}")
    print(f"  Full α shape: {alpha_pred_full.shape}")

    # ================================================================
    # Step 4: Temporal smoothing (Theorem 3)
    # ================================================================
    print(f"\n[Step 4] Applying temporal smoothing (window={window_size})...")
    alpha_smooth = temporal_smoothing(alpha_pred_full, window_size)

    # ================================================================
    # Step 5: Evaluate
    # ================================================================
    print("\n[Step 5] Evaluation...")

    # Extract true alpha for target domains
    alpha_true_subset = alpha_true[:, domains]  # (T, K)

    # Calibrate
    alpha_calibrated = calibrate_alpha(alpha_pred_full, alpha_true_subset)
    alpha_smooth_calibrated = calibrate_alpha(alpha_smooth, alpha_true_subset)

    # Compute metrics
    print(f"\n{'Method':<20} {'Domain':<10} {'Corr':>8} {'MSE':>10} {'MSE_cal':>10}")
    print("-" * 60)

    results = {}
    for method_name, alpha_pred in [
        ('Raw (no smooth)', alpha_pred_full),
        ('Smoothed (w=5)', alpha_smooth),
    ]:
        alpha_cal = calibrate_alpha(alpha_pred, alpha_true_subset)
        for i, d in enumerate(domains):
            corr = np.corrcoef(alpha_pred[:, i], alpha_true_subset[:, i])[0, 1]
            mse = np.mean((alpha_pred[:, i] - alpha_true_subset[:, i]) ** 2)
            mse_cal = np.mean((alpha_cal[:, i] - alpha_true_subset[:, i]) ** 2)
            print(f"{method_name:<20} {f'Domain {d}':<10} {corr:>8.4f} {mse:>10.4f} {mse_cal:>10.4f}")

        # Overall metrics
        corr_avg = np.mean([np.corrcoef(alpha_pred[:, i], alpha_true_subset[:, i])[0, 1]
                           for i in range(len(domains))])
        mse_total = np.mean((alpha_cal - alpha_true_subset) ** 2)
        results[method_name] = {'corr_avg': corr_avg, 'mse_cal': mse_total}
        print(f"{method_name:<20} {'Average':<10} {corr_avg:>8.4f} {'':<10} {mse_total:>10.4f}")
        print()

    return {
        'domains': domains,
        'alpha_true': alpha_true_subset,
        'alpha_pred_raw': alpha_pred_full,
        'alpha_pred_smooth': alpha_smooth,
        'alpha_calibrated': alpha_calibrated,
        'alpha_smooth_calibrated': alpha_smooth_calibrated,
        'results': results,
        'B_matrix': B_matrix,
        'mu_0': mu_0,
        'domain_means': domain_means,
    }


def visualize_results(results, save_path_prefix):
    """Create visualization plots."""
    domains = results['domains']
    alpha_true = results['alpha_true']
    alpha_pred = results['alpha_pred_raw']
    alpha_smooth = results['alpha_pred_smooth']
    alpha_cal = results['alpha_calibrated']
    alpha_smooth_cal = results['alpha_smooth_calibrated']

    T = alpha_true.shape[0]
    t_axis = np.arange(T)
    colors = ['blue', 'green', 'red', 'purple', 'orange'][:len(domains)]

    # ================================================================
    # Figure 1: Raw vs Smoothed vs Calibrated
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 0: Raw predictions
    ax = axes[0, 0]
    for i, d in enumerate(domains):
        ax.plot(t_axis, alpha_true[:, i], color=colors[i], linestyle='-', lw=2, label=f'True α_{d}')
        ax.plot(t_axis, alpha_pred[:, i], color=colors[i], linestyle='--', lw=1.5, alpha=0.7, label=f'Raw α_{d}')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Alpha')
    ax.set_title('Raw Predictions (Least Squares)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Plot 1: Smoothed predictions
    ax = axes[0, 1]
    for i, d in enumerate(domains):
        ax.plot(t_axis, alpha_true[:, i], color=colors[i], linestyle='-', lw=2, label=f'True α_{d}')
        ax.plot(t_axis, alpha_smooth[:, i], color=colors[i], linestyle='--', lw=1.5, alpha=0.7, label=f'Smooth α_{d}')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Alpha')
    ax.set_title('Smoothed Predictions (w=5)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Plot 2: Calibrated (raw)
    ax = axes[1, 0]
    for i, d in enumerate(domains):
        ax.plot(t_axis, alpha_true[:, i], color=colors[i], linestyle='-', lw=2, label=f'True α_{d}')
        ax.plot(t_axis, alpha_cal[:, i], color=colors[i], linestyle='--', lw=1.5, alpha=0.7, label=f'Cal α_{d}')
    mse = np.mean((alpha_cal - alpha_true) ** 2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Alpha')
    ax.set_title(f'Calibrated Raw (MSE={mse:.4f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # Plot 3: Calibrated (smoothed)
    ax = axes[1, 1]
    for i, d in enumerate(domains):
        ax.plot(t_axis, alpha_true[:, i], color=colors[i], linestyle='-', lw=2, label=f'True α_{d}')
        ax.plot(t_axis, alpha_smooth_cal[:, i], color=colors[i], linestyle='--', lw=1.5, alpha=0.7, label=f'Cal α_{d}')
    mse = np.mean((alpha_smooth_cal - alpha_true) ** 2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Alpha')
    ax.set_title(f'Calibrated Smoothed (MSE={mse:.4f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    save_path = f'{save_path_prefix}_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()

    # ================================================================
    # Figure 2: Scatter plots
    # ================================================================
    fig, axes = plt.subplots(1, len(domains), figsize=(5*len(domains), 4))
    if len(domains) == 1:
        axes = [axes]

    for i, d in enumerate(domains):
        ax = axes[i]
        ax.scatter(alpha_true[:, i], alpha_smooth[:, i], alpha=0.6, c=colors[i], s=30)
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

        corr = np.corrcoef(alpha_smooth[:, i], alpha_true[:, i])[0, 1]
        ax.set_xlabel(f'True α_{d}')
        ax.set_ylabel(f'Predicted α_{d}')
        ax.set_title(f'Domain {d} (Corr={corr:.4f})')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    save_path = f'{save_path_prefix}_scatter.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Scatter plot saved to: {save_path}")
    plt.close()


def find_trajectory_file(data_path, traj_type='simple', k_active=3):
    """Find trajectory file matching the given criteria."""
    transition_dir = os.path.join(data_path, 'transition')
    if not os.path.exists(transition_dir):
        raise FileNotFoundError(f"Transition directory not found: {transition_dir}")

    # Look for matching files
    pattern = f"traj_{k_active}domain_*_{traj_type}.npz"
    import glob
    matches = glob.glob(os.path.join(transition_dir, pattern))

    if matches:
        return matches[0]

    # Fallback: look for any matching file with k_active domains
    pattern = f"traj_{k_active}domain_*.npz"
    matches = glob.glob(os.path.join(transition_dir, pattern))

    if matches:
        return matches[0]

    # List available files
    all_files = glob.glob(os.path.join(transition_dir, "*.npz"))
    if all_files:
        print(f"Available trajectory files: {[os.path.basename(f) for f in all_files]}")
    raise FileNotFoundError(f"No trajectory file found in {transition_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run TRACE inference')
    parser.add_argument('-d', '--domains', type=int, default=5,
                        choices=[5, 10], help='Number of domains (5 or 10)')
    parser.add_argument('-k', '--k_active', type=int, default=3,
                        help='Number of active domains in trajectory')
    parser.add_argument('-t', '--traj_type', type=str, default='simple',
                        help='Trajectory type (simple, medium, complex)')
    parser.add_argument('--traj_path', type=str, default=None,
                        help='Explicit trajectory file path')
    parser.add_argument('-w', '--window', type=int, default=5,
                        help='Smoothing window size')

    args = parser.parse_args()

    # Load config
    cfg = load_inference_config()

    # Paths from config based on domain count
    domain_key = f'{args.domains}_DOMAINS'
    ckpt_path = cfg['CHECKPOINTS'][domain_key]
    data_path = cfg['DATASETS'][domain_key]
    training_data_path = os.path.join(data_path, 'data.npz')

    # Find or use specified trajectory path
    if args.traj_path:
        transition_data_path = args.traj_path
    else:
        transition_data_path = find_trajectory_file(data_path, args.traj_type, args.k_active)

    print("=" * 60)
    print("TRACE Inference")
    print("=" * 60)
    print(f"Domains: {args.domains}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Training data: {training_data_path}")
    print(f"Trajectory: {transition_data_path}")
    print(f"Window size: {args.window}")

    model_cfg = cfg['MODELS'][domain_key]
    model = load_model(ckpt_path, model_cfg)
    print("Model loaded successfully")

    # Run inference with paper's method
    results = run_inference(model, training_data_path, transition_data_path, window_size=args.window)

    # Visualize
    output_prefix = os.path.join(cfg['OUTPUT']['RESULTS'], 'paper_method')
    os.makedirs(cfg['OUTPUT']['RESULTS'], exist_ok=True)
    visualize_results(results, output_prefix)

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
