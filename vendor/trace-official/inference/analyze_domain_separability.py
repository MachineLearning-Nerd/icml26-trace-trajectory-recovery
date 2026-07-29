"""
Analyze domain separability in the latent space.

Check if the model can distinguish between different domains by examining:
1. Domain mean differences (δμ norms)
2. Basis matrix condition number
3. Pairwise cosine similarity between domain means
4. Pairwise Euclidean distances
5. Singular value analysis of basis matrix
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
from scipy.spatial.distance import cosine, euclidean
from itertools import combinations


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


def compute_all_domain_means(model, training_data_path, n_domains=10):
    """Compute conditional expectation μ^{(k)} for all domains."""
    data = np.load(training_data_path)
    xt = torch.FloatTensor(data['xt']).cuda()
    ct = data['ct'].flatten()

    domain_means = {}
    domain_stds = {}
    domain_counts = {}

    for d in range(n_domains):
        mask = ct == d
        xt_domain = xt[mask]

        with torch.no_grad():
            z_encoded = encode(model, xt_domain)

        z_t = z_encoded[:, -1, :]  # Last time step
        domain_means[d] = z_t.mean(dim=0).cpu().numpy()
        domain_stds[d] = z_t.std(dim=0).cpu().numpy()
        domain_counts[d] = mask.sum().item()

    return domain_means, domain_stds, domain_counts


def analyze_domain_separability(domain_means, n_domains=10):
    """Analyze separability metrics."""

    print("=" * 80)
    print("DOMAIN SEPARABILITY ANALYSIS")
    print("=" * 80)

    # 1. Domain mean norms
    print("\n[1] Domain Mean Norms (||μ^{(k)}||)")
    print("-" * 40)
    for d in range(n_domains):
        norm = np.linalg.norm(domain_means[d])
        print(f"  Domain {d}: ||μ|| = {norm:.4f}")

    # 2. Delta mu norms (relative to domain 0)
    print("\n[2] Delta Mean Norms (||δμ^{(k)}|| = ||μ^{(k)} - μ^{(0)}||)")
    print("-" * 40)
    mu_0 = domain_means[0]
    delta_norms = {}
    for d in range(1, n_domains):
        delta_mu = domain_means[d] - mu_0
        delta_norms[d] = np.linalg.norm(delta_mu)
        print(f"  Domain {d}: ||δμ|| = {delta_norms[d]:.4f}")

    # 3. Pairwise Euclidean distances
    print("\n[3] Pairwise Euclidean Distances")
    print("-" * 40)
    distances = np.zeros((n_domains, n_domains))
    for i in range(n_domains):
        for j in range(n_domains):
            distances[i, j] = euclidean(domain_means[i], domain_means[j])

    # Print as matrix
    header = "     " + "".join([f"  D{d:d}   " for d in range(n_domains)])
    print(header)
    for i in range(n_domains):
        row = f"D{i}  "
        for j in range(n_domains):
            row += f"{distances[i, j]:6.3f} "
        print(row)

    print(f"\n  Mean pairwise distance: {distances[np.triu_indices(n_domains, k=1)].mean():.4f}")
    print(f"  Min pairwise distance:  {distances[np.triu_indices(n_domains, k=1)].min():.4f}")
    print(f"  Max pairwise distance:  {distances[np.triu_indices(n_domains, k=1)].max():.4f}")

    # 4. Pairwise Cosine Similarities
    print("\n[4] Pairwise Cosine Similarities")
    print("-" * 40)
    cosine_sims = np.zeros((n_domains, n_domains))
    for i in range(n_domains):
        for j in range(n_domains):
            cosine_sims[i, j] = 1 - cosine(domain_means[i], domain_means[j])

    header = "     " + "".join([f"  D{d:d}   " for d in range(n_domains)])
    print(header)
    for i in range(n_domains):
        row = f"D{i}  "
        for j in range(n_domains):
            row += f"{cosine_sims[i, j]:6.3f} "
        print(row)

    print(f"\n  Mean cosine similarity: {cosine_sims[np.triu_indices(n_domains, k=1)].mean():.4f}")

    # 5. Basis matrix analysis for different K
    print("\n[5] Basis Matrix Analysis (Condition Number & Singular Values)")
    print("-" * 60)
    print(f"{'K':<5} {'Cond. Number':<15} {'Singular Values':<50}")
    print("-" * 60)

    condition_numbers = {}
    singular_values_all = {}

    for K in range(2, n_domains + 1):
        # Select K domains evenly spaced
        if K == n_domains:
            domains = list(range(n_domains))
        else:
            step = n_domains / K
            domains = [int(i * step) for i in range(K)]
            domains = list(set(domains))
            while len(domains) < K:
                for d in range(n_domains):
                    if d not in domains:
                        domains.append(d)
                        break
            domains = sorted(domains[:K])

        # Construct basis matrix
        baseline = domains[0]
        delta_mus = []
        for d in domains[1:]:
            delta_mu = domain_means[d] - domain_means[baseline]
            delta_mus.append(delta_mu)

        B = np.stack(delta_mus, axis=1)  # (z_dim, K-1)

        # Compute SVD
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        cond_num = S.max() / S.min() if S.min() > 1e-10 else float('inf')

        condition_numbers[K] = cond_num
        singular_values_all[K] = S

        sv_str = ", ".join([f"{s:.4f}" for s in S])
        print(f"{K:<5} {cond_num:<15.2f} [{sv_str}]")

    # 6. Cosine similarity of delta_mu vectors
    print("\n[6] Cosine Similarity of Differential Vectors (δμ)")
    print("-" * 60)

    # Compute all delta_mu relative to domain 0
    delta_mus = {}
    for d in range(1, n_domains):
        delta_mus[d] = domain_means[d] - domain_means[0]

    print("     " + "".join([f"  D{d:d}   " for d in range(1, n_domains)]))
    for i in range(1, n_domains):
        row = f"D{i}  "
        for j in range(1, n_domains):
            if i == j:
                row += f"  1.00 "
            else:
                sim = 1 - cosine(delta_mus[i], delta_mus[j])
                row += f"{sim:6.3f} "
        print(row)

    # High similarity pairs
    print("\n  High similarity pairs (>0.9):")
    high_sim_count = 0
    for i in range(1, n_domains):
        for j in range(i+1, n_domains):
            sim = 1 - cosine(delta_mus[i], delta_mus[j])
            if sim > 0.9:
                print(f"    δμ_{i} vs δμ_{j}: {sim:.4f}")
                high_sim_count += 1
    if high_sim_count == 0:
        print("    None")

    return {
        'distances': distances,
        'cosine_sims': cosine_sims,
        'condition_numbers': condition_numbers,
        'singular_values': singular_values_all,
        'delta_norms': delta_norms,
    }


def visualize_analysis(domain_means, analysis_results, save_prefix):
    """Create visualization plots."""
    n_domains = len(domain_means)

    # Figure 1: Domain means heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Domain means as heatmap
    ax = axes[0, 0]
    means_matrix = np.array([domain_means[d] for d in range(n_domains)])
    im = ax.imshow(means_matrix, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Domain')
    ax.set_title('Domain Means μ^{(k)}')
    ax.set_yticks(range(n_domains))
    ax.set_yticklabels([f'D{d}' for d in range(n_domains)])
    plt.colorbar(im, ax=ax)

    # Plot 2: Pairwise distances
    ax = axes[0, 1]
    im = ax.imshow(analysis_results['distances'], cmap='viridis')
    ax.set_xlabel('Domain')
    ax.set_ylabel('Domain')
    ax.set_title('Pairwise Euclidean Distances')
    ax.set_xticks(range(n_domains))
    ax.set_xticklabels([f'D{d}' for d in range(n_domains)])
    ax.set_yticks(range(n_domains))
    ax.set_yticklabels([f'D{d}' for d in range(n_domains)])
    plt.colorbar(im, ax=ax)

    # Plot 3: Condition number vs K
    ax = axes[1, 0]
    Ks = list(analysis_results['condition_numbers'].keys())
    conds = [analysis_results['condition_numbers'][k] for k in Ks]
    ax.plot(Ks, conds, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Domains (K)')
    ax.set_ylabel('Condition Number')
    ax.set_title('Basis Matrix Condition Number vs K')
    ax.set_xticks(Ks)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Singular values for different K
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(Ks)))
    for i, K in enumerate(Ks):
        svs = analysis_results['singular_values'][K]
        ax.plot(range(1, len(svs)+1), svs, 'o-', color=colors[i],
                label=f'K={K}', linewidth=1.5, markersize=6)
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values of Basis Matrix B')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    save_path = f'{save_prefix}_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()

    # Figure 2: Delta mu cosine similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    delta_mus = {}
    for d in range(1, n_domains):
        delta_mus[d] = domain_means[d] - domain_means[0]

    delta_cosine = np.zeros((n_domains-1, n_domains-1))
    for i in range(1, n_domains):
        for j in range(1, n_domains):
            delta_cosine[i-1, j-1] = 1 - cosine(delta_mus[i], delta_mus[j])

    im = ax.imshow(delta_cosine, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xlabel('Domain')
    ax.set_ylabel('Domain')
    ax.set_title('Cosine Similarity of Differential Vectors δμ^{(k)}')
    ax.set_xticks(range(n_domains-1))
    ax.set_xticklabels([f'D{d}' for d in range(1, n_domains)])
    ax.set_yticks(range(n_domains-1))
    ax.set_yticklabels([f'D{d}' for d in range(1, n_domains)])

    # Add text annotations
    for i in range(n_domains-1):
        for j in range(n_domains-1):
            text = ax.text(j, i, f'{delta_cosine[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=8)

    plt.colorbar(im, ax=ax)

    save_path = f'{save_prefix}_delta_cosine.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Delta cosine similarity saved to: {save_path}")
    plt.close()

    # Figure 3: 2D PCA visualization of domain means
    fig, ax = plt.subplots(figsize=(10, 8))

    means_matrix = np.array([domain_means[d] for d in range(n_domains)])

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    means_2d = pca.fit_transform(means_matrix)

    colors = plt.cm.tab10(np.linspace(0, 1, n_domains))
    for d in range(n_domains):
        ax.scatter(means_2d[d, 0], means_2d[d, 1], c=[colors[d]], s=200,
                  label=f'D{d}', edgecolors='black', linewidth=1.5)
        ax.annotate(f'D{d}', (means_2d[d, 0], means_2d[d, 1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=12)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA Visualization of Domain Means')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f'{save_prefix}_pca.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"PCA visualization saved to: {save_path}")
    plt.close()


def main():
    # Load config
    cfg = load_inference_config()

    # Paths from config
    ckpt_path = cfg['CHECKPOINTS']['10_DOMAINS']
    data_path = cfg['DATASETS']['10_DOMAINS']
    training_data_path = os.path.join(data_path, 'data.npz')
    save_prefix = os.path.join(cfg['OUTPUT']['RESULTS'], 'domain_separability')
    nclass = cfg['MODELS']['10_DOMAINS']['NCLASS']

    print("=" * 60)
    print("Loading Model")
    print("=" * 60)
    model = load_model(ckpt_path, nclass=nclass)
    print("Model loaded successfully")

    print("\n" + "=" * 60)
    print("Computing Domain Means")
    print("=" * 60)
    domain_means, domain_stds, domain_counts = compute_all_domain_means(
        model, training_data_path, n_domains=10
    )

    for d in range(10):
        print(f"  Domain {d}: {domain_counts[d]} samples")

    print("\n")
    analysis_results = analyze_domain_separability(domain_means, n_domains=10)

    # Visualize
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    visualize_analysis(domain_means, analysis_results, save_prefix)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Why performance degrades with larger K")
    print("=" * 80)

    cond_nums = analysis_results['condition_numbers']
    print(f"\n1. Condition Number Growth:")
    print(f"   K=2: {cond_nums[2]:.2f}")
    print(f"   K=5: {cond_nums[5]:.2f}")
    print(f"   K=10: {cond_nums[10]:.2f}")
    print(f"   -> Higher condition number = less stable least squares solution")

    # Find most similar delta_mu pairs
    delta_mus = {}
    for d in range(1, 10):
        delta_mus[d] = domain_means[d] - domain_means[0]

    most_similar = []
    for i in range(1, 10):
        for j in range(i+1, 10):
            sim = 1 - cosine(delta_mus[i], delta_mus[j])
            most_similar.append((i, j, sim))
    most_similar.sort(key=lambda x: -x[2])

    print(f"\n2. Most Similar Differential Vectors (top 5):")
    for i, j, sim in most_similar[:5]:
        print(f"   δμ_{i} vs δμ_{j}: cosine similarity = {sim:.4f}")
    print(f"   -> High similarity = near-linear dependence = ill-conditioned basis")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
