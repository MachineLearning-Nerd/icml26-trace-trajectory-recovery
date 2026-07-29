import os
import shutil
import numpy as np
import torch
from scipy.stats import ortho_group

def leaky_ReLU(D, negSlope):
    """Vectorized LeakyReLU"""
    return np.where(D > 0, D, D * negSlope)

def generateUniformMat(Ncomp, condThresh):
    """Generate well-conditioned matrix"""
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condThresh:
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
    return A

def verify_matrix_independence(matrices, names):
    """Verify that each matrix in the list is independent"""
    print("\n=== Matrix Independence Check ===")
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            if matrices[i] is matrices[j]:
                raise ValueError(f"Error: {names[i]} and {names[j]} are the same object!")
            if np.shares_memory(matrices[i], matrices[j]):
                raise ValueError(f"Error: {names[i]} and {names[j]} share memory!")
    print("OK: All matrices are independent")

def verify_delta_structure(W_base, delta_Ws, W_domains, edge_list, delta_value):
    """Verify additive structure"""
    print("\n=== Additive Structure Verification ===")
    for k, (delta_W, W_domain, (i, j)) in enumerate(zip(delta_Ws, W_domains, edge_list)):
        nonzero_count = np.count_nonzero(delta_W)
        if nonzero_count != 1:
            raise ValueError(f"Error: delta_W_{k} has {nonzero_count} non-zero elements")

        if delta_W[i, j] != delta_value:
            raise ValueError(f"Error: delta_W_{k}[{i},{j}] = {delta_W[i, j]}, should be {delta_value}")

        expected = W_base + delta_W
        if not np.allclose(W_domain, expected):
            diff = np.abs(W_domain - expected).max()
            raise ValueError(f"Error: W_domain_{k} != W_base + delta_W_{k}, max diff = {diff}")

        print(f"OK: Domain {k}: edge ({i},{j}), delta = {delta_value}")
    print("OK: All domains have correct additive structure")

def verify_edge_uniqueness(edge_list):
    """Verify that edges across domains do not overlap"""
    print("\n=== Edge Uniqueness Check ===")
    edge_set = set()
    for k, (i, j) in enumerate(edge_list):
        edge_tuple = (i, j)
        if edge_tuple in edge_set:
            raise ValueError(f"Error: edge ({i},{j}) appears in multiple domains!")
        edge_set.add(edge_tuple)
    print(f"OK: All {len(edge_list)} edges are unique")

def verify_saved_files(path, NClass, Nlayer, check_legacy=True):
    """Verify that all files have been saved correctly"""
    print("\n=== File Save Verification ===")
    params_dir = os.path.join(path, "params")

    # New format files
    required_files = [
        "params.npy",
        "W_base_lag1.npy",
        "W_base_lag2.npy",
        "edge_list.npy",
    ]

    for k in range(NClass):
        required_files.append(f"delta_W_{k}.npy")
        required_files.append(f"W_domain_{k}.npy")

    for l in range(Nlayer - 1):
        required_files.append(f"mixing_{l}.npy")

    for fname in required_files:
        fpath = os.path.join(params_dir, fname)
        if not os.path.exists(fpath):
            raise ValueError(f"Error: file {fname} does not exist!")
        try:
            if fname == "params.npy":
                np.load(fpath, allow_pickle=True)
            else:
                np.load(fpath)
        except Exception as e:
            raise ValueError(f"Error: file {fname} cannot be loaded: {e}")

    # Verify data.npz (new format)
    data_path = os.path.join(path, "data.npz")
    if not os.path.exists(data_path):
        raise ValueError("Error: data.npz does not exist!")

    data = np.load(data_path)
    for key in ['yt', 'xt', 'ct']:
        if key not in data:
            raise ValueError(f"Error: data.npz is missing key '{key}'")

    new_format_count = len(required_files) + 1

    # Legacy format files
    legacy_count = 0
    if check_legacy:
        source_dir = os.path.join(path, "source")
        if os.path.exists(source_dir):
            legacy_files = ["data.npz"]
            for k in range(NClass):
                legacy_files.append(f"W1_{k}.npy")
                legacy_files.append(f"W2_{k}.npy")

            for fname in legacy_files:
                fpath = os.path.join(source_dir, fname)
                if not os.path.exists(fpath):
                    raise ValueError(f"Error: legacy file source/{fname} does not exist!")
            legacy_count = len(legacy_files)
            print(f"OK: Legacy format (source/): {legacy_count} files verified")

    print(f"OK: New format: {new_format_count} files verified")

def verify_data_consistency(path, NClass, batch_size, length, lags, latent_size):
    """Verify generated data dimensions and content consistency"""
    print("\n=== Data Consistency Verification ===")

    data = np.load(os.path.join(path, "data.npz"))
    yt, xt, ct = data['yt'], data['xt'], data['ct']

    expected_samples = NClass * batch_size
    expected_time = lags + length

    assert yt.shape == (expected_samples, expected_time, latent_size), \
        f"yt shape error: {yt.shape} vs expected {(expected_samples, expected_time, latent_size)}"
    assert xt.shape == (expected_samples, expected_time, latent_size), \
        f"xt shape error: {xt.shape} vs expected {(expected_samples, expected_time, latent_size)}"
    assert ct.shape == (expected_samples, 1), \
        f"ct shape error: {ct.shape} vs expected {(expected_samples, 1)}"

    for j in range(NClass):
        start_idx = j * batch_size
        end_idx = (j + 1) * batch_size
        domain_labels = ct[start_idx:end_idx]
        if not np.all(domain_labels == j):
            raise ValueError(f"Error: domain {j} labels are incorrect")

    if np.any(np.isnan(yt)) or np.any(np.isinf(yt)):
        raise ValueError("Error: yt contains NaN or Inf")
    if np.any(np.isnan(xt)) or np.any(np.isinf(xt)):
        raise ValueError("Error: xt contains NaN or Inf")

    print(f"OK: yt shape: {yt.shape}")
    print(f"OK: xt shape: {xt.shape}")
    print(f"OK: ct shape: {ct.shape}")
    print("OK: No NaN/Inf")
    print("OK: Domain labels correct")

def verify_legacy_format(path, NClass, batch_size, length, lags, latent_size):
    """Verify legacy format data correctness"""
    print("\n=== Legacy Format Data Verification ===")

    source_dir = os.path.join(path, "source")
    data = np.load(os.path.join(source_dir, "data.npz"))
    yt, xt, ct = data['yt'], data['xt'], data['ct']

    expected_samples = NClass * batch_size
    expected_time = lags + length

    assert yt.shape == (expected_samples, expected_time, latent_size), \
        f"source/yt shape error: {yt.shape}"
    assert xt.shape == (expected_samples, expected_time, latent_size), \
        f"source/xt shape error: {xt.shape}"

    # Verify W matrices match new format
    params_dir = os.path.join(path, "params")
    for k in range(NClass):
        W1_legacy = np.load(os.path.join(source_dir, f"W1_{k}.npy"))
        W1_new = np.load(os.path.join(params_dir, f"W_domain_{k}.npy"))
        if not np.allclose(W1_legacy, W1_new):
            raise ValueError(f"Error: W1_{k} mismatch between new and legacy format")

        W2_legacy = np.load(os.path.join(source_dir, f"W2_{k}.npy"))
        W2_new = np.load(os.path.join(params_dir, "W_base_lag2.npy"))
        if not np.allclose(W2_legacy, W2_new):
            raise ValueError(f"Error: W2_{k} mismatch between new and legacy format")

    print("OK: source/ data dimensions correct")
    print("OK: W matrices match between new and legacy format")

def verify_transition_dynamics(path, gen_params):
    """Verify transition dynamics"""
    print("\n=== Transition Dynamics Verification ===")

    data = np.load(os.path.join(path, "data.npz"))
    yt = data['yt']

    params = gen_params['params']
    W_bases = gen_params['W_bases']
    delta_Ws = gen_params['delta_Ws']
    negSlope = params['negSlope']
    noise_scale = params['noise_scale']
    batch_size = params['batch_size']
    lags = params['lags']
    NClass = params['NClass']

    for j in range(NClass):
        W_lag1 = W_bases[0] + delta_Ws[j]
        W_lag2 = W_bases[1]

        idx = j * batch_size + 100
        y_sample = yt[idx]

        t = lags
        y_t_actual = y_sample[t]

        y_t_expected_no_noise = np.zeros_like(y_t_actual)
        y_t_expected_no_noise += leaky_ReLU(np.dot(y_sample[t-1], W_lag1), negSlope)
        y_t_expected_no_noise += leaky_ReLU(np.dot(y_sample[t-2], W_lag2), negSlope)

        residual = y_t_actual - leaky_ReLU(y_t_expected_no_noise, negSlope)
        residual_norm = np.linalg.norm(residual)

        if residual_norm > 10 * noise_scale * np.sqrt(params['latent_size']):
            print(f"Warning: Domain {j} residual norm = {residual_norm:.4f}")
        else:
            print(f"OK: Domain {j}: residual norm = {residual_norm:.4f}")


def pnl_additive_structure(NClass=5, seed=42):
    """
    Generate multi-domain time series data with additive structure

    Outputs both:
    1. New format (with complete parameters for mixed trajectory generation)
    2. Legacy format (compatible with original training code)
    """
    print("=" * 60)
    print("Starting data generation")
    print("=" * 60)

    # ============ Set random seed ============
    np.random.seed(seed)
    print(f"\nRandom seed: {seed}")

    # ============ Hyperparameters ============
    lags = 2
    Nlayer = 3
    length = 1
    negSlope = 0.2
    latent_size = 8
    noise_scale = 0.1
    batch_size = 40000
    delta_value = 0.5

    print(f"\nHyperparameters:")
    print(f"  NClass = {NClass}")
    print(f"  lags = {lags}")
    print(f"  length = {length}")
    print(f"  latent_size = {latent_size}")
    print(f"  batch_size = {batch_size}")
    print(f"  delta_value = {delta_value}")

    # ============ Path setup ============
    root_dir = 'datasets'
    path = os.path.join(root_dir, f"{NClass}_domains")

    # Clean old files
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"\nCleaned old directory: {path}")

    # Create directory structure
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "params"), exist_ok=True)
    os.makedirs(os.path.join(path, "source"), exist_ok=True)  # Legacy format
    os.makedirs(os.path.join(path, "target"), exist_ok=True)  # Legacy format (if needed)
    print(f"Data directory: {path}")

    # ============ Generate base transition matrices ============
    print("\nGenerating base transition matrices...")
    condList = []
    for i in range(10000):
        A = np.random.uniform(1, 2, (latent_size, latent_size))
        for j in range(latent_size):
            A[:, j] /= np.sqrt((A[:, j] ** 2).sum())
        condList.append(np.linalg.cond(A))
    condThresh = np.percentile(condList, 25)

    W_base_lag1_original = generateUniformMat(latent_size, condThresh)
    W_base_lag2_original = generateUniformMat(latent_size, condThresh)

    # Save (new format)
    np.save(os.path.join(path, "params", "W_base_lag1"), W_base_lag1_original.copy())
    np.save(os.path.join(path, "params", "W_base_lag2"), W_base_lag2_original.copy())
    print("OK: W_base saved")

    # ============ Generate delta_W ============
    print("\nGenerating delta_W matrices...")
    assert NClass <= latent_size * latent_size, f"NClass ({NClass}) is too large"

    all_edges = [(i, j) for i in range(latent_size) for j in range(latent_size)]
    np.random.shuffle(all_edges)
    edge_list = all_edges[:NClass]

    delta_Ws = []
    for k in range(NClass):
        delta_W = np.zeros((latent_size, latent_size))
        i, j = edge_list[k]
        delta_W[i, j] = delta_value
        delta_Ws.append(delta_W.copy())
        np.save(os.path.join(path, "params", f"delta_W_{k}"), delta_W)

    np.save(os.path.join(path, "params", "edge_list"), np.array(edge_list))
    print(f"OK: {NClass} delta_W matrices saved")
    print(f"  Edge list: {edge_list}")

    verify_edge_uniqueness(edge_list)

    # ============ Generate mixing function ============
    print("\nGenerating mixing functions...")
    mixingList = []
    for l in range(Nlayer - 1):
        A = ortho_group.rvs(latent_size)
        mixingList.append(A.copy())
        np.save(os.path.join(path, "params", f"mixing_{l}"), A)
    print(f"OK: {Nlayer - 1} mixing matrices saved")

    # ============ Save hyperparameters ============
    params = {
        'lags': lags,
        'Nlayer': Nlayer,
        'length': length,
        'negSlope': negSlope,
        'latent_size': latent_size,
        'noise_scale': noise_scale,
        'batch_size': batch_size,
        'delta_value': delta_value,
        'NClass': NClass,
        'seed': seed
    }
    np.save(os.path.join(path, "params", "params"), params)
    print("OK: Hyperparameters saved")

    # ============ Mixing function ============
    def mixing_function(y, mixingList, negSlope):
        mixedDat = np.copy(y)
        for A in mixingList:
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, A)
        return mixedDat

    # ============ Generate data ============
    print("\nStarting data generation...")
    yt_all = []
    xt_all = []
    ct_all = []
    W_domains = []

    for j in range(NClass):
        print(f"  Domain {j}/{NClass}...", end=" ")

        # Transition matrix for this domain
        W_lag1 = W_base_lag1_original.copy() + delta_Ws[j]
        W_lag2 = W_base_lag2_original.copy()

        # Save (new format)
        np.save(os.path.join(path, "params", f"W_domain_{j}"), W_lag1.copy())
        W_domains.append(W_lag1.copy())

        # Save (legacy format) - W1 is lag1, W2 is lag2
        np.save(os.path.join(path, "source", f"W1_{j}"), W_lag1.copy())
        np.save(os.path.join(path, "source", f"W2_{j}"), W_lag2.copy())

        transitions = [W_lag1, W_lag2]

        # Initialize latent
        y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
        y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0, keepdims=True)

        yt = []
        xt = []

        # Initial lags
        for i in range(lags):
            yt.append(y_l[:, i, :].copy())

        x_l = mixing_function(y_l, mixingList, negSlope)
        for i in range(lags):
            xt.append(x_l[:, i, :].copy())

        # Generate subsequent time steps
        for t in range(length):
            y_t = np.random.normal(0, noise_scale, (batch_size, latent_size))
            y_t += leaky_ReLU(np.dot(y_l[:, 1, :], transitions[0]), negSlope)  # t-1
            y_t += leaky_ReLU(np.dot(y_l[:, 0, :], transitions[1]), negSlope)  # t-2
            y_t = leaky_ReLU(y_t, negSlope)
            yt.append(y_t.copy())

            x_t = mixing_function(y_t.reshape(batch_size, 1, latent_size), mixingList, negSlope)
            x_t = x_t.squeeze()
            xt.append(x_t.copy())

            y_l = np.concatenate((y_l[:, 1:, :], y_t[:, np.newaxis, :]), axis=1)

        yt = np.array(yt).transpose(1, 0, 2)
        xt = np.array(xt).transpose(1, 0, 2)
        ct = j * np.ones((batch_size, 1))

        yt_all.append(yt)
        xt_all.append(xt)
        ct_all.append(ct)

        print(f"OK: yt shape: {yt.shape}")

    # ============ Merge data ============
    yt_all = np.vstack(yt_all)
    xt_all = np.vstack(xt_all)
    ct_all = np.vstack(ct_all)

    # ============ Save data (new format) ============
    np.savez(os.path.join(path, "data"),
             yt=yt_all,
             xt=xt_all,
             ct=ct_all)
    print(f"\nOK: New format data saved: data.npz")

    # ============ Save data (legacy format) ============
    np.savez(os.path.join(path, "source", "data"),
             yt=yt_all,
             xt=xt_all,
             ct=ct_all)
    print(f"OK: Legacy format data saved: source/data.npz")

    # If target is needed (can be slightly extrapolated version, here just copy source)
    np.savez(os.path.join(path, "target", "data"),
             yt=yt_all,
             xt=xt_all,
             ct=ct_all)
    for j in range(NClass):
        W_lag1 = W_base_lag1_original.copy() + delta_Ws[j]
        W_lag2 = W_base_lag2_original.copy()
        np.save(os.path.join(path, "target", f"W1_{j}"), W_lag1.copy())
        np.save(os.path.join(path, "target", f"W2_{j}"), W_lag2.copy())
    print(f"OK: Legacy format data saved: target/data.npz")

    # ============ Verification ============
    print("\n" + "=" * 60)
    print("Starting verification")
    print("=" * 60)

    all_matrices = [W_base_lag1_original] + delta_Ws + W_domains
    all_names = ["W_base_lag1"] + [f"delta_W_{k}" for k in range(NClass)] + [f"W_domain_{k}" for k in range(NClass)]
    verify_matrix_independence(all_matrices, all_names)

    verify_delta_structure(W_base_lag1_original, delta_Ws, W_domains, edge_list, delta_value)

    verify_saved_files(path, NClass, Nlayer, check_legacy=True)

    verify_data_consistency(path, NClass, batch_size, length, lags, latent_size)

    verify_legacy_format(path, NClass, batch_size, length, lags, latent_size)

    gen_params = load_generation_params(path)
    verify_transition_dynamics(path, gen_params)

    # ============ Final report ============
    print("\n" + "=" * 60)
    print("Data generation complete")
    print("=" * 60)
    print(f"Path: {path}")
    print(f"Total samples: {yt_all.shape[0]}")
    print(f"Time steps: {yt_all.shape[1]}")
    print(f"Dimensions: {yt_all.shape[2]}")
    print(f"\nDirectory structure:")
    print(f"  {path}/")
    print(f"  ├── data.npz                 # New format data")
    print(f"  ├── params/                  # New format params (for mixed trajectory generation)")
    print(f"  │   ├── W_base_lag1.npy")
    print(f"  │   ├── W_base_lag2.npy")
    print(f"  │   ├── delta_W_*.npy")
    print(f"  │   ├── W_domain_*.npy")
    print(f"  │   ├── mixing_*.npy")
    print(f"  │   ├── edge_list.npy")
    print(f"  │   └── params.npy")
    print(f"  ├── source/                  # Legacy format (compatible with training code)")
    print(f"  │   ├── data.npz")
    print(f"  │   ├── W1_*.npy")
    print(f"  │   └── W2_*.npy")
    print(f"  └── target/                  # Legacy format")
    print(f"      ├── data.npz")
    print(f"      ├── W1_*.npy")
    print(f"      └── W2_*.npy")

    return path


def load_generation_params(path):
    """Load saved generation parameters"""
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

    edge_list = np.load(os.path.join(params_dir, "edge_list.npy"))

    W_domains = []
    for k in range(params['NClass']):
        W_domain = np.load(os.path.join(params_dir, f"W_domain_{k}.npy"))
        W_domains.append(W_domain)

    return {
        'params': params,
        'W_bases': W_bases,
        'delta_Ws': delta_Ws,
        'mixingList': mixingList,
        'edge_list': edge_list,
        'W_domains': W_domains
    }


def verify_loaded_params(path):
    """Verify after loading"""
    print("\n" + "=" * 60)
    print("Post-load verification")
    print("=" * 60)

    gen_params = load_generation_params(path)
    params = gen_params['params']
    W_bases = gen_params['W_bases']
    delta_Ws = gen_params['delta_Ws']
    W_domains = gen_params['W_domains']

    print("\nVerifying loaded data...")
    for k in range(params['NClass']):
        expected = W_bases[0] + delta_Ws[k]
        if not np.allclose(W_domains[k], expected):
            raise ValueError(f"Post-load verification failed: W_domain_{k}")

    print("OK: Post-load verification passed")
    return gen_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate multi-domain time series data')
    parser.add_argument('-d', '--domains', type=int, default=5,
                        help='Number of domains to generate (default: 5)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    path = pnl_additive_structure(NClass=args.domains, seed=args.seed)
    gen_params = verify_loaded_params(path)

    print("\nFinal edge_list:")
    for k, (i, j) in enumerate(gen_params['edge_list']):
        print(f"  Domain {k}: edge ({i}, {j})")
