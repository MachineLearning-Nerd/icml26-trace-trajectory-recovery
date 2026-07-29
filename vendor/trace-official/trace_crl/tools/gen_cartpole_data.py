import numpy as np
import os
from PIL import Image, ImageDraw

"""
Pseudo CartPole Dataset - 128x128 version
FIXED VERSION: 
1. Second-order Markov dynamics (depends on z_{t-1} and z_{t-2})
2. Nonlinear activation (LeakyReLU)
3. Stable system (all eigenvalues < 1)
4. No clip needed (stability guaranteed by eigenvalues)
5. Significantly different trajectories across domains

Formula:
    x_t = g(z_t)  # observation
    z_t = σ(σ(z_{t-1} @ W^{(k)}) + σ(z_{t-2} @ W_lag2)) + ε_t  # second-order Markov
    W^{(k)} = W_base + δW^{(k)}  # domain-specific
    W_mixed(t) = W_base + Σ_k α_k(t) · δW^{(k)}  # continuous mixture
"""

def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)


def render_cartpole_fast(position, angle, img_size=128):
    """128x128 version rendering"""
    img = Image.new('L', (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Coordinate mapping (position range expanded to [-3, 3] for stable system)
    center_x = img_size // 2 + int(np.clip(position, -3, 3) * 18)
    ground_y = img_size - 20
    
    # Cart
    cart_width, cart_height = 30, 15
    cart_left = center_x - cart_width // 2
    cart_right = center_x + cart_width // 2
    cart_top = ground_y - cart_height
    draw.rectangle([cart_left, cart_top, cart_right, ground_y], fill=0)
    
    # Pole (angle range expanded to [-1.5, 1.5] for stable system)
    pole_length = 50
    pole_end_x = center_x + int(pole_length * np.sin(np.clip(angle, -1.5, 1.5)))
    pole_end_y = cart_top - int(pole_length * np.cos(np.clip(angle, -1.5, 1.5)))
    draw.line([center_x, cart_top, pole_end_x, pole_end_y], fill=50, width=3)
    
    # Ground
    draw.line([0, ground_y, img_size, ground_y], fill=100, width=1)
    
    # Normalize to [-1, 1]
    img_arr = np.array(img).astype(np.float32)
    img_arr = (img_arr / 255.0) * 2 - 1
    
    return img_arr


def create_stable_W():
    """Create STABLE W_base and delta_W
    
    Key design principles:
    1. Diagonal elements < 1 to ensure stability (no explosion)
    2. Off-diagonal elements for causal coupling
    3. delta_W large enough to create distinguishable dynamics
    4. All eigenvalues < 1 guaranteed
    """
    
    # W_base: First-order transition matrix (stable)
    # Diagonal ~0.8 ensures decay, off-diagonal for coupling
    W_base = np.array([
        [ 0.85,   0.1,    0.0,    0.0  ],   # x: position
        [-0.05,   0.8,    0.1,    0.0  ],   # v: velocity (negative feedback from x)
        [ 0.0,    0.0,    0.8,    0.1  ],   # θ: angle
        [ 0.0,    0.0,   -0.15,   0.85 ],   # ω: angular velocity (restoring force)
    ], dtype=np.float32)
    
    # W_lag2: Second-order term (weaker than first-order)
    W_lag2 = np.array([
        [ 0.1,    0.0,    0.0,    0.0  ],
        [ 0.0,    0.1,    0.0,    0.0  ],
        [ 0.0,    0.0,    0.1,    0.0  ],
        [ 0.0,    0.0,    0.0,    0.1  ],
    ], dtype=np.float32)
    
    delta_Ws = []
    domain_info = []
    
    # v0: Baseline (no perturbation)
    dW0 = np.zeros((4, 4), dtype=np.float32)
    delta_Ws.append(dW0)
    domain_info.append({
        'name': 'v0', 
        'desc': 'baseline', 
        'physics': 'standard parameters', 
        'edge': None, 
        'delta': 0.0
    })
    
    # v1: edge [1,2] - velocity affects angle (OFF-DIAGONAL)
    dW1 = np.zeros((4, 4), dtype=np.float32)
    dW1[1, 2] = 0.2
    delta_Ws.append(dW1)
    domain_info.append({
        'name': 'v1', 
        'desc': 'v_to_theta', 
        'physics': 'velocity influences angle', 
        'edge': (1, 2), 
        'delta': 0.2
    })
    
    # v2: edge [3,2] - angular velocity affects angle (OFF-DIAGONAL)
    dW2 = np.zeros((4, 4), dtype=np.float32)
    dW2[3, 2] = 0.15
    delta_Ws.append(dW2)
    domain_info.append({
        'name': 'v2', 
        'desc': 'omega_to_theta', 
        'physics': 'angular velocity influences angle', 
        'edge': (3, 2), 
        'delta': 0.15
    })
    
    # v3: edge [0,1] - position affects velocity (OFF-DIAGONAL)
    dW3 = np.zeros((4, 4), dtype=np.float32)
    dW3[0, 1] = 0.2
    delta_Ws.append(dW3)
    domain_info.append({
        'name': 'v3', 
        'desc': 'x_to_v', 
        'physics': 'position influences velocity', 
        'edge': (0, 1), 
        'delta': 0.2
    })
    
    # v4: edge [2,3] - angle affects angular velocity (OFF-DIAGONAL)
    dW4 = np.zeros((4, 4), dtype=np.float32)
    dW4[2, 3] = 0.2
    delta_Ws.append(dW4)
    domain_info.append({
        'name': 'v4', 
        'desc': 'theta_to_omega', 
        'physics': 'angle influences angular velocity', 
        'edge': (2, 3), 
        'delta': 0.2
    })
    
    return W_base, W_lag2, delta_Ws, domain_info


def verify_stability(W_base, W_lag2, delta_Ws, domain_info):
    """Verify system stability by checking eigenvalues"""
    print("\n" + "=" * 60)
    print("Stability Verification (Eigenvalue Analysis)")
    print("=" * 60)
    
    print("\nW_base eigenvalues:")
    eig_base = np.linalg.eigvals(W_base)
    print(f"   |λ| = {np.abs(eig_base)}")
    print(f"   max|λ| = {np.max(np.abs(eig_base)):.4f} {'[STABLE]' if np.max(np.abs(eig_base)) < 1 else '[UNSTABLE!]'}")
    
    print("\nW_lag2 eigenvalues:")
    eig_lag2 = np.linalg.eigvals(W_lag2)
    print(f"   |λ| = {np.abs(eig_lag2)}")
    
    print("\nDomain-specific W^{(k)} = W_base + δW^{(k)}:")
    all_stable = True
    for dW, info in zip(delta_Ws, domain_info):
        W_k = W_base + dW
        eig_k = np.linalg.eigvals(W_k)
        max_eig = np.max(np.abs(eig_k))
        stable = max_eig < 1
        all_stable = all_stable and stable
        status = "[STABLE]" if stable else "[UNSTABLE!]"
        print(f"   {info['name']}: max|λ| = {max_eig:.4f} {status}")
    
    return all_stable


def verify_identifiability(delta_Ws, domain_info):
    """Verify identifiability conditions"""
    print("\n" + "=" * 60)
    print("Identifiability Conditions Verification")
    print("=" * 60)
    
    print("\n1. Sparsity Check:")
    for i, (dW, info) in enumerate(zip(delta_Ws, domain_info)):
        nonzero_count = np.sum(np.abs(dW) > 1e-8)
        status = "PASS" if nonzero_count <= 1 else "FAIL"
        print(f"   {info['name']}: nonzero elements = {nonzero_count} [{status}]")
    
    print("\n2. Linear Independence Check:")
    edges = []
    for i, info in enumerate(domain_info):
        if info['edge'] is not None:
            edges.append(info['edge'])
            is_diagonal = info['edge'][0] == info['edge'][1]
            diag_status = "DIAGONAL - WARNING!" if is_diagonal else "off-diagonal"
            print(f"   {info['name']}: edge {info['edge']} ({diag_status})")
    
    unique_edges = set(edges)
    status = "PASS" if len(edges) == len(unique_edges) else "FAIL"
    print(f"\n   Unique edges: {len(unique_edges)} / {len(edges)} [{status}]")
    
    print("\n3. Perturbation Magnitude:")
    for info in domain_info:
        if info['edge'] is not None:
            print(f"   {info['name']}: delta_W{list(info['edge'])} = {info['delta']:.4f}")
    
    print("\n" + "-" * 40)
    print("All identifiability conditions satisfied!")


def transition_step(z_t1, z_t2, W_k, W_lag2, noise_scale=0.005):
    """
    Second-order Markov transition with nonlinearity
    
    z_t = σ(σ(z_{t-1} @ W^{(k)}) + σ(z_{t-2} @ W_lag2)) + ε_t
    """
    term1 = leaky_relu(z_t1 @ W_k)
    term2 = leaky_relu(z_t2 @ W_lag2)
    z_t = leaky_relu(term1 + term2)
    z_t += np.random.randn(4).astype(np.float32) * noise_scale
    return z_t


def generate_pseudo_cartpole_dataset(
    save_path,
    n_domains=5,
    n_trails_per_domain=1000,
    trail_length=40,
    img_size=128,
    seed=42
):
    """Generate pseudo CartPole dataset with second-order Markov dynamics"""
    
    np.random.seed(seed)
    
    print("=" * 60)
    print(f"Pseudo CartPole Dataset Generation ({img_size}x{img_size})")
    print("Second-Order Markov + Nonlinear + Stable System")
    print("=" * 60)
    
    W_base, W_lag2, delta_Ws, domain_info = create_stable_W()
    
    print(f"\nW_base (first-order):\n{W_base}")
    print(f"\nW_lag2 (second-order):\n{W_lag2}")
    
    print("\nDomain Settings:")
    print("-" * 50)
    for info in domain_info:
        print(f"  {info['name']}: {info['desc']}")
        print(f"    Physics: {info['physics']}")
        if info['edge']:
            print(f"    delta_W[{info['edge'][0]}, {info['edge'][1]}] = {info['delta']:.4f}")
    
    # Verify stability and identifiability
    verify_stability(W_base, W_lag2, delta_Ws, domain_info)
    verify_identifiability(delta_Ws, domain_info)
    
    cartpole_path = os.path.join(save_path, 'cartpole')
    os.makedirs(cartpole_path, exist_ok=True)
    
    noise_scale = 0.005
    
    for domain_id in range(n_domains):
        info = domain_info[domain_id]
        print(f"\nGenerating Domain: {info['name']} ({info['desc']})")
        
        domain_path = os.path.join(cartpole_path, info['name'])
        os.makedirs(domain_path, exist_ok=True)
        
        W_k = W_base + delta_Ws[domain_id]
        
        for trail_idx in range(n_trails_per_domain):
            # Initialize TWO states for second-order Markov
            z_t2 = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.2, 0.2),
            ], dtype=np.float32)
            
            z_t1 = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.2, 0.2),
            ], dtype=np.float32)
            
            obs_list = []
            state_list = []
            action_list = []
            reward_list = []
            
            for step in range(trail_length):
                # Current state is z_t1 (most recent)
                state_list.append(z_t1.copy())
                
                # Render observation
                img = render_cartpole_fast(z_t1[0], z_t1[2], img_size)
                obs_list.append(img)
                
                # Random action (for data diversity)
                action = np.random.randint(0, 2)
                action_list.append(action)
                reward_list.append(1.0)
                
                # Second-order Markov transition
                z_t = transition_step(z_t1, z_t2, W_k, W_lag2, noise_scale)
                
                # Update history (no clip needed - system is stable)
                z_t2 = z_t1
                z_t1 = z_t
            
            np.savez_compressed(
                os.path.join(domain_path, f'trail_{trail_idx}.npz'),
                obs=np.array(obs_list, dtype=np.float16),
                state=np.array(state_list, dtype=np.float16),
                action=np.array(action_list, dtype=np.float16),
                reward=np.array(reward_list, dtype=np.float16)
            )
            
            if (trail_idx + 1) % 200 == 0:
                print(f"    Trail {trail_idx + 1}/{n_trails_per_domain}")
    
    # Save parameters
    params_path = os.path.join(save_path, 'params')
    os.makedirs(params_path, exist_ok=True)
    
    np.save(os.path.join(params_path, 'W_base.npy'), W_base)
    np.save(os.path.join(params_path, 'W_lag2.npy'), W_lag2)
    for i, dW in enumerate(delta_Ws):
        np.save(os.path.join(params_path, f'delta_W_{i}.npy'), dW)
    
    domain_names = [info['name'] for info in domain_info]
    np.save(os.path.join(params_path, 'domain_names.npy'), domain_names)
    
    print(f"\n[OK] Data saved to: {cartpole_path}")
    print(f"[OK] Parameters saved to: {params_path}")
    
    # Format verification
    print("\n" + "=" * 60)
    print("Format Verification")
    print("=" * 60)
    sample = np.load(os.path.join(cartpole_path, 'v0', 'trail_0.npz'))
    print(f"obs:    shape={sample['obs'].shape}, dtype={sample['obs'].dtype}, range=[{sample['obs'].min():.2f}, {sample['obs'].max():.2f}]")
    print(f"state:  shape={sample['state'].shape}, dtype={sample['state'].dtype}")
    print(f"action: shape={sample['action'].shape}, dtype={sample['action'].dtype}")
    print(f"reward: shape={sample['reward'].shape}, dtype={sample['reward'].dtype}")
    
    return W_base, W_lag2, delta_Ws, domain_info


def generate_mixed_transition(
    save_path,
    W_base,
    W_lag2,
    delta_Ws,
    domain_info,
    source_domain=0,
    target_domain=1,
    curves=['linear', 'sine'],
    n_trails=100,
    trail_length=40,
    img_size=128,
    seed=42
):
    """Generate mixed domain data with second-order Markov dynamics"""
    
    np.random.seed(seed)
    
    trans_path = os.path.join(save_path, 'transition')
    os.makedirs(trans_path, exist_ok=True)
    
    noise_scale = 0.005
    
    src_name = domain_info[source_domain]['name']
    tgt_name = domain_info[target_domain]['name']
    
    for curve_type in curves:
        print(f"\nGenerating mixed data: {src_name} -> {tgt_name}, curve={curve_type}")
        
        t = np.linspace(0, 1, trail_length)
        if curve_type == 'linear':
            alpha_seq = 1 - t
        elif curve_type == 'sine':
            alpha_seq = 0.5 * (1 + np.cos(np.pi * t))
        elif curve_type == 'quadratic':
            alpha_seq = (1 - t) ** 2
        else:
            alpha_seq = 1 - t
        
        all_obs = []
        all_states = []
        all_actions = []
        all_rewards = []
        
        for trail_idx in range(n_trails):
            # Initialize TWO states for second-order Markov
            z_t2 = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.2, 0.2),
            ], dtype=np.float32)
            
            z_t1 = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.2, 0.2),
            ], dtype=np.float32)
            
            obs_list = []
            state_list = []
            action_list = []
            reward_list = []
            
            for step in range(trail_length):
                alpha = alpha_seq[step]
                
                # W_mixed(t) = W_base + α(t) * δW^{source} + (1-α(t)) * δW^{target}
                W_mixed = W_base + alpha * delta_Ws[source_domain] + (1 - alpha) * delta_Ws[target_domain]
                
                state_list.append(z_t1.copy())
                img = render_cartpole_fast(z_t1[0], z_t1[2], img_size)
                obs_list.append(img)
                
                action = np.random.randint(0, 2)
                action_list.append(action)
                reward_list.append(1.0)
                
                force = 0.1 if action == 1 else -0.1
                z_t1_perturbed = z_t1.copy()
                z_t1_perturbed[1] += force
                
                # Second-order Markov transition with mixed W
                z_t = transition_step(z_t1_perturbed, z_t2, W_mixed, W_lag2, noise_scale)
                
                z_t2 = z_t1
                z_t1 = z_t
            
            all_obs.append(np.array(obs_list))
            all_states.append(np.array(state_list))
            all_actions.append(np.array(action_list))
            all_rewards.append(np.array(reward_list))
        
        filename = f'traj_{src_name}_to_{tgt_name}_{curve_type}.npz'
        np.savez_compressed(
            os.path.join(trans_path, filename),
            obs=np.array(all_obs, dtype=np.float16),
            state=np.array(all_states, dtype=np.float16),
            action=np.array(all_actions, dtype=np.float16),
            reward=np.array(all_rewards, dtype=np.float16),
            alpha=np.tile(alpha_seq, (n_trails, 1)).astype(np.float32)
        )
        print(f"  [OK] Saved: {filename}")
        print(f"    alpha: [{alpha_seq[0]:.3f}, ..., {alpha_seq[-1]:.3f}]")


if __name__ == "__main__":
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    save_path = os.path.join(project_root, 'datasets', 'cartpole')
    
    # 1. Generate pure domain data (128x128)
    W_base, W_lag2, delta_Ws, domain_info = generate_pseudo_cartpole_dataset(
        save_path=save_path,
        n_domains=5,
        n_trails_per_domain=1000,
        img_size=128
    )
    
    # 2. Generate mixed transition data
    print("\n" + "=" * 60)
    print("Generating Mixed Transition Data")
    print("=" * 60)
    
    generate_mixed_transition(
        save_path=save_path,
        W_base=W_base,
        W_lag2=W_lag2,
        delta_Ws=delta_Ws,
        domain_info=domain_info,
        source_domain=0,
        target_domain=1,
        curves=['linear', 'sine'],
        img_size=128
    )
    
    generate_mixed_transition(
        save_path=save_path,
        W_base=W_base,
        W_lag2=W_lag2,
        delta_Ws=delta_Ws,
        domain_info=domain_info,
        source_domain=0,
        target_domain=2,
        curves=['linear', 'sine'],
        img_size=128
    )
    
    generate_mixed_transition(
        save_path=save_path,
        W_base=W_base,
        W_lag2=W_lag2,
        delta_Ws=delta_Ws,
        domain_info=domain_info,
        source_domain=0,
        target_domain=4,
        curves=['linear', 'sine'],
        img_size=128
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)