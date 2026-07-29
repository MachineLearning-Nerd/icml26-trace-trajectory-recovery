# generate_transition_data_3domain.py
"""
Generate 3-domain mixed transition data for cartpole
- Non-linear alpha curves
- Domain 0: cosine decay (1 -> 0)
- Domain 1: bell curve (0 -> peak -> 0)
- Domain 2: complementary rise (0 -> 1)
"""

import numpy as np
import os
from PIL import Image, ImageDraw


def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)


def render_cartpole_fast(position, angle, img_size=128):
    img = Image.new('L', (img_size, img_size), color=255)
    draw = ImageDraw.Draw(img)
    
    center_x = img_size // 2 + int(np.clip(position, -3, 3) * 18)
    ground_y = img_size - 20
    
    cart_width, cart_height = 30, 15
    cart_left = center_x - cart_width // 2
    cart_right = center_x + cart_width // 2
    cart_top = ground_y - cart_height
    draw.rectangle([cart_left, cart_top, cart_right, ground_y], fill=0)
    
    pole_length = 50
    pole_end_x = center_x + int(pole_length * np.sin(np.clip(angle, -1.5, 1.5)))
    pole_end_y = cart_top - int(pole_length * np.cos(np.clip(angle, -1.5, 1.5)))
    draw.line([center_x, cart_top, pole_end_x, pole_end_y], fill=50, width=3)
    
    draw.line([0, ground_y, img_size, ground_y], fill=100, width=1)
    
    img_arr = np.array(img).astype(np.float32)
    img_arr = (img_arr / 255.0) * 2 - 1
    
    return img_arr


def transition_step(z_t1, z_t2, W_k, W_lag2, noise_scale=0.02):
    term1 = leaky_relu(z_t1 @ W_k)
    term2 = leaky_relu(z_t2 @ W_lag2)
    z_t = leaky_relu(term1 + term2)
    z_t += np.random.randn(4).astype(np.float32) * noise_scale
    return z_t


def load_params(params_path):
    W_base = np.load(os.path.join(params_path, 'W_base.npy'))
    W_lag2 = np.load(os.path.join(params_path, 'W_lag2.npy'))
    
    delta_Ws = []
    for i in range(5):
        dW = np.load(os.path.join(params_path, f'delta_W_{i}.npy'))
        delta_Ws.append(dW)
    
    domain_names = np.load(os.path.join(params_path, 'domain_names.npy'))
    
    return W_base, W_lag2, delta_Ws, domain_names


def generate_alpha_curves(trail_length, curve_type='nonlinear', peak_scale=0.6):
    """
    生成三个domain的alpha曲线
    
    Args:
        trail_length: 轨迹长度
        curve_type: 'nonlinear' (钟形) 或 'linear' (线性)
        peak_scale: 中间domain的峰值高度
    
    Returns:
        alpha_0, alpha_1, alpha_2: 三个domain的alpha序列
    """
    t = np.linspace(0, 1, trail_length)
    
    if curve_type == 'nonlinear':
        # Domain 0: cosine 下降 (1 -> 0)
        alpha_0 = 0.5 * (1 + np.cos(np.pi * t))
        
        # Domain 1: 钟形曲线，中间最高 (0 -> peak -> 0)
        alpha_1 = np.sin(np.pi * t) ** 2 * peak_scale
        
        # Domain 2: 1 - alpha_0 - alpha_1，确保和为 1
        alpha_2 = 1 - alpha_0 - alpha_1
        alpha_2 = np.clip(alpha_2, 0, 1)
        
    elif curve_type == 'linear':
        # 线性版本
        # Domain 0: 线性下降
        alpha_0 = 1 - t
        
        # Domain 1: 三角形，中间最高
        alpha_1 = np.where(t < 0.5, t * 2 * peak_scale, (1 - t) * 2 * peak_scale)
        
        # Domain 2: 补足
        alpha_2 = 1 - alpha_0 - alpha_1
        alpha_2 = np.clip(alpha_2, 0, 1)
        
    elif curve_type == 'smooth':
        # 更平滑的版本，用高斯
        alpha_0 = np.exp(-((t - 0) ** 2) / (2 * 0.3 ** 2))
        alpha_1 = np.exp(-((t - 0.5) ** 2) / (2 * 0.2 ** 2)) * peak_scale
        alpha_2 = np.exp(-((t - 1) ** 2) / (2 * 0.3 ** 2))
    
    else:
        raise ValueError(f"Unknown curve_type: {curve_type}")
    
    # 归一化确保和为1
    alpha_sum = alpha_0 + alpha_1 + alpha_2
    alpha_0 = alpha_0 / alpha_sum
    alpha_1 = alpha_1 / alpha_sum
    alpha_2 = alpha_2 / alpha_sum
    
    return alpha_0, alpha_1, alpha_2


def generate_3domain_transition(
    save_path,
    W_base,
    W_lag2,
    delta_Ws,
    domain_names,
    domains=[0, 1, 2],
    curve_types=['nonlinear', 'linear'],
    n_trails=100,
    trail_length=30,
    img_size=128,
    noise_scale=0.03,
    peak_scale=0.6,
    seed=42
):
    """
    生成三domain混合的transition数据
    
    Args:
        domains: 三个参与的domain id，例如 [0, 1, 2]
        curve_types: alpha曲线类型列表
        peak_scale: 中间domain的峰值
    """
    np.random.seed(seed)
    
    trans_path = os.path.join(save_path, 'transition')
    os.makedirs(trans_path, exist_ok=True)
    
    d0, d1, d2 = domains
    nclass = len(delta_Ws)
    
    for curve_type in curve_types:
        print(f"\nGenerating 3-domain transition:")
        print(f"  Domains: {d0} -> {d1} -> {d2}")
        print(f"  Curve type: {curve_type}")
        
        # 生成alpha曲线
        alpha_0, alpha_1, alpha_2 = generate_alpha_curves(
            trail_length, curve_type=curve_type, peak_scale=peak_scale
        )
        
        # 构造完整的alpha sequence (n_trails, T, nclass)
        alpha_sequence = np.zeros((trail_length, nclass))
        alpha_sequence[:, d0] = alpha_0
        alpha_sequence[:, d1] = alpha_1
        alpha_sequence[:, d2] = alpha_2
        
        print(f"  Alpha at key points:")
        print(f"    t=0:   α_{d0}={alpha_0[0]:.3f}, α_{d1}={alpha_1[0]:.3f}, α_{d2}={alpha_2[0]:.3f}")
        mid = trail_length // 2
        print(f"    t={mid}:  α_{d0}={alpha_0[mid]:.3f}, α_{d1}={alpha_1[mid]:.3f}, α_{d2}={alpha_2[mid]:.3f}")
        print(f"    t={trail_length-1}:  α_{d0}={alpha_0[-1]:.3f}, α_{d1}={alpha_1[-1]:.3f}, α_{d2}={alpha_2[-1]:.3f}")
        
        all_obs = []
        all_states = []
        
        for trail_idx in range(n_trails):
            # 初始化状态
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
            
            for step in range(trail_length):
                # 混合 W: W = W_base + sum_k alpha_k * delta_W_k
                W_mixed = W_base.copy()
                for k in range(nclass):
                    W_mixed += alpha_sequence[step, k] * delta_Ws[k]
                
                state_list.append(z_t1.copy())
                img = render_cartpole_fast(z_t1[0], z_t1[2], img_size)
                obs_list.append(img)
                
                z_t = transition_step(z_t1, z_t2, W_mixed, W_lag2, noise_scale)
                
                z_t2 = z_t1
                z_t1 = z_t
            
            all_obs.append(np.array(obs_list))
            all_states.append(np.array(state_list))
        
        # 保存
        filename = f'traj_3domain_{d0}_{d1}_{d2}_{curve_type}.npz'
        np.savez_compressed(
            os.path.join(trans_path, filename),
            obs=np.array(all_obs, dtype=np.float16),
            state=np.array(all_states, dtype=np.float16),
            alpha=np.tile(alpha_sequence, (n_trails, 1, 1)).astype(np.float32),  # (n_trails, T, nclass)
            alpha_per_domain=np.stack([
                np.tile(alpha_0, (n_trails, 1)),
                np.tile(alpha_1, (n_trails, 1)),
                np.tile(alpha_2, (n_trails, 1))
            ], axis=-1).astype(np.float32),  # (n_trails, T, 3) 方便可视化
            domains=np.array(domains),
            curve_type=curve_type
        )
        print(f"  Saved: {filename}")
        
        # 验证
        states = np.array(all_states)
        print(f"  State norm: start={np.linalg.norm(states[:, 0, :], axis=1).mean():.4f}, "
              f"end={np.linalg.norm(states[:, -1, :], axis=1).mean():.4f}")


def visualize_alpha_curves(trail_length=30, save_path=None):
    """可视化不同类型的alpha曲线"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, curve_type in enumerate(['nonlinear', 'linear']):
        alpha_0, alpha_1, alpha_2 = generate_alpha_curves(trail_length, curve_type)
        t = np.arange(trail_length)
        
        axes[idx].plot(t, alpha_0, 'b-', lw=2, label='Domain 0 (decay)')
        axes[idx].plot(t, alpha_1, 'g-', lw=2, label='Domain 1 (bell)')
        axes[idx].plot(t, alpha_2, 'r-', lw=2, label='Domain 2 (rise)')
        axes[idx].plot(t, alpha_0 + alpha_1 + alpha_2, 'k--', lw=1, alpha=0.5, label='Sum')
        
        axes[idx].set_xlabel('Time step')
        axes[idx].set_ylabel('Alpha')
        axes[idx].set_title(f'Alpha Curves - {curve_type}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved alpha curves visualization to: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    base_path = os.path.join(project_root, 'datasets', 'cartpole')
    params_path = os.path.join(base_path, 'params')
    
    print("Loading parameters...")
    W_base, W_lag2, delta_Ws, domain_names = load_params(params_path)
    
    print("\n" + "="*60)
    print("Generating 3-Domain Transition Data")
    print("="*60)
    
    # 可视化alpha曲线
    visualize_alpha_curves(
        trail_length=30, 
        save_path=os.path.join(base_path, 'transition', 'alpha_curves_3domain.png')
    )
    
    # 生成 domain 0 -> 1 -> 2 的transition
    generate_3domain_transition(
        save_path=base_path,
        W_base=W_base,
        W_lag2=W_lag2,
        delta_Ws=delta_Ws,
        domain_names=domain_names,
        domains=[0, 1, 2],
        curve_types=['nonlinear', 'linear'],
        n_trails=100,
        trail_length=30,
        noise_scale=0.002,  # 降低噪声以满足A4条件
        peak_scale=0.6,
        seed=42
    )
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)