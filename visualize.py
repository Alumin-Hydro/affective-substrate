"""
affective_substrate.visualize
==============================
可视化工具：波形、相图、PCA轨迹、Lyapunov指数
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
import os


def get_chinese_font():
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return FontProperties(fname=fp)
    return FontProperties(family='sans-serif')


def plot_waveforms(history: dict, output_path: str = "waveforms.png",
                   title: str = "双系统动力学波形"):
    """
    绘制x(t)和y(t)的时间序列
    """
    font_prop = get_chinese_font()
    n_steps = len(history['x'])
    n_hormones = history['x'].shape[1]
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), facecolor='#0d1b2a')
    
    colors_x = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff'][:n_hormones]
    colors_y = ['#ff8787', '#ffe066', '#80d8a0', '#74b3ff'][:n_hormones]
    labels = ['快系统 x(t) — 交感', '慢系统 y(t) — 副交感', '有效阻尼 γ_eff']
    
    for ax_idx, (data_key, colors, label) in enumerate([
        ('x', colors_x, labels[0]),
        ('y', colors_y, labels[1]),
        ('gamma_eff', colors_y, labels[2]),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor('#0d1b2a')
        for i in range(n_hormones):
            ax.plot(history[data_key][:, i], color=colors[i], 
                   alpha=0.8, linewidth=0.8, label=f'hormone_{i}')
        ax.set_ylabel(label, fontproperties=font_prop, color='white', fontsize=10)
        ax.tick_params(colors='#888888')
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a2e', 
                 edgecolor='#333333', labelcolor='white')
    
    axes[-1].set_xlabel('步数', fontproperties=font_prop, color='white', fontsize=10)
    axes[0].set_title(title, fontproperties=font_prop, color='white', fontsize=14, pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.close()
    print(f"Waveforms saved to {output_path}")
    return output_path


def plot_phase_portrait(history: dict, output_path: str = "phase_portrait.png",
                        title: str = "快慢系统相图"):
    """
    绘制x-y相图（快慢系统耦合）
    """
    font_prop = get_chinese_font()
    n_hormones = history['x'].shape[1]
    
    fig, axes = plt.subplots(1, n_hormones, figsize=(5*n_hormones, 5), 
                             facecolor='#0d1b2a')
    if n_hormones == 1:
        axes = [axes]
    
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff'][:n_hormones]
    
    for i, ax in enumerate(axes):
        ax.set_facecolor('#0d1b2a')
        x = history['x'][:, i]
        y = history['y'][:, i]
        
        # 用时间着色
        points = ax.scatter(x, y, c=np.arange(len(x)), cmap='coolwarm', 
                          s=1, alpha=0.6)
        ax.set_xlabel(f'x_{i} (快)', fontproperties=font_prop, color='white', fontsize=9)
        ax.set_ylabel(f'y_{i} (慢)', fontproperties=font_prop, color='white', fontsize=9)
        ax.set_title(f'hormone_{i}', fontproperties=font_prop, color=colors[i], fontsize=10)
        ax.tick_params(colors='#888888')
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle(title, fontproperties=font_prop, color='white', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.close()
    print(f"Phase portrait saved to {output_path}")
    return output_path


def plot_reservoir_pca(history: dict, output_path: str = "reservoir_pca.png",
                       title: str = "储备池状态PCA轨迹"):
    """
    对储备池高维状态做PCA降维，绘制轨迹
    """
    font_prop = get_chinese_font()
    
    r = history['r']  # [n_steps, n_units]
    pca = PCA(n_components=3)
    r_pca = pca.fit_transform(r)
    
    fig = plt.figure(figsize=(12, 8), facecolor='#0d1b2a')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d1b2a')
    
    # 时间着色
    n_steps = len(r_pca)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_steps))
    
    for i in range(n_steps - 1):
        ax.plot(r_pca[i:i+2, 0], r_pca[i:i+2, 1], r_pca[i:i+2, 2],
               color=colors[i], alpha=0.6, linewidth=0.5)
    
    ax.set_xlabel('PC1', color='white', fontsize=9)
    ax.set_ylabel('PC2', color='white', fontsize=9)
    ax.set_zlabel('PC3', color='white', fontsize=9)
    ax.set_title(f'{title}\n解释方差: {pca.explained_variance_ratio_.sum():.1%}',
                fontproperties=font_prop, color='white', fontsize=12, pad=15)
    ax.tick_params(colors='#888888')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.close()
    print(f"Reservoir PCA saved to {output_path}")
    return output_path


def plot_injection_strength(history: dict, output_path: str = "injection.png",
                            title: str = "激活注入强度监控"):
    """
    监控 ||α·v|| / ||h|| 的比例
    """
    font_prop = get_chinese_font()
    
    # 计算注入强度（用x的范数近似）
    injection_norm = np.linalg.norm(history['x'], axis=1)
    
    fig, ax = plt.subplots(figsize=(14, 4), facecolor='#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    
    ax.plot(injection_norm, color='#ff6b6b', linewidth=0.8)
    ax.axhline(y=np.mean(injection_norm), color='#ffd93d', linestyle='--', 
              alpha=0.5, label=f'均值: {np.mean(injection_norm):.3f}')
    
    ax.set_ylabel('||x|| (注入强度)', fontproperties=font_prop, color='white', fontsize=10)
    ax.set_xlabel('步数', fontproperties=font_prop, color='white', fontsize=10)
    ax.set_title(title, fontproperties=font_prop, color='white', fontsize=12, pad=10)
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0d1b2a')
    plt.close()
    print(f"Injection strength saved to {output_path}")
    return output_path


def generate_all(substrate, output_dir: str = ".", n_steps: int = 3000):
    """生成所有可视化"""
    from core import AffectiveVisualizer
    
    print("Running simulation...")
    history = AffectiveVisualizer.run_simulation(substrate, n_steps=n_steps)
    
    print("Generating plots...")
    plot_waveforms(history, os.path.join(output_dir, "waveforms.png"))
    plot_phase_portrait(history, os.path.join(output_dir, "phase_portrait.png"))
    plot_reservoir_pca(history, os.path.join(output_dir, "reservoir_pca.png"))
    plot_injection_strength(history, os.path.join(output_dir, "injection.png"))
    
    print("All done!")
    return history
