import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pathlib import Path
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def parse_loss_log(log_file, sample_interval=50, debug=False):
    """解析日志文件 - 修复全局步数计算"""
    steps = []
    losses = []

    # 🔥 关键: step 字段已经是全局步数，直接捕获
    loss_pattern = r'\[.*?\]:\s*epoch\s+\d+-step\s+(\d+)\s+loss:\s+([\d.]+)'

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            loss_count = 0
            last_global_step = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue

                loss_match = re.search(loss_pattern, line)
                if loss_match:
                    global_step = int(loss_match.group(1))  # 直接使用
                    loss = float(loss_match.group(2))

                    last_global_step = global_step

                    if loss_count % sample_interval == 0:
                        steps.append(global_step)
                        losses.append(loss)

                    loss_count += 1

    except Exception as e:
        print(f"❌ 解析 {log_file} 错误: {e}")
        return None, None

    if debug and steps:
        arr = np.array(steps)
        print(f"  📄 匹配 {loss_count} 条 | 最后全局 Step: {last_global_step:,}")
        print(f"  📊 采样后: {len(steps)} 点 | 最大 Step: {arr.max():,} ({arr.max() / 1000:.1f}K)")

    return np.array(steps) if steps else None, np.array(losses) if losses else None


def smooth_curve(data, window_size=50):
    if len(data) < window_size:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def plot_loss_comparison(log_configs, save_path='loss_comparison.png',
                         window_size=50, max_step_k=None, debug=False):
    if not log_configs:
        print("❌ 没有配置任何日志文件")
        return

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    all_max_step = 0

    for idx, (model_name, log_file) in enumerate(log_configs.items()):
        print(f"📁 解析: {model_name} -> {Path(log_file).name}")
        steps, losses = parse_loss_log(log_file, sample_interval=50, debug=debug)

        if steps is None or len(steps) == 0:
            print(f"⚠️ 跳过 {model_name}: 无有效数据")
            continue

        # 平滑
        if len(losses) >= window_size:
            losses_smooth = smooth_curve(losses, window_size)
            steps_smooth = steps[window_size - 1:]
        else:
            losses_smooth, steps_smooth = losses, steps

        steps_k = steps_smooth / 1000
        if steps_k.max() > all_max_step:
            all_max_step = steps_k.max()

        # 绘图
        color = colors[idx % len(colors)]
        ax.plot(steps_k, losses_smooth, color=color, linewidth=1.8, label=model_name, alpha=0.95)

        # 终点标注
        if len(losses_smooth) > 0:
            ax.plot(steps_k[-1], losses_smooth[-1], 'o', color=color, markersize=4)
            ax.text(steps_k[-1] + all_max_step * 0.01, losses_smooth[-1],
                    f'{losses_smooth[-1]:.3f}', fontsize=7, color=color, va='center')

    if ax.lines:
        # 🔥 X 轴动态跟随数据
        x_max = max_step_k if max_step_k else all_max_step * 1.05
        ax.set_xlim(0, x_max)
        print(f"📏 X 轴范围: 0 - {x_max:.1f}K (数据最大 {all_max_step:.1f}K)")

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5, zorder=0)
        ax.set_xlabel('Training Step (K)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='gray')
        ax.set_title('Training Loss Comparison', fontsize=13, fontweight='bold', pad=15)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {save_path}")
        plt.close()
    else:
        print("❌ 没有有效数据可绘制")


# ============ 主程序 ============
if __name__ == "__main__":
    LOG_CONFIGS = {
        'ProMoE-L-Full MOE': 'joint_phase1.log',
    }

    SAVE_PATH = 'loss_comparison.png'
    SMOOTH_WINDOW = 50
    MAX_STEP_K = None
    DEBUG = True  # 🔥 开启调试输出

    print("🎨 生成 Loss 对比图...")
    plot_loss_comparison(LOG_CONFIGS, SAVE_PATH, SMOOTH_WINDOW, MAX_STEP_K, debug=DEBUG)