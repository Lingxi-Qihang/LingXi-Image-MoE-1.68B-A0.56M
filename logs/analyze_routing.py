import re
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import warnings

# 尝试导入 wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB 未安装，将跳过云端日志上传功能 (pip install wandb)")

warnings.filterwarnings('ignore')

# =================配置区域=================
EXCLUDE_EXPERT_ID = 12  # 无条件专家 ID (CFG)
COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


# =========================================

def parse_moe_log(filepath):
    """全量解析日志文件"""
    pattern = re.compile(r'Step (\d+), Layer (\d+): (\{.*?\})')
    data = defaultdict(lambda: defaultdict(dict))
    parse_errors = 0

    print(f"📁 正在解析日志: {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            try:
                step = int(match.group(1))
                layer = int(match.group(2))
                dict_str = match.group(3)
                util_dict = ast.literal_eval(dict_str)
                util_dict = {int(k): v for k, v in util_dict.items()}
                data[step][layer] = util_dict
            except Exception as e:
                parse_errors += 1

    if parse_errors > 0:
        print(f"⚠️ 解析过程中发生 {parse_errors} 次错误，已自动跳过。")
    if not data:
        raise ValueError("❌ 未解析到任何有效数据。")
    return data


def compute_cv(values):
    """计算变异系数 CV = std / mean"""
    if len(values) < 2: return np.nan
    mean_val = np.mean(values)
    if mean_val == 0: return np.nan
    return np.std(values, ddof=1) / mean_val


def compute_layer_cv(step_data, exclude_uncond=True):
    """计算某一步所有层的 CV"""
    layer_cvs = {}
    for layer, expert_utils in step_data.items():
        values = list(expert_utils.values())
        if exclude_uncond and EXCLUDE_EXPERT_ID in expert_utils:
            values = [v for k, v in expert_utils.items() if k != EXCLUDE_EXPERT_ID]
        layer_cvs[layer] = compute_cv(values)
    return layer_cvs


def detect_dead_experts(all_step_data, threshold=0.001):
    """检测死专家"""
    if not all_step_data: return set()
    all_experts = set()
    for step_data in all_step_data.values():
        for layer_data in step_data.values():
            all_experts.update(layer_data.keys())
    if EXCLUDE_EXPERT_ID in all_experts: all_experts.remove(EXCLUDE_EXPERT_ID)

    dead_experts = set()
    for exp_id in all_experts:
        max_util = max(
            [layer_data.get(exp_id, 0) for step_data in all_step_data.values() for layer_data in step_data.values()])
        if max_util < threshold: dead_experts.add(exp_id)
    return dead_experts


def compute_expert_activity(all_step_data):
    """计算专家平均活跃度"""
    expert_sums = defaultdict(float)
    expert_counts = defaultdict(int)
    for step_data in all_step_data.values():
        for layer_data in step_data.values():
            for exp_id, util in layer_data.items():
                if exp_id != EXCLUDE_EXPERT_ID:
                    expert_sums[exp_id] += util
                    expert_counts[exp_id] += 1
    return {exp_id: expert_sums[exp_id] / expert_counts[exp_id] for exp_id in expert_sums}


# =================新增可视化函数=================

def plot_layer_expert_preference(all_step_data, target_steps, save_path=None):
    """
    绘制所有 12 层 (Layer 0-11) 的专家利用率柱状图。
    布局：4 行 x 3 列
    直观展示每一层的专家分工模式。
    """
    # 获取所有存在的层并排序
    # 假设第一步的数据包含所有层作为参考
    first_step = min(all_step_data.keys())
    all_layers = sorted(all_step_data[first_step].keys())

    n_rows = 4
    n_cols = 6

    # 创建子图网格
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 16), sharey=True)
    # 将 axes 展平为一维数组方便迭代
    axes_flat = axes.flatten()

    # 取最后一个目标步数的数据进行展示（代表收敛后的状态）
    display_step = target_steps[-1] if target_steps else max(all_step_data.keys())
    if display_step not in all_step_data:
        print(f"⚠️ Step {display_step} 数据缺失，使用最后一步替代")
        display_step = max(all_step_data.keys())

    step_data = all_step_data[display_step]

    for i, layer_idx in enumerate(all_layers):
        ax = axes_flat[i]

        if layer_idx not in step_data:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer_idx}')
            continue

        expert_utils = step_data[layer_idx]
        # 排除 CFG 专家 (Expert 12) 以便看清路由专家分布
        experts = sorted([k for k in expert_utils.keys() if k != EXCLUDE_EXPERT_ID])

        if not experts:
            ax.text(0.5, 0.5, 'No Routing Experts', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer_idx}')
            continue

        utils = [expert_utils[k] for k in experts]

        # 使用不同颜色区分不同的层，或者统一颜色
        # 这里使用 viridis 色谱根据层深着色，体现深度感
        color = plt.cm.viridis(layer_idx / len(all_layers))

        bars = ax.bar(range(len(experts)), utils, color=color, alpha=0.8, edgecolor='gray', linewidth=0.5)

        # 设置标题和标签
        ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(experts)))
        ax.set_xticklabels([str(e) for e in experts], rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 可选：在柱状图上标注数值（如果太挤可以注释掉）
        # for bar in bars:
        #     height = bar.get_height()
        #     ax.text(bar.get_x() + bar.get_width() / 2., height,
        #             f'{height:.2f}', ha='center', va='bottom', fontsize=6)

    # 隐藏多余的子图（如果层数少于 12）
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # 设置总标题和 Y 轴标签
    plt.suptitle(f'Expert Utilization Distribution Across All Layers at Step {display_step:,}',
                 fontsize=16, fontweight='bold', y=0.98)

    # 只在最左侧一列显示 Y 轴标签
    for ax in axes[:, 0]:
        ax.set_ylabel('Utilization', fontsize=12)

    # 调整布局以防止重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 全层专家偏好图 (4x3) 已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cfg_expert_tracking(all_step_data, save_path=None):
    """
    追踪 CFG 无条件专家 (Expert 12) 的利用率随时间的变化。
    验证其是否稳定在预期值（通常与 class_dropout_prob 相关，如 0.1 或 0.25）。
    """
    steps = sorted(all_step_data.keys())
    cfg_utils_per_step = []

    for step in steps:
        # 取所有层中 Expert 12 的平均利用率，或者取第一层的作为代表
        # 这里我们取所有层的平均值，更能反映整体行为
        layer_utils = []
        for layer_data in all_step_data[step].values():
            if EXCLUDE_EXPERT_ID in layer_data:
                layer_utils.append(layer_data[EXCLUDE_EXPERT_ID])

        if layer_utils:
            cfg_utils_per_step.append(np.mean(layer_utils))
        else:
            cfg_utils_per_step.append(np.nan)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, cfg_utils_per_step, color='#d62728', linewidth=2, marker='.', markersize=4,
             label=f'Expert {EXCLUDE_EXPERT_ID} (CFG)')

    # 添加参考线：假设 class_dropout_prob = 0.1 或 0.25，这里画一条 0.1 和 0.25 的线供参考
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Ref: 0.1 (Dropout 10%)')
    plt.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, label='Ref: 0.25 (Dropout 25% or TopK effect)')

    plt.title(f'CFG Unconditional Expert (Exp {EXCLUDE_EXPERT_ID}) Utilization Over Time', fontsize=14)
    plt.xlabel('Training Step')
    plt.ylabel('Average Utilization')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 CFG 专家追踪图已保存: {save_path}")
    else:
        plt.show()
    plt.close()


# =================原有绘图函数（略作精简以节省空间）=================
def plot_single_step_cv(layer_cvs, step, save_path=None):
    layers = sorted(layer_cvs.keys())
    cvs = [layer_cvs[l] for l in layers]
    plt.figure(figsize=(10, 5))
    plt.plot(layers, cvs, marker='o', color='#2E86AB', linewidth=2, markersize=6, label=f'Step {step:,}')
    plt.axhline(y=np.nanmean(cvs), color='gray', linestyle='--', alpha=0.5, label='Mean CV')
    plt.title(f'Router Differentiation at Step {step:,}', fontsize=14)
    plt.xlabel('Layer Index');
    plt.ylabel('Differentiation (CV)')
    plt.xticks(layers);
    plt.grid(True, alpha=0.3);
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"📊 单步图已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_multi_step_cv(all_step_data, steps_list, save_path=None):
    plt.figure(figsize=(12, 6))
    for i, step in enumerate(steps_list):
        if step not in all_step_data: continue
        layer_cvs = compute_layer_cv(all_step_data[step])
        layers = sorted(layer_cvs.keys())
        cvs = [layer_cvs[l] for l in layers]
        plt.plot(layers, cvs, marker='o', linestyle='-', color=COLOR_PALETTE[i % len(COLOR_PALETTE)], linewidth=2,
                 markersize=6, label=f'Step {step:,}')
    plt.title('Router Differentiation Evolution (Multi-Step)', fontsize=14)
    plt.xlabel('Layer Index');
    plt.ylabel('Differentiation (CV)')
    plt.xticks(sorted(layers));
    plt.grid(True, alpha=0.3);
    plt.legend(loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"📊 多步对比图已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_cv_heatmap(all_step_data, save_path=None):
    steps = sorted(all_step_data.keys())
    base_layers = sorted(all_step_data[steps[0]].keys())
    cv_matrix = np.full((len(steps), len(base_layers)), np.nan)
    for i, step in enumerate(steps):
        layer_cvs = compute_layer_cv(all_step_data[step])
        for j, layer in enumerate(base_layers):
            if layer in layer_cvs: cv_matrix[i, j] = layer_cvs[layer]
    plt.figure(figsize=(14, 8))
    sns.heatmap(cv_matrix, annot=False, cmap='viridis', cbar_kws={'label': 'CV Value'},
                xticklabels=[f'L{l}' for l in base_layers], yticklabels=[f'{s:,}' for s in steps])
    plt.title('Evolution of Layer-wise Router Differentiation (Heatmap)', fontsize=14)
    plt.xlabel('Layer Index');
    plt.ylabel('Training Step')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f" 热力图已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_expert_activity(activity_dict, dead_experts, save_path=None):
    experts = sorted(activity_dict.keys())
    utils = [activity_dict[e] for e in experts]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(experts)), utils, color='#2E86AB')
    for i, exp_id in enumerate(experts):
        if exp_id in dead_experts: bars[i].set_color('#d62728')
    plt.xticks(range(len(experts)), [str(e) for e in experts])
    plt.title('Average Expert Utilization (Activity)', fontsize=14)
    plt.xlabel('Expert ID');
    plt.ylabel('Avg Utilization')
    plt.grid(True, axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2E86AB', label='Active Expert'),
                       Patch(facecolor='#d62728', label='Dead Expert (<0.1%)')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"📊 专家活跃度图已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_active_experts_evolution(all_step_data, threshold=0.02, save_path=None):
    """优化版：更清晰的可视化"""
    steps = sorted(all_step_data.keys())
    layers = sorted(all_step_data[steps[0]].keys())

    # 构建矩阵
    active_counts = np.zeros((len(steps), len(layers)))
    for i, step in enumerate(steps):
        for j, layer in enumerate(layers):
            layer_data = all_step_data[step][layer]
            count = sum(1 for exp_id, util in layer_data.items()
                        if exp_id != EXCLUDE_EXPERT_ID and util > threshold)
            active_counts[i, j] = count

    # ========== 图1：分层统计（每层单独子图）==========
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10), sharex=True)
    axes_flat = axes.flatten()

    for layer_idx, ax in enumerate(axes_flat):
        if layer_idx >= len(layers):
            ax.set_visible(False)
            continue

        ax.plot(steps, active_counts[:, layer_idx], linewidth=2, color='#2E86AB')
        ax.axhline(y=6, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title(f'Layer {layer_idx}', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 12)
        ax.grid(True, alpha=0.2)

        # 只在最左侧显示Y轴标签
        if layer_idx % n_cols != 0:
            ax.set_yticklabels([])

    axes_flat[0].set_ylabel('Active Experts', fontsize=11)
    plt.suptitle(f'Active Experts per Layer Over Time (Threshold > {threshold * 100}%)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path.replace('.png', '_per_layer.png'), dpi=150, bbox_inches='tight')
        print(f"📊 分层激活专家数图已保存")
    plt.close()

    # ========== 图2：聚合统计（更清晰！）==========
    plt.figure(figsize=(12, 6))

    # 计算每步的平均/最小/最大激活数
    avg_counts = active_counts.mean(axis=1)
    min_counts = active_counts.min(axis=1)
    max_counts = active_counts.max(axis=1)

    plt.fill_between(steps, min_counts, max_counts, alpha=0.3, color='#2E86AB',
                     label='Range (Min-Max)')
    plt.plot(steps, avg_counts, linewidth=3, color='#2E86AB', label='Average')
    plt.axhline(y=6, color='red', linestyle='--', linewidth=2, label='Target: 6 experts')

    plt.title('Overall Active Experts Statistics', fontsize=14, fontweight='bold')
    plt.xlabel('Training Step')
    plt.ylabel('Number of Active Experts')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_stats.png'), dpi=150, bbox_inches='tight')
        print(f"📊 聚合统计图已保存")
    plt.close()

    # ========== 图3：热力图（优化版）==========
    plt.figure(figsize=(14, 6))

    # 采样X轴标签（只显示10个关键步数）
    sample_indices = np.linspace(0, len(steps) - 1, 10, dtype=int)
    sample_steps = [steps[i] for i in sample_indices]

    sns.heatmap(active_counts.T, annot=False, cmap='YlGnBu', vmin=0, vmax=8,
                cbar_kws={'label': 'Active Experts'},
                xticklabels=[f'{s:,}' for s in sample_steps],
                yticklabels=[f'L{l}' for l in layers])

    plt.title(f'Evolution of Active Experts per Layer (Threshold > {threshold * 100}%)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Training Step')
    plt.ylabel('Layer Index')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 热力图已保存: {save_path}")
    else:
        plt.show()
    plt.close()

    # ========== 打印关键统计 ==========
    print(f"\n📊 激活专家数统计 (Threshold > {threshold * 100}%):")
    print("-" * 60)

    # 关键步数统计
    # 关键步数统计 - 每 10k 步打印一次
    interval = 50000
    key_steps = sorted(list(set(
        [s for s in steps if s % interval == 0] +
        [steps[0], steps[-1]]  # 确保首尾
    )))
    for step in key_steps:
        idx = steps.index(step)
        avg = active_counts[idx].mean()
        min_val = active_counts[idx].min()
        max_val = active_counts[idx].max()
        layers_below_3 = (active_counts[idx] < 3).sum()

        print(f"Step {step:5,}: Avg={avg:4.1f} | Min={min_val:3.0f} | Max={max_val:3.0f} | "
              f"Layers<3={layers_below_3}/12")

    # 趋势判断
    first_quarter_avg = active_counts[:len(steps) // 4].mean()
    last_quarter_avg = active_counts[-len(steps) // 4:].mean()
    trend = "↑ 改善" if last_quarter_avg > first_quarter_avg else "↓ 恶化" if last_quarter_avg < first_quarter_avg else "→ 持平"

    print(f"\n📈 总体趋势: {trend} ({first_quarter_avg:.1f} → {last_quarter_avg:.1f})")

    if last_quarter_avg < 4:
        print("⚠️  警告：平均激活专家数仍低于4，可能需要干预！")
    elif last_quarter_avg >= 6:
        print("✅ 优秀：平均激活专家数已达目标！")


def plot_comprehensive_summary(all_step_data, save_path=None):
    """
    Generate comprehensive summary plot containing:
    (a) Evolution of active experts
    (b) CV evolution heatmap
    (c) Expert utilization distribution across layers at current step
    (d) Global average expert utilization
    """
    print("🎨 Generating comprehensive summary plot...")

    # Prepare data
    steps = sorted(all_step_data.keys())
    layers = sorted(all_step_data[steps[0]].keys())

    # 1. Calculate evolution of active experts
    threshold = 0.02
    active_counts = []
    for step in steps:
        step_counts = []
        for layer in layers:
            layer_data = all_step_data[step][layer]
            count = sum(1 for exp_id, util in layer_data.items()
                        if exp_id != EXCLUDE_EXPERT_ID and util > threshold)
            step_counts.append(count)
        active_counts.append(step_counts)
    active_counts = np.array(active_counts)

    # 2. Calculate CV heatmap data
    cv_matrix = np.full((len(steps), len(layers)), np.nan)
    for i, step in enumerate(steps):
        for j, layer in enumerate(layers):
            expert_utils = all_step_data[step][layer]
            values = [v for k, v in expert_utils.items() if k != EXCLUDE_EXPERT_ID]
            cv_matrix[i, j] = compute_cv(values)

    # 3. Calculate global expert activity
    activity = compute_expert_activity(all_step_data)

    # 4. Get expert utilization distribution at final step
    final_step = steps[-1]

    # ========== Create comprehensive layout ==========
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

    # --- (a) Evolution of Active Experts ---
    ax1 = fig.add_subplot(gs[0, :2])  # Occupy first row, first two columns

    avg_counts = active_counts.mean(axis=1)
    min_counts = active_counts.min(axis=1)
    max_counts = active_counts.max(axis=1)

    ax1.fill_between(steps, min_counts, max_counts, alpha=0.3, color='#2E86AB',
                     label='Range (Min-Max)')
    ax1.plot(steps, avg_counts, linewidth=3, color='#2E86AB', label='Average')
    ax1.axhline(y=6, color='red', linestyle='--', linewidth=2, label='Target: 6 experts')

    ax1.set_title('(a) Evolution of Active Experts', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Number of Active Experts')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 12)

    # --- (b) CV Evolution Heatmap ---
    ax2 = fig.add_subplot(gs[0, 2])  # First row, third column

    # Sampling display (avoid crowding)
    if len(steps) > 20:
        sample_indices = np.linspace(0, len(steps) - 1, 20, dtype=int)
        sample_steps = [steps[i] for i in sample_indices]
        cv_matrix_sampled = cv_matrix[sample_indices, :]
    else:
        sample_steps = steps
        cv_matrix_sampled = cv_matrix

    # 修改：使用 origin='upper'，让 Y 轴从上到下显示从 0 到最大步数
    im = ax2.imshow(cv_matrix_sampled, aspect='auto', cmap='viridis',
                    vmin=0, vmax=2.5, origin='upper')

    # Y轴是步数（从上到下：0 → 最大），X轴是层
    ax2.set_yticks(range(len(sample_steps)))
    ax2.set_yticklabels([f'{s // 1000}k' for s in sample_steps], fontsize=8)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels([f'L{l}' for l in layers])

    ax2.set_title('(b) CV Evolution Heatmap', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Training Step')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('CV Value', rotation=270, labelpad=15)

    # --- (c) Expert Utilization Distribution Across Layers ---
    ax3 = fig.add_subplot(gs[1, :])  # Second row, all columns

    final_step_data = all_step_data[final_step]
    n_layers = len(layers)
    n_experts = len([k for k in final_step_data[0].keys() if k != EXCLUDE_EXPERT_ID])

    # Create stacked bar chart
    x = np.arange(n_layers)
    width = 0.6

    # Get all experts
    experts = sorted([k for k in final_step_data[0].keys() if k != EXCLUDE_EXPERT_ID])

    # Collect data
    util_matrix = np.zeros((n_layers, n_experts))
    for i, layer in enumerate(layers):
        for j, exp_id in enumerate(experts):
            util_matrix[i, j] = final_step_data[layer].get(exp_id, 0)

    # Stacked bar chart
    bottom = np.zeros(n_layers)
    colors = plt.cm.Set3(np.linspace(0, 1, n_experts))

    for j, exp_id in enumerate(experts):
        ax3.bar(x, util_matrix[:, j], width, bottom=bottom,
                label=f'Exp {exp_id}', color=colors[j], alpha=0.8, edgecolor='gray', linewidth=0.3)
        bottom += util_matrix[:, j]

    ax3.set_title(f'(c) Expert Utilization Distribution at Step {final_step:,}',
                  fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Utilization')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'L{l}' for l in layers])
    ax3.legend(loc='upper right', ncol=4, fontsize=8)
    ax3.grid(True, axis='y', alpha=0.3)

    # --- (d) Global Average Expert Utilization ---
    ax4 = fig.add_subplot(gs[2, :])  # Third row, all columns

    exp_ids = sorted(activity.keys())
    utils = [activity[e] for e in exp_ids]

    bars = ax4.bar(range(len(exp_ids)), utils, color='#2E86AB', alpha=0.8,
                   edgecolor='gray', linewidth=0.5)

    # Annotate values
    for i, (bar, util) in enumerate(zip(bars, utils)):
        if util > 0.01:  # Only show significant ones
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{util:.3f}', ha='center', va='bottom', fontsize=9)

    ax4.set_title('(d) Global Average Expert Utilization', fontsize=14, fontweight='bold', pad=10)
    ax4.set_xlabel('Expert ID')
    ax4.set_ylabel('Avg Utilization')
    ax4.set_xticks(range(len(exp_ids)))
    ax4.set_xticklabels([str(e) for e in exp_ids])
    ax4.grid(True, axis='y', alpha=0.3)

    # Add statistics text box
    stats_text = (f"Total Steps: {len(steps):,}\n"
                  f"Final Step: {final_step:,}\n"
                  f"Avg Active Experts: {avg_counts[-1]:.2f}\n"
                  f"Avg CV: {np.nanmean(cv_matrix[-1, :]):.3f}")

    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Main title
    plt.suptitle(f'Cross-Attention Router Health Comprehensive Report',
                 fontsize=18, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Comprehensive summary plot saved: {save_path}")
    else:
        plt.show()
    plt.close()

# =================主逻辑=================
def main():
    parser = argparse.ArgumentParser(description="Advanced MoE Router Analyzer v3")
    parser.add_argument('--log', type=str, default="moe_routing.log", help='Path to MoE routing log file')
    parser.add_argument('--steps', type=str, default="100000 300000 600000 900000 1200000 14000000",
                        help='Specific steps to analyze')
    parser.add_argument('--mode', type=str, choices=['single', 'multi', 'heatmap', 'all'], default='all',
                        help='Analysis mode')
    parser.add_argument('--output_dir', type=str, default='moe_analysis_output', help='Directory to save plots')
    parser.add_argument('--wandb_project', type=str, default=None, help='WandB project name (optional)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. 解析日志
    all_data = parse_moe_log(args.log)
    available_steps = sorted(all_data.keys())

    # 2. 确定要分析的步数
    if args.steps:
        target_steps = [int(i) for i in (args.steps).split()]
    else:
        target_steps = available_steps[-5:] if len(available_steps) >= 5 else available_steps
    print(f"🎯 目标分析步数: {target_steps}")

    # 3. 全局分析
    print("\n--- 🔍 全局诊断 ---")
    dead_experts = detect_dead_experts(all_data)
    if dead_experts:
        print(f"⚠️ 发现死专家 (Util < 0.1%): {sorted(dead_experts)}")
    else:
        print("✅ 未发现死专家，所有专家均有激活。")
    activity = compute_expert_activity(all_data)
    print(f"📊 专家平均活跃度范围: [{min(activity.values()):.4f}, {max(activity.values()):.4f}]")

    # WandB 初始化
    if WANDB_AVAILABLE and args.wandb_project:
        wandb.init(project=args.wandb_project, config={"log_file": args.log})
        wandb.log({"dead_experts_count": len(dead_experts)})
        wandb.log({"avg_expert_utilization": np.mean(list(activity.values()))})

    # 4. 执行绘图模式
    if args.mode in ['single', 'all']:
        for step in target_steps:
            if step in all_data:
                layer_cvs = compute_layer_cv(all_data[step])
                save_p = output_dir / f"cv_step_{step}.png"
                plot_single_step_cv(layer_cvs, step, save_path=str(save_p))
                if WANDB_AVAILABLE and args.wandb_project:
                    wandb.log({f"mean_cv_step_{step}": np.nanmean(list(layer_cvs.values())), "step": step})

    if args.mode in ['multi', 'all']:
        save_p = output_dir / "cv_multi_step_compare.png"
        plot_multi_step_cv(all_data, target_steps, save_path=str(save_p))

    if args.mode in ['heatmap', 'all']:
        heatmap_steps = available_steps[::max(1, len(available_steps) // 20)]
        heatmap_data = {s: all_data[s] for s in heatmap_steps if s in all_data}
        if heatmap_data:
            save_p = output_dir / "cv_evolution_heatmap.png"
            plot_cv_heatmap(heatmap_data, save_path=str(save_p))

    # 5. 专家活跃度图
    save_p = output_dir / "expert_activity_ranking.png"
    plot_expert_activity(activity, dead_experts, save_path=str(save_p))

    # =================新增功能调用=================
    # 6. 分层专家偏好可视化 (Layer 0, 4, 8, 11)
    save_p = output_dir / "layer_expert_preference.png"
    plot_layer_expert_preference(all_data, target_steps, save_path=str(save_p))

    # 7. CFG 无条件专家行为追踪
    save_p = output_dir / "cfg_expert_tracking.png"
    plot_cfg_expert_tracking(all_data, save_path=str(save_p))
    # ============================================

    # 🆕 8. 激活专家数随时间演化（新增！）
    save_p = output_dir / "active_experts_evolution.png"
    plot_active_experts_evolution(all_data, threshold=0.02, save_path=str(save_p))
    # ============================================

    # 🆕 9. 综合性总图（新增！）
    save_p = output_dir / "comprehensive_summary.png"
    plot_comprehensive_summary(all_data, save_path=str(save_p))

    if WANDB_AVAILABLE and args.wandb_project:
        wandb.finish()

    print(f"\n✅ 分析完成！结果已保存至: {output_dir.absolute()}")


if __name__ == "__main__":
    main()