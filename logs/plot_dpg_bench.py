import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
# 设置全局字体为黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 解析 dpg-bench.txt
# ==========================================
file_path = "dpg-bench.txt"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

blocks = re.split(r'\n(?=Model: outputs_)', content)

data_l1 = {}
data_l2 = {}

l1_keys = ['relation', 'entity', 'attribute', 'global', 'other']
l2_key_map = {
    'relation - spatial': 'relation-spatial',
    'relation - non-spatial': 'relation-non-spatial',
    'entity - whole': 'entity-whole',
    'entity - part': 'entity-part',
    'entity - state': 'entity-state',
    'attribute - color': 'attribute-color',
    'attribute - texture': 'attribute-texture',
    'attribute - shape': 'attribute-shape',
    'attribute - size': 'attribute-size',
    'attribute - other': 'attribute-other',
    'other - count': 'other-count',
    'other - text': 'other-text',
    'global -': 'global'
}

for block in blocks:
    model_match = re.search(r'Model: outputs_(\w+)', block)
    if not model_match:
        continue
    step_str = model_match.group(1)
    try:
        step = int(step_str)
        if step < 1000:
            step *= 10000
    except ValueError:
        continue

    overall_match = re.search(r'DPG-Bench score:\s*([\d.]+)', block)
    overall = float(overall_match.group(1)) if overall_match else None

    l1_scores = {}
    for key in l1_keys:
        match = re.search(rf'{key}:\s*([\d.]+)', block, re.IGNORECASE)
        if match:
            l1_scores[key.capitalize()] = float(match.group(1))
    l1_scores['Overall'] = overall
    data_l1[step] = l1_scores

    l2_scores = {}
    lines = block.split('\n')
    in_l2 = False
    for line in lines:
        if 'L2 category scores:' in line:
            in_l2 = True
            continue
        if in_l2 and line.strip() == '':
            break
        if in_l2:
            match = re.match(r'\s*([a-zA-Z\s\-]+):\s*([\d.]+)', line)
            if match:
                raw_key = match.group(1).strip()
                val = float(match.group(2))
                for map_key, pretty_name in l2_key_map.items():
                    if map_key == raw_key:
                        l2_scores[pretty_name] = val
                        break
    data_l2[step] = l2_scores

steps = sorted(data_l1.keys())
print(f"解析完成，共 {len(steps)} 个评测点")

# ==========================================
# 2. 全局样式设置
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

L1_COLORS = {
    'Overall': '#d62728',
    'Relation': '#1f77b4',
    'Entity': '#ff7f0e',
    'Attribute': '#2ca02c',
    'Global': '#9467bd',
    'Other': '#8c564b'
}
L1_MARKERS = {'Overall': 'o', 'Relation': 's', 'Entity': '^', 'Attribute': 'D', 'Global': 'v', 'Other': 'p'}
L1_LW = {'Overall': 2.5, 'Relation': 1.8, 'Entity': 1.8, 'Attribute': 1.8, 'Global': 1.8, 'Other': 1.8}

# 按评测大类分组
L2_GROUPS = {
    'Attribute': ['attribute-color', 'attribute-texture', 'attribute-shape', 'attribute-size', 'attribute-other'],
    'Entity': ['entity-whole', 'entity-part', 'entity-state'],
    'Other': ['other-count', 'other-text', 'global'],
    'Relation': ['relation-spatial', 'relation-non-spatial']
}

# ==========================================
# 3. 绘制 L1 图
# ==========================================
fig1, ax1 = plt.subplots(figsize=(12, 8))
fig1.suptitle('LingXi-Image-MoE (1.68B) DPG-Bench L1 Training Dynamics',
              fontsize=16, fontweight='bold', y=0.98)

for metric in ['Overall', 'Relation', 'Entity', 'Attribute', 'Global', 'Other']:
    x = []
    y = []
    for s in steps:
        if metric in data_l1[s]:
            x.append(s)
            y.append(data_l1[s][metric])
    ax1.plot(x, y, color=L1_COLORS[metric], label=metric,
             linewidth=L1_LW[metric], marker=L1_MARKERS[metric],
             markersize=7, markeredgecolor='white', markeredgewidth=1.2)
    for xi, yi in zip(x, y):
        ax1.annotate(f'{yi:.1f}', (xi, yi),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=7, fontweight='bold',
                     color=L1_COLORS[metric])

ax1.axhline(y=80, color='red', linestyle=':', linewidth=1.5, alpha=0.6, label='Target: 80')
ax1.set_title('L1 Categories & Overall Score', fontweight='bold', pad=15)
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('DPG-Bench Score')
ax1.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
ax1.set_ylim(40, 100)
ax1.set_xlim(steps[0] - 20000, steps[-1] + 20000)

plt.tight_layout()
plt.savefig('dpg_bench_L1.png', dpi=300)
plt.show()

# ==========================================
# 4. 绘制 L2 分组子图（按评测大类：2×2 布局）
# ==========================================
fig2, axes = plt.subplots(2, 2, figsize=(18, 14))
fig2.suptitle('LingXi-Image-MoE (1.68B) DPG-Bench L2 Category Dynamics',
              fontsize=16, fontweight='bold', y=1.01)

colors = plt.cm.tab10.colors

for idx, (group_name, metrics) in enumerate(L2_GROUPS.items()):
    ax = axes[idx // 2, idx % 2]
    for i, metric in enumerate(metrics):
        x = []
        y = []
        for s in steps:
            if metric in data_l2[s]:
                x.append(s)
                y.append(data_l2[s][metric])
        ax.plot(x, y, color=colors[i % len(colors)], label=metric,
                linewidth=1.8, marker='o', markersize=4, markeredgecolor='white')
        if x:
            ax.annotate(f'{y[-1]:.1f}', (x[-1], y[-1]),
                        textcoords="offset points", xytext=(5, 0),
                        ha='left', fontsize=6, color=colors[i % len(colors)])

    ax.set_title(group_name, fontweight='bold', pad=10, fontsize=13)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Score')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x / 1000)}K'))
    ax.legend(loc='best', fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.autoscale(enable=True, axis='y')
    ax.margins(y=0.1)
    ax.set_xlim(steps[0] - 20000, steps[-1] + 20000)

if len(L2_GROUPS) < 4:
    for i in range(len(L2_GROUPS), 4):
        axes[i // 2, i % 2].set_visible(False)

plt.tight_layout()
plt.savefig('dpg_bench_L2.png', dpi=300)
plt.show()

print("图表已保存：dpg_bench_L1.png 和 dpg_bench_L2.png")