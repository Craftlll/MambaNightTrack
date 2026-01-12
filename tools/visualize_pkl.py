#!/usr/bin/env python
"""
可视化 eval_data.pkl 文件为图表

用法:
    # 生成所有图表
    python tools/visualize_pkl.py output/test/result_plots/nat2021l/eval_data.pkl
    
    # 指定保存目录
    python tools/visualize_pkl.py eval_data.pkl --output_dir ./plots
    
    # 只生成特定图表
    python tools/visualize_pkl.py eval_data.pkl --plots success prec
    
    # 显示图表（而非只保存）
    python tools/visualize_pkl.py eval_data.pkl --show
"""

import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from pathlib import Path

# 设置专业学术论文配色方案
# 使用深色、饱和度适中的颜色，适合论文打印和展示
ACADEMIC_COLORS = [
    '#1f77b4',  # 深蓝色 - 专业、稳重
    '#d62728',  # 深红色 - 对比强烈
    '#2ca02c',  # 深绿色 - 清晰可辨
    '#ff7f0e',  # 橙色 - 温暖明亮
    '#9467bd',  # 紫色 - 优雅高贵
    '#8c564b',  # 棕色 - 沉稳内敛
    '#e377c2',  # 粉色 - 柔和对比
    '#7f7f7f',  # 灰色 - 中性平衡
    '#bcbd22',  # 黄绿色 - 活力四射
    '#17becf',  # 青色 - 清新明快
]

# 设置matplotlib全局样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16


def get_tracker_display_name(tracker):
    """获取 tracker 显示名称"""
    if tracker.get('disp_name'):
        return tracker['disp_name']
    elif tracker.get('run_id') is not None:
        return f"{tracker['name']}_{tracker['param']}_{tracker['run_id']:03d}"
    else:
        return f"{tracker['name']}_{tracker['param']}"


def plot_success_curve(data, output_dir, show=False):
    """绘制成功率曲线（AUC）"""
    print("📊 绘制成功率曲线 (Success Plot / AUC)...")
    
    threshold_set = torch.tensor(data['threshold_set_overlap'])
    ave_success_rate = torch.tensor(data['ave_success_rate_plot_overlap'])
    valid_sequence = torch.tensor(data['valid_sequence'], dtype=torch.bool)
    trackers = data['trackers']
    
    # 只取有效序列
    ave_success_rate = ave_success_rate[valid_sequence, :, :]
    auc_curve = ave_success_rate.mean(0) * 100.0  # (num_trackers, num_thresholds)
    auc = auc_curve.mean(-1)  # (num_trackers,)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = [ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i in range(len(trackers))]
    
    for trk_id, (tracker, color) in enumerate(zip(trackers, colors)):
        name = get_tracker_display_name(tracker)
        auc_score = auc[trk_id].item()
        
        ax.plot(threshold_set.numpy(), auc_curve[trk_id, :].numpy(),
                label=f'{name} [AUC: {auc_score:.2f}]',
                linewidth=2.5, color=color)
    
    ax.set_xlabel('Overlap threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success rate [%]', fontsize=14, fontweight='bold')
    ax.set_title('Success Plot (AUC)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # 保存
    save_path = output_dir / 'success_plot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_precision_curve(data, output_dir, show=False, normalized=False):
    """绘制精度曲线"""
    curve_type = "归一化精度" if normalized else "精度"
    print(f"📊 绘制{curve_type}曲线 (Precision Plot)...")
    
    if normalized:
        threshold_set = torch.tensor(data['threshold_set_center_norm'])
        ave_precision_rate = torch.tensor(data['ave_success_rate_plot_center_norm'])
        filename = 'normalized_precision_plot.png'
        title = 'Normalized Precision Plot'
        xlabel = 'Location error threshold'
    else:
        threshold_set = torch.tensor(data['threshold_set_center'])
        ave_precision_rate = torch.tensor(data['ave_success_rate_plot_center'])
        filename = 'precision_plot.png'
        title = 'Precision Plot'
        xlabel = 'Location error threshold [pixels]'
    
    valid_sequence = torch.tensor(data['valid_sequence'], dtype=torch.bool)
    trackers = data['trackers']
    
    # 只取有效序列
    ave_precision_rate = ave_precision_rate[valid_sequence, :, :]
    prec_curve = ave_precision_rate.mean(0) * 100.0
    prec_score = prec_curve[:, 20]  # Precision at threshold 20
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = [ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i in range(len(trackers))]
    
    for trk_id, (tracker, color) in enumerate(zip(trackers, colors)):
        name = get_tracker_display_name(tracker)
        score = prec_score[trk_id].item()
        
        ax.plot(threshold_set.numpy(), prec_curve[trk_id, :].numpy(),
                label=f'{name} [Prec: {score:.2f}]',
                linewidth=2.5, color=color)
    
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision [%]', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=12)
    
    if normalized:
        ax.set_xlim([0, 0.5])
    else:
        ax.set_xlim([0, 50])
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # 保存
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_bar(data, output_dir, show=False):
    """绘制性能对比柱状图"""
    print("📊 绘制性能对比柱状图...")
    
    trackers = data['trackers']
    valid_sequence = torch.tensor(data['valid_sequence'], dtype=torch.bool)
    
    # 计算各项指标
    threshold_set_overlap = torch.tensor(data['threshold_set_overlap'])
    ave_success_rate_plot_overlap = torch.tensor(data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    
    # 找到对应阈值的索引
    op50_idx = (threshold_set_overlap == 0.50).nonzero(as_tuple=True)[0][0].item()
    op75_idx = (threshold_set_overlap == 0.75).nonzero(as_tuple=True)[0][0].item()
    
    metrics = {}
    metrics['AUC'] = auc_curve.mean(-1).numpy().tolist()
    metrics['OP50'] = auc_curve[:, op50_idx].numpy().tolist()
    metrics['OP75'] = auc_curve[:, op75_idx].numpy().tolist()
    
    if 'ave_success_rate_plot_center' in data:
        ave_prec = torch.tensor(data['ave_success_rate_plot_center'])
        ave_prec = ave_prec[valid_sequence, :, :]
        prec_curve = ave_prec.mean(0) * 100.0
        metrics['Precision'] = prec_curve[:, 20].numpy().tolist()
    
    if 'ave_success_rate_plot_center_norm' in data:
        ave_norm_prec = torch.tensor(data['ave_success_rate_plot_center_norm'])
        ave_norm_prec = ave_norm_prec[valid_sequence, :, :]
        norm_prec_curve = ave_norm_prec.mean(0) * 100.0
        metrics['Norm Prec'] = norm_prec_curve[:, 20].numpy().tolist()
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    tracker_names = [get_tracker_display_name(t) for t in trackers]
    x = np.arange(len(metrics))
    width = 0.8 / len(trackers)
    
    colors = [ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i in range(len(trackers))]
    
    # 确保所有metrics的值都是标量列表
    for key in metrics:
        if isinstance(metrics[key], np.ndarray):
            if metrics[key].ndim == 0:
                metrics[key] = [float(metrics[key])]
            else:
                metrics[key] = metrics[key].tolist()
    
    for trk_id, (name, color) in enumerate(zip(tracker_names, colors)):
        offset = (trk_id - len(trackers)/2 + 0.5) * width
        values = [float(metrics[metric][trk_id]) for metric in metrics.keys()]
        ax.bar(x + offset, values, width, label=name, color=color, alpha=0.85, edgecolor='white', linewidth=1.2)
    
    ax.set_xlabel('指标', fontsize=14, fontweight='bold')
    ax.set_ylabel('分数 (%)', fontsize=14, fontweight='bold')
    ax.set_title('性能对比', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()))
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    # 添加数值标签
    for trk_id in range(len(trackers)):
        offset = (trk_id - len(trackers)/2 + 0.5) * width
        for i, metric in enumerate(metrics.keys()):
            value = metrics[metric][trk_id]
            ax.text(i + offset, value + 2, f'{value:.1f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    save_path = output_dir / 'comparison_bar.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_per_sequence_performance(data, output_dir, show=False):
    """绘制每个序列的性能"""
    print("📊 绘制每序列性能...")
    
    sequences = data['sequences']
    trackers = data['trackers']
    avg_overlap_all = torch.tensor(data['avg_overlap_all']) * 100.0  # (num_seq, num_trackers)
    valid_sequence = torch.tensor(data['valid_sequence'], dtype=torch.bool)
    
    # 只显示有效序列
    valid_sequences = [seq for seq, valid in zip(sequences, valid_sequence) if valid]
    avg_overlap_valid = avg_overlap_all[valid_sequence, :]
    
    # 如果序列太多，只显示前 30 个
    max_display = 30
    if len(valid_sequences) > max_display:
        print(f"   序列数过多 ({len(valid_sequences)})，只显示前 {max_display} 个")
        valid_sequences = valid_sequences[:max_display]
        avg_overlap_valid = avg_overlap_valid[:max_display, :]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x = np.arange(len(valid_sequences))
    width = 0.8 / len(trackers)
    colors = [ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i in range(len(trackers))]
    
    tracker_names = [get_tracker_display_name(t) for t in trackers]
    
    for trk_id, (name, color) in enumerate(zip(tracker_names, colors)):
        offset = (trk_id - len(trackers)/2 + 0.5) * width
        values = avg_overlap_valid[:, trk_id].numpy().tolist()
        ax.bar(x + offset, values, width, label=name, color=color, alpha=0.85, edgecolor='white', linewidth=1.2)
    
    ax.set_xlabel('序列', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均重叠率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('每序列性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_sequences, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # 保存
    save_path = output_dir / 'per_sequence_performance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_in_one(data, output_dir, show=False):
    """绘制综合图表（所有曲线在一张图）"""
    print("📊 绘制综合图表...")
    
    trackers = data['trackers']
    valid_sequence = torch.tensor(data['valid_sequence'], dtype=torch.bool)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = [ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i in range(len(trackers))]
    tracker_names = [get_tracker_display_name(t) for t in trackers]
    
    # 1. Success Plot (AUC)
    ax = axes[0, 0]
    threshold_set_overlap = torch.tensor(data['threshold_set_overlap'])
    ave_success_rate = torch.tensor(data['ave_success_rate_plot_overlap'])
    ave_success_rate = ave_success_rate[valid_sequence, :, :]
    auc_curve = ave_success_rate.mean(0) * 100.0
    auc = auc_curve.mean(-1)
    
    for trk_id, (name, color) in enumerate(zip(tracker_names, colors)):
        ax.plot(threshold_set_overlap.numpy(), auc_curve[trk_id, :].numpy(),
                label=f'{name} [{auc[trk_id].item():.2f}]',
                linewidth=2.5, color=color)
    
    ax.set_xlabel('Overlap threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success rate [%]', fontsize=12, fontweight='bold')
    ax.set_title('Success Plot (AUC)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])
    
    # 2. Precision Plot
    if 'ave_success_rate_plot_center' in data:
        ax = axes[0, 1]
        threshold_set_center = torch.tensor(data['threshold_set_center'])
        ave_prec = torch.tensor(data['ave_success_rate_plot_center'])
        ave_prec = ave_prec[valid_sequence, :, :]
        prec_curve = ave_prec.mean(0) * 100.0
        prec_score = prec_curve[:, 20]
        
        for trk_id, (name, color) in enumerate(zip(tracker_names, colors)):
            ax.plot(threshold_set_center.numpy(), prec_curve[trk_id, :].numpy(),
                    label=f'{name} [{prec_score[trk_id].item():.2f}]',
                    linewidth=2.5, color=color)
        
        ax.set_xlabel('Location error threshold [pixels]', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision [%]', fontsize=12, fontweight='bold')
        ax.set_title('Precision Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([0, 50])
        ax.set_ylim([0, 100])
    
    # 3. Normalized Precision Plot
    if 'ave_success_rate_plot_center_norm' in data:
        ax = axes[1, 0]
        threshold_set_norm = torch.tensor(data['threshold_set_center_norm'])
        ave_norm_prec = torch.tensor(data['ave_success_rate_plot_center_norm'])
        ave_norm_prec = ave_norm_prec[valid_sequence, :, :]
        norm_prec_curve = ave_norm_prec.mean(0) * 100.0
        norm_prec_score = norm_prec_curve[:, 20]
        
        for trk_id, (name, color) in enumerate(zip(tracker_names, colors)):
            ax.plot(threshold_set_norm.numpy(), norm_prec_curve[trk_id, :].numpy(),
                    label=f'{name} [{norm_prec_score[trk_id].item():.2f}]',
                    linewidth=2.5, color=color)
        
        ax.set_xlabel('Location error threshold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Precision [%]', fontsize=12, fontweight='bold')
        ax.set_title('Normalized Precision Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([0, 0.5])
        ax.set_ylim([0, 100])
    
    # 4. Performance Summary (Bar Chart)
    ax = axes[1, 1]
    
    # 确保所有值都是标量numpy数组
    def extract_scalar(tensor, idx):
        """从张量中提取标量值"""
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 0:
                return tensor.item()
            elif tensor.dim() == 1:
                return tensor[idx].item()
            else:
                return tensor[idx, 0].item() if tensor.shape[1] == 1 else tensor[idx].mean().item()
        elif isinstance(tensor, np.ndarray):
            if tensor.ndim == 0:
                return float(tensor)
            elif tensor.ndim == 1:
                return float(tensor[idx])
            else:
                return float(tensor[idx, 0]) if tensor.shape[1] == 1 else float(tensor[idx].mean())
        else:
            return float(tensor)
    
    op50_idx = (threshold_set_overlap == 0.50).nonzero(as_tuple=True)[0][0].item()
    op75_idx = (threshold_set_overlap == 0.75).nonzero(as_tuple=True)[0][0].item()
    
    metrics = {}
    metrics['AUC'] = [extract_scalar(auc, i) for i in range(len(trackers))]
    metrics['OP50'] = [extract_scalar(auc_curve[:, op50_idx], i) for i in range(len(trackers))]
    metrics['OP75'] = [extract_scalar(auc_curve[:, op75_idx], i) for i in range(len(trackers))]
    
    if 'ave_success_rate_plot_center' in data:
        metrics['Prec'] = [extract_scalar(prec_score, i) for i in range(len(trackers))]
    if 'ave_success_rate_plot_center_norm' in data:
        metrics['NPrec'] = [extract_scalar(norm_prec_score, i) for i in range(len(trackers))]
    
    x_pos = np.arange(len(metrics))
    width_bar = 0.8 / len(trackers)
    
    for trk_id, (name, color) in enumerate(zip(tracker_names, colors)):
        offset = (trk_id - len(trackers)/2 + 0.5) * width_bar
        values = [metrics[m][trk_id] for m in metrics.keys()]
        ax.bar(x_pos + offset, values, width_bar, label=name, color=color, alpha=0.85, edgecolor='white', linewidth=1.2)
        
        # 添加数值标签
        for i, val in enumerate(values):
            ax.text(i + offset, val + 2, f'{val:.1f}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('指标', fontsize=12, fontweight='bold')
    ax.set_ylabel('分数 (%)', fontsize=12, fontweight='bold')
    ax.set_title('性能摘要', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(metrics.keys()))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    # 保存
    save_path = output_dir / 'all_in_one.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ 已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化跟踪评估结果')
    parser.add_argument('pkl_file', type=str, help='eval_data.pkl 文件路径')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='输出目录 (默认: pkl文件同目录下的 plots/)')
    parser.add_argument('--plots', '-p', nargs='+', 
                        choices=['success', 'prec', 'norm_prec', 'bar', 'per_seq', 'all', 'all_in_one'],
                        default=['all_in_one'],
                        help='要生成的图表类型')
    parser.add_argument('--show', '-s', action='store_true',
                        help='显示图表（而不只是保存）')
    
    args = parser.parse_args()
    
    # 加载数据
    pkl_path = Path(args.pkl_file)
    if not pkl_path.exists():
        print(f"❌ 错误: 文件不存在: {pkl_path}")
        return
    
    print(f"📂 加载文件: {pkl_path}")
    print("=" * 80)
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 文件加载成功\n")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pkl_path.parent / 'plots'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 输出目录: {output_dir}\n")
    
    # 显示基本信息
    trackers = data.get('trackers', [])
    sequences = data.get('sequences', [])
    valid_sequence = data.get('valid_sequence', [])
    
    print(f"📊 数据集信息:")
    print(f"   • 序列数: {len(sequences)}")
    print(f"   • 有效序列: {sum(valid_sequence)}")
    print(f"   • Tracker 数: {len(trackers)}")
    for t in trackers:
        print(f"      - {get_tracker_display_name(t)}")
    print("\n" + "=" * 80 + "\n")
    
    # 生成图表
    plot_types = args.plots
    if 'all' in plot_types:
        plot_types = ['success', 'prec', 'norm_prec', 'bar', 'per_seq']
    
    if 'all_in_one' in plot_types:
        plot_all_in_one(data, output_dir, args.show)
    
    if 'success' in plot_types:
        plot_success_curve(data, output_dir, args.show)
    
    if 'prec' in plot_types:
        plot_precision_curve(data, output_dir, args.show, normalized=False)
    
    if 'norm_prec' in plot_types:
        plot_precision_curve(data, output_dir, args.show, normalized=True)
    
    if 'bar' in plot_types:
        plot_comparison_bar(data, output_dir, args.show)
    
    if 'per_seq' in plot_types:
        plot_per_sequence_performance(data, output_dir, args.show)
    
    print("\n" + "=" * 80)
    print(f"🎉 可视化完成！所有图表已保存至: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
