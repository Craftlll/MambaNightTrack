#!/usr/bin/env python
"""
灵活的跟踪结果分析脚本

用法:
    # 默认: 分析 MambaNUT-300ep 在 NAT2021 上的结果
    python tracking/analysis_results.py
    
    # 指定数据集
    python tracking/analysis_results.py -d nat2021l
    python tracking/analysis_results.py -d uavdark135
    
    # 指定单个模型
    python tracking/analysis_results.py -c mambar_small_patch16_224 -n MambaNUT
    
    # 对比多个模型
    python tracking/analysis_results.py --configs config1 config2 --names Name1 Name2
    
    # 使用预设
    python tracking/analysis_results.py -p quick   # 快速验证 (20 epoch)
    python tracking/analysis_results.py -p full    # 完整实验 (100 epoch)
    python tracking/analysis_results.py -p baseline # 原始 baseline (300 epoch)
    python tracking/analysis_results.py -p all     # 全部模型对比
    
    # 附加选项
    python tracking/analysis_results.py --plot           # 生成图表
    python tracking/analysis_results.py --per_sequence   # 每序列详情
    python tracking/analysis_results.py --save_plot out.png # 保存图表
"""

import _init_paths
import argparse
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


def parse_args():
    parser = argparse.ArgumentParser(description='MambaNightTrack 跟踪结果分析工具')
    
    # 数据集选择
    parser.add_argument('--dataset', '-d', type=str, default='nat2021',
                        help='测试数据集 (默认: nat2021). 可选: nat2021, nat2021l, uavdark135, lasot, otb, got10k_test, trackingnet')
    
    # 单模型模式
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='单个模型配置名称 (如: mambar_small_patch16_224)')
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='单个模型显示名称')
    
    # 多模型对比模式
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='多个模型配置名称列表')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                        help='多个模型显示名称列表')
    
    # 预设模式
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['quick', 'full', 'baseline', 'all'],
                        help='预设: quick(20ep), full(100ep), baseline(300ep), all(全部对比)')
    
    # 输出选项
    parser.add_argument('--plot', action='store_true', help='生成可视化图表')
    parser.add_argument('--per_sequence', '-s', action='store_true', help='显示每序列详细结果')
    parser.add_argument('--save_plot', type=str, default=None, help='保存图表到指定路径')
    
    return parser.parse_args()


# 预设配置
PRESETS = {
    'quick': {
        'configs': ['mambar_small_patch16_224_baseline_quick', 'mambar_small_patch16_224_lyt_quick'],
        'names': ['Baseline-20ep', 'LYT-20ep'],
        'desc': '快速验证 (20 epoch, 15000 samples/epoch)'
    },
    'full': {
        'configs': ['mambar_small_patch16_224_baseline_100', 'mambar_small_patch16_224_lyt'],
        'names': ['Baseline-100ep', 'LYT-100ep'],
        'desc': '完整实验 (100 epoch)'
    },
    'baseline': {
        'configs': ['mambar_small_patch16_224'],
        'names': ['MambaNUT-300ep'],
        'desc': '原始 Baseline (300 epoch, 7个数据集)'
    },
    'all': {
        'configs': ['mambar_small_patch16_224', 'mambar_small_patch16_224_baseline_100', 'mambar_small_patch16_224_lyt'],
        'names': ['Original-300ep', 'Baseline-100ep', 'LYT-100ep'],
        'desc': '全部模型对比'
    }
}

# 数据集信息
DATASET_INFO = {
    'nat2021': {'sequences': 180, 'frames': '~100k', 'time': '~10-15 min'},
    'nat2021l': {'sequences': 23, 'frames': '~54k', 'time': '~3-5 min'},
    'uavdark135': {'sequences': 135, 'frames': '~100k', 'time': '~10 min'},
    'lasot': {'sequences': 280, 'frames': '~700k', 'time': '~30 min'},
    'otb': {'sequences': 100, 'frames': '~50k', 'time': '~5 min'},
    'got10k_test': {'sequences': 180, 'frames': '~100k', 'time': '~10 min'},
    'trackingnet': {'sequences': 511, 'frames': '~500k', 'time': '~25 min'},
}


def main():
    args = parse_args()
    
    # 确定配置
    configs, names, desc = [], [], ""
    
    if args.preset:
        # 使用预设配置
        preset = PRESETS[args.preset]
        configs, names, desc = preset['configs'], preset['names'], preset['desc']
    elif args.configs:
        # 多模型对比模式
        configs = args.configs
        names = args.names if args.names else args.configs
        desc = "自定义对比"
    elif args.config:
        # 单模型模式
        configs = [args.config]
        names = [args.name] if args.name else [args.config]
        desc = f"单模型: {args.config}"
    else:
        # 默认: 原始 baseline
        preset = PRESETS['baseline']
        configs, names, desc = preset['configs'], preset['names'], preset['desc']
    
    # 检查参数
    if len(configs) != len(names):
        print(f"❌ 错误: configs ({len(configs)}) 和 names ({len(names)}) 数量不匹配!")
        print(f"   configs: {configs}")
        print(f"   names: {names}")
        return
    
    dataset_name = args.dataset.lower()
    
    # 构建 tracker 列表
    trackers = []
    for cfg, name in zip(configs, names):
        trackers.extend(trackerlist(
            name='mambanut', 
            parameter_name=cfg, 
            dataset_name=dataset_name, 
            run_ids=None, 
            display_name=name
        ))
    
    # 加载数据集
    try:
        dataset = get_dataset(dataset_name)
    except Exception as e:
        print(f"❌ 加载数据集 '{dataset_name}' 失败: {e}")
        print(f"   可用数据集: {list(DATASET_INFO.keys())}")
        return
    
    # 数据集信息
    ds_info = DATASET_INFO.get(dataset_name, {'sequences': len(dataset), 'frames': '未知', 'time': '未知'})
    
    # 打印表头
    print("\n" + "=" * 80)
    print(f"🔬 MambaNightTrack 结果分析")
    print("=" * 80)
    print(f"📋 实验: {desc}")
    print(f"📊 数据集: {dataset_name.upper()}")
    print(f"   • 序列数: {ds_info['sequences']}")
    print(f"   • 总帧数: {ds_info['frames']}")
    print(f"   • 预计时间: {ds_info['time']}")
    print(f"🎯 模型数: {len(configs)}")
    for cfg, name in zip(configs, names):
        print(f"   • {name}: {cfg}")
    print("=" * 80 + "\n")
    
    # 打印结果
    print_results(trackers, dataset, dataset_name, merge_results=True, 
                  plot_types=('success', 'norm_prec', 'prec'))
    
    # 每序列详情
    if args.per_sequence:
        print("\n" + "=" * 80)
        print("📋 每序列详细结果:")
        print("=" * 80)
        print_per_sequence_results(trackers, dataset, dataset_name)
    
    # 结果解读
    if len(configs) == 2:
        print("\n" + "-" * 80)
        print("📌 结果解读:")
        print(f"   • {names[1]} > {names[0]} → {names[1]} 更优 ✅")
        print(f"   • {names[1]} ≈ {names[0]} → 两者相当")
        print(f"   • {names[1]} < {names[0]} → {names[0]} 更优")
        print("-" * 80)
    
    # 生成图表
    if args.plot or args.save_plot:
        print("\n📊 生成可视化图表...")
        plot_results(trackers, dataset, dataset_name, merge_results=True,
                     plot_types=('success', 'norm_prec', 'prec'),
                     skip_missing_seq=False, force_evaluation=False)
        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f"   图表已保存: {args.save_plot}")
        if args.plot:
            plt.show()
    
    print("\n✅ 分析完成!")


if __name__ == '__main__':
    main()
