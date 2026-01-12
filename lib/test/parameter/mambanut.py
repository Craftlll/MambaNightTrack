from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.mambanut.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/mambanut/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path - 支持自定义保存目录
    # 优先级: 1. 环境变量 CHECKPOINT_DIR  2. 配置名称匹配的目录  3. 默认 save_dir
    checkpoint_dir = os.environ.get('CHECKPOINT_DIR', None)
    
    if checkpoint_dir is None:
        # 检查是否存在配置名称对应的特定目录
        if 'baseline_quick' in yaml_name:
            custom_dir = os.path.join(prj_dir, 'output_quick_baseline')
            if os.path.exists(custom_dir):
                checkpoint_dir = custom_dir
        elif 'lyt_quick' in yaml_name:
            custom_dir = os.path.join(prj_dir, 'output_quick_lyt')
            if os.path.exists(custom_dir):
                checkpoint_dir = custom_dir
    
    if checkpoint_dir is None:
        checkpoint_dir = save_dir
    
    params.checkpoint = os.path.join(checkpoint_dir, "checkpoints/train/mambanut/%s/MambaNUT_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))
    
    print(f"Checkpoint path: {params.checkpoint}")

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
