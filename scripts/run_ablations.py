"""
Sleep-HOP 消融实验脚本
========================
运行方式: python scripts/run_ablations.py --ablation A1
或批量运行: python scripts/run_ablations.py --all
"""

import subprocess
import argparse
import os
import json
from datetime import datetime

# 消融实验配置
ABLATION_CONFIGS = {
    "A0": {
        "name": "full_model",
        "desc": "完整模型 (Baseline)",
        "flags": "--use_sleep --alpha 0.5",
        "code_changes": None  # 无需修改
    },
    "A1": {
        "name": "wo_cosine_linear",
        "desc": "w/o CosineLinear (使用 nn.Linear)",
        "flags": "--use_sleep --alpha 0.5",
        "code_changes": "model.py: ABLATION_NO_COSINE = True"
    },
    "A2": {
        "name": "wo_nrem",
        "desc": "w/o NREM (跳过突触缩减)",
        "flags": "--use_sleep --alpha 0.0",  # alpha=0 禁用NREM
        "code_changes": None
    },
    "A3": {
        "name": "wo_rem",
        "desc": "w/o REM (跳过梦境修复)",
        "flags": "--use_sleep --alpha 0.5 --no_rem",
        "code_changes": "需要添加 --no_rem 参数支持"
    },
    "A4": {
        "name": "wo_lora_soft",
        "desc": "w/o LoRA Soft Immunity (100% 冻结)",
        "flags": "--use_sleep --alpha 0.5 --lora_alpha 0.0",
        "code_changes": "需要添加 --lora_alpha 参数支持"
    },
    "A5": {
        "name": "wo_prototype_replay",
        "desc": "w/o Prototype Replay (取消 Wake 回放)",
        "flags": "--use_sleep --alpha 0.5 --proto_lambda 0.0",
        "code_changes": None  # 已有参数支持
    },
    "A6": {
        "name": "wo_sleep",
        "desc": "w/o Sleep (关闭整个睡眠机制)",
        "flags": "",  # 不传 --use_sleep
        "code_changes": None
    },
}

def run_single_ablation(ablation_id, data_root="data/clinc150", epochs=10):
    """运行单个消融实验"""
    config = ABLATION_CONFIGS.get(ablation_id)
    if not config:
        print(f"❌ 未知的消融实验ID: {ablation_id}")
        return
    
    exp_name = f"ablation_{config['name']}"
    
    print("=" * 60)
    print(f"🔬 消融实验 {ablation_id}: {config['desc']}")
    print(f"📁 实验名称: {exp_name}")
    print(f"⚙️  参数: {config['flags']}")
    if config['code_changes']:
        print(f"⚠️  需要代码修改: {config['code_changes']}")
    print("=" * 60)
    
    cmd = f"python main.py --exp_name {exp_name} --data_root {data_root} --epochs {epochs} {config['flags']}"
    print(f"🚀 执行命令: {cmd}\n")
    
    # 执行命令
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    return result.returncode == 0

def run_all_ablations(data_root="data/clinc150", epochs=10):
    """批量运行所有消融实验"""
    results = {}
    
    for ablation_id in ABLATION_CONFIGS:
        print(f"\n{'#' * 60}")
        print(f"# 开始消融实验 {ablation_id}")
        print(f"{'#' * 60}\n")
        
        success = run_single_ablation(ablation_id, data_root, epochs)
        results[ablation_id] = "成功" if success else "失败"
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("📊 消融实验汇总")
    print("=" * 60)
    for aid, status in results.items():
        config = ABLATION_CONFIGS[aid]
        print(f"  {aid}: {config['desc']} - {status}")

def main():
    parser = argparse.ArgumentParser(description="Sleep-HOP 消融实验")
    parser.add_argument("--ablation", type=str, help="指定消融实验ID (A0-A6)")
    parser.add_argument("--all", action="store_true", help="运行所有消融实验")
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--list", action="store_true", help="列出所有消融实验")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 可用的消融实验:\n")
        for aid, config in ABLATION_CONFIGS.items():
            print(f"  {aid}: {config['desc']}")
            if config['code_changes']:
                print(f"      ⚠️ {config['code_changes']}")
        return
    
    if args.all:
        run_all_ablations(args.data_root, args.epochs)
    elif args.ablation:
        run_single_ablation(args.ablation, args.data_root, args.epochs)
    else:
        print("请指定 --ablation <ID> 或 --all 或 --list")

if __name__ == "__main__":
    main()
