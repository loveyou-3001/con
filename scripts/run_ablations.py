import subprocess
import argparse
import os

# 优化后的消融实验配置
ABLATION_CONFIGS = {
    "A0": {
        "name": "full_model",
        "desc": "完整模型 (Baseline: NREM + REM + Cosine + LoRA-Soft)",
        "flags": "--use_sleep --alpha 0.5",
    },
    "A1": {
        "name": "wo_cosine_linear",
        "desc": "w/o CosineLinear (替换为普通 nn.Linear)",
        "flags": "--use_sleep --alpha 0.5 --no_cosine",
    },
    "A2": {
        "name": "wo_nrem",
        "desc": "w/o NREM (alpha=0, 不执行突触物理缩减)",
        "flags": "--use_sleep --alpha 0.0", 
    },
    "A3": {
        "name": "wo_rem",
        "desc": "w/o REM (跳过梦境边界修复，直接测试无回放极限)",
        "flags": "--use_sleep --alpha 0.5 --no_rem",
    },
    "A4": {
        "name": "wo_lora_soft",
        "desc": "w/o LoRA Soft Immunity (完全冻结旧任务 LoRA, lora_alpha=0)",
        "flags": "--use_sleep --alpha 0.5 --lora_alpha 0.0",
    },
    "A5": {
        "name": "wo_prototype_replay",
        "desc": "w/o Prototype Replay (关闭 Wake 阶段的原型回放)",
        "flags": "--use_sleep --alpha 0.5 --proto_lambda 0.0",
    },
    "A6": {
        "name": "wo_sleep",
        "desc": "w/o Sleep (关闭整个睡眠机制，退化为传统微调)",
        "flags": "", # 不传 --use_sleep
    },
}

def run_single_ablation(ablation_id, data_root="data/clinc150", epochs=10):
    """运行单个消融实验"""
    config = ABLATION_CONFIGS.get(ablation_id)
    if not config:
        print(f"❌ 未知的消融实验ID: {ablation_id}")
        return False
    
    exp_name = f"ablation_{config['name']}"
    
    print("\n" + "=" * 60)
    print(f"🔬 正在执行消融实验 {ablation_id}: {config['desc']}")
    print(f"📁 存档目录: output/{exp_name}")
    print("=" * 60)
    
    # 组合命令
    cmd = f"python main.py --exp_name {exp_name} --data_root {data_root} --epochs {epochs} {config['flags']}"
    print(f"🚀 终端指令: {cmd}\n")
    
    # 执行命令并实时输出日志
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"\n✅ {ablation_id} 实验执行完毕！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {ablation_id} 实验运行崩溃，请检查 main.py 参数支持！错误码: {e.returncode}")
        return False

def run_all_ablations(data_root="data/clinc150", epochs=10):
    """批量连续运行所有实验"""
    results = {}
    print(f"🌟 准备连续运行 {len(ABLATION_CONFIGS)} 组消融实验...")
    
    for ablation_id in ABLATION_CONFIGS:
        success = run_single_ablation(ablation_id, data_root, epochs)
        results[ablation_id] = "成功" if success else "失败"
    
    # 打印最终计分板
    print("\n" + "🏆 最终消融实验运行汇总")
    print("=" * 60)
    for aid, status in results.items():
        config = ABLATION_CONFIGS[aid]
        icon = "🟢" if status == "成功" else "🔴"
        print(f"{icon} {aid}: {config['desc']} -> {status}")

def main():
    parser = argparse.ArgumentParser(description="Sleep-HOP 自动化消融引擎")
    parser.add_argument("--ablation", type=str, help="指定消融实验ID (如 A0)")
    parser.add_argument("--all", action="store_true", help="通宵模式：一键运行所有消融实验")
    parser.add_argument("--data_root", type=str, default="data/clinc150")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--list", action="store_true", help="查看菜单：列出所有可用的消融实验")
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 实验清单:\n")
        for aid, config in ABLATION_CONFIGS.items():
            print(f"  [{aid}] {config['desc']}")
            print(f"       Flags: {config['flags']}\n")
        return
    
    if args.all:
        run_all_ablations(args.data_root, args.epochs)
    elif args.ablation:
        run_single_ablation(args.ablation, args.data_root, args.epochs)
    else:
        print("💡 提示: 请输入运行指令。例如: python scripts/run_ablations.py --all")

if __name__ == "__main__":
    main()