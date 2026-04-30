import subprocess
import os
import time


def run_command(cmd):
    print(f"\n{'=' * 20} STARTING NEW TASK {'=' * 20}")
    print(f"Executing: {cmd}")

    # 使用 subprocess.run 可以实时看到控制台输出
    # check=True 表示如果脚本报错退出，Python 也会抛出异常停止，防止无效运行
    try:
        start_time = time.time()
        subprocess.run(cmd, shell=True, check=True)
        end_time = time.time()
        print(f"Task completed in {(end_time - start_time) / 60:.2f} minutes.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running: {cmd}")
        print(f"Error details: {e}")
        # 如果你想在一个实验挂掉后继续下一个，就把 exit(1) 注释掉
        exit(1)


def main_a():
    # 基础公共参数
    base_cmd = "python ST.py --arch resnet50 --src_path ./original_datasets/officehome/Art --tgt_path ./original_datasets/officehome/Real_World --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --num_classes 65"

    # 定义具体的实验变体
    experiments = [
        {"method": "ST", "extra": "", "save_tag": "ST"},
        {"method": "CBST", "extra": "", "save_tag": "CBST"},
        {"method": "CRST", "extra": "--alpha 0.05 --beta 0 --gamma 0 --delta 0", "save_tag": "CRST_LRENT"},
        {"method": "CRST", "extra": "--alpha 0 --beta 1e-3 --gamma 0 --delta 0", "save_tag": "CRST_MRKLD"},
        {"method": "CRST", "extra": "--alpha 0 --beta 0 --gamma 1e-3 --delta 0", "save_tag": "CRST_MRENT"},
        {"method": "CRST", "extra": "--alpha 0 --beta 0 --gamma 0 --delta 1e-3", "save_tag": "CRST_MRL2"},
        {"method": "CRST", "extra": "--alpha 0.05 --beta 1e-3 --gamma 0 --delta 0", "save_tag": "CRST_LRENT_MRKLD"},

    ]

    for exp in experiments:

        save_dir = f"./checkpoints/art_to_realworld_{exp['save_tag']}"
        full_cmd = f"{base_cmd} --method {exp['method']} {exp['extra']} --save_dir {save_dir}"
        # 执行
        run_command(full_cmd)
    print("\n" + "#" * 50)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("#" * 50)

def main_b():
    # 基础公共参数
    base_cmd = "python ST.py --arch resnet50 --src_path ./original_datasets/PACS/photo --tgt_path ./original_datasets/PACS/sketch --apply_aug --num_rounds 50 --epochs_per_round 2 --init_portion 0.1 --portion_step 0.02 --max_portion 0.8 --lr 1e-5 --num_classes 7 --warmup_limit 60.0 --lr_warm 1e-6"

    # 定义具体的实验变体
    experiments = [
        {"method": "ST", "extra": "", "save_tag": "ST_ES"},
        {"method": "CBST", "extra": "", "save_tag": "CBST_ES"},
        {"method": "CRST", "extra": "--alpha 0.05 --beta 0 --gamma 0 --delta 0", "save_tag": "CRST_LRENT_ES"},
        {"method": "CRST", "extra": "--alpha 0 --beta 1e-3 --gamma 0 --delta 0", "save_tag": "CRST_MRKLD_ES"},
        {"method": "CRST", "extra": "--alpha 0 --beta 0 --gamma 1e-3 --delta 0", "save_tag": "CRST_MRENT_ES"},
        {"method": "CRST", "extra": "--alpha 0 --beta 0 --gamma 0 --delta 1e-3", "save_tag": "CRST_MRL2_ES"},
        {"method": "CRST", "extra": "--alpha 0.05 --beta 1e-3 --gamma 0 --delta 0", "save_tag": "CRST_LRENT_MRKLD_ES"},
    ]

    for exp in experiments:

        save_dir = f"./checkpoints/photo_to_sketch_{exp['save_tag']}"
        full_cmd = f"{base_cmd} --method {exp['method']} {exp['extra']} --save_dir {save_dir}"
        # 执行
        run_command(full_cmd)
    print("\n" + "#" * 50)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("#" * 50)


if __name__ == "__main__":
    # main_a()
    main_b()