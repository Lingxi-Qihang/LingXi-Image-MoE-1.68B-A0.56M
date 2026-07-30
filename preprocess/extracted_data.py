import os
import sys
import tarfile
import time
import shutil
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings("ignore", category=DeprecationWarning)
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul')
    except:
        pass

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] 缺少 tqdm 库，请先运行: pip install tqdm")
    sys.exit(1)

# ================= 配置区域 =================
TAR_DIR = r"/data/coding/synthetic_enhanced_prompt_square_resolution"
EXTRACT_DIR = r"/data/coding/synthetic_enhanced_prompt_square_resolution_extracted_data"
DELETE_AFTER_EXTRACT = True
MAX_WORKERS = 4  # 并行解压进程数，可根据CPU/磁盘情况调整
# ===========================================

def get_disk_usage(path):
    total, used, free = shutil.disk_usage(path)
    return {'total': total / (1024**3), 'used': used / (1024**3), 'free': free / (1024**3)}

def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    if not abs_directory.endswith(os.sep):
        abs_directory += os.sep
    return abs_target.startswith(abs_directory)

def extract_single_tar(args):
    """单个 tar 文件的解压任务，供 Pool.map 使用"""
    tar_path, extract_dir, delete_after = args
    try:
        with tarfile.open(tar_path, 'r:') as tar:
            for member in tar.getmembers():
                member_path = os.path.join(extract_dir, member.name)
                if not is_within_directory(extract_dir, member_path):
                    raise Exception(f"Path Traversal Detected: {member.name}")
            if sys.version_info >= (3, 12):
                tar.extractall(path=extract_dir, filter='data')
            else:
                tar.extractall(path=extract_dir)

        # 解压成功后删除 tar 文件
        if delete_after:
            try:
                os.remove(tar_path)
            except Exception as e:
                # 删除失败不影响整体成功状态
                pass
        return True, tar_path, None
    except Exception as e:
        return False, tar_path, str(e)

def main():
    tar_dir = Path(TAR_DIR)
    extract_dir = Path(EXTRACT_DIR)
    extract_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(list(tar_dir.glob("train-*.tar")))
    total_files = len(tar_files)

    if total_files == 0:
        print(f"[ERROR] 在 {tar_dir} 中未找到 train-*.tar 文件")
        return

    print(f"[INFO] 找到 {total_files} 个 tar 文件")
    print(f"[INFO] 解压目录: {extract_dir}")
    print(f"[INFO] 解压后删除 tar: {DELETE_AFTER_EXTRACT}")
    print(f"[INFO] 并行进程数: {MAX_WORKERS}")

    disk = get_disk_usage(str(extract_dir))
    print(f"[INFO] 磁盘剩余空间: {disk['free']:.1f} GB")
    if disk['free'] < 500:
        print(f"[WARNING] 磁盘空间不足！建议至少保留 500GB 剩余空间")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return

    # 构建任务列表：每个任务是 (tar_path, extract_dir, DELETE_AFTER_EXTRACT)
    tasks = [(str(t), str(extract_dir), DELETE_AFTER_EXTRACT) for t in tar_files]

    print("\n" + "=" * 60)
    print("开始多进程解压...")
    print("=" * 60 + "\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    # 使用进程池
    with Pool(processes=MAX_WORKERS) as pool:
        # 使用 imap_unordered 实时返回结果，配合 tqdm 显示进度
        for success, tar_path, err in tqdm(
            pool.imap_unordered(extract_single_tar, tasks),
            total=total_files,
            desc="解压进度",
            unit="file"
        ):
            tar_name = os.path.basename(tar_path)
            if success:
                success_count += 1
                tqdm.write(f"[OK] {tar_name}")
            else:
                fail_count += 1
                tqdm.write(f"[FAIL] {tar_name}: {err}")

    elapsed = time.time() - start_time
    disk_after = get_disk_usage(str(extract_dir))

    print("\n" + "=" * 60)
    print("解压完成！")
    print("=" * 60)
    print(f"[OK] 成功: {success_count}/{total_files}")
    print(f"[FAIL] 失败: {fail_count}/{total_files}")
    print(f"[TIME] 耗时: {elapsed / 60:.1f} 分钟")
    print(f"[DISK] 当前磁盘剩余: {disk_after['free']:.1f} GB")
    print("=" * 60)

if __name__ == "__main__":
    main()