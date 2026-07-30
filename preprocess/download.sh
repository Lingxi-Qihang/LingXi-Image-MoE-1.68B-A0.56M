#!/bin/sh

# ================= Config =================
REPO_ID="hf/ma-xu-fine-t2i"
FILE_PREFIX="synthetic_original_prompt_square_resolution/train-"
END=1310
FILE_SUFFIX=".tar"
LOCAL_DIR="/data/coding/"
START=0
MAX_PROCS=4   # 并发下载数，可按需修改
# ==========================================

mkdir -p "$LOCAL_DIR"

echo "[INFO] Generating file list..."
# 生成所有需要下载的文件名列表（跳过已存在的）
file_list=""
i=$START
while [ $i -le $END ]; do
    NUM=$(printf "%06d" $i)
    FILENAME="${FILE_PREFIX}${NUM}${FILE_SUFFIX}"
    FILEPATH="${LOCAL_DIR}/${FILENAME}"
    if [ ! -f "$FILEPATH" ]; then
        file_list="$file_list
$FILENAME"
    else
        echo "[SKIP] $FILENAME"
    fi
    i=$((i + 1))
done

count=$(echo "$file_list" | sed '/^$/d' | wc -l)
echo "[INFO] Total files to download: $count"
echo "[INFO] Max parallel downloads: $MAX_PROCS"
echo

# 使用 xargs 进行并发下载
echo "$file_list" | sed '/^$/d' | xargs -P "$MAX_PROCS" -I {} sh -c "
    echo '[*] Downloading: {}'
    modelscope download --dataset \"$REPO_ID\" \"{}\" --local_dir \"$LOCAL_DIR\"
    if [ \$? -eq 0 ]; then
        echo '[OK] {} downloaded'
    else
        echo '[FAIL] {}'
    fi
"

echo
echo "[SUCCESS] All downloads completed!"