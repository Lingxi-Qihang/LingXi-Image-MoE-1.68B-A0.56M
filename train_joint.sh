#!/bin/bash
set -e

# ================================================================
# ProMoE  文生图 + 图片编辑 联合训练
# 数据: /data/coding/preprocessed_all_moe (含双任务)
# 配置: configs/joint.yaml (ProMoE_TC_L, 256)
# 512 配置: configs/joint_512.yaml
#   - image_size: 512, total_train_batch_size: 16 (256版=64)
#   - latent_root: .../latents_512 (软链 -> .../latents/target_512)
#   - 无 source/ 目录, 训练时 source_latent 用零填充
#   用 --size 512 自动切换至 512 配置:
#   bash train_joint.sh --size 512 --ckpt ... --steps 2000000
# ================================================================
# 1-step:
#   bash train_joint.sh --steps 200000
#   bash train_joint.sh --ckpt ProMoE_L/ProMoE_TC_L/joint_d4HRU/checkpoints/ckpt_step_1250000.pth --steps 2000000
#   bash train_joint.sh --size 512 --ckpt ProMoE_L/ProMoE_TC_L/joint_d4HRU/checkpoints/ckpt_step_1250000.pth --steps 2000000
# 2-step (分段训练, 学习率/步数可变):
#   bash train_joint.sh --lr 1e-4 5e-5 --steps 150000 50000
# 通用:
#   --size N    --ckpt path  --accum N  --batch_size N  --img_size N  --gpu N
# ================================================================

cd "$(dirname "$0")"

LR=(1e-4)
STEPS=(200000)
CKPT=""
GRAD_ACCUM_STEPS=1
BATCH_SIZE=16
IMG_SIZE=256
GPU_ID=0
SIZE=256

while [ $# -gt 0 ]; do
    case "$1" in
        --lr)     shift; LR=(); while [ $# -gt 0 ] && [[ "$1" != -* ]]; do LR+=("$1"); shift; done ;;
        --steps)  shift; STEPS=(); while [ $# -gt 0 ] && [[ "$1" != -* ]]; do STEPS+=("$1"); shift; done ;;
        --ckpt)   shift; CKPT="$1"; shift ;;
        --accum)  shift; GRAD_ACCUM_STEPS="$1"; shift ;;
        --batch_size) shift; BATCH_SIZE="$1"; shift ;;
        --img_size)  shift; IMG_SIZE="$1"; shift ;;
        --gpu)    shift; GPU_ID="$1"; shift ;;
        --size)   shift; SIZE="$1"; shift ;;
        --help|-h) head -20 "$0"; exit 0 ;;
        *) echo "未知: $1"; exit 1 ;;
    esac
done

MODE=${#LR[@]}
[ ${#STEPS[@]} -eq 1 ] && STEPS=(${STEPS[0]} ${STEPS[0]})

TRAIN_SCRIPT="train_GradientAccumulationSteps.py"
if [ "$SIZE" = "512" ]; then
    BASE_CONFIG="configs/joint_512.yaml"
    [ "$IMG_SIZE" = "256" ] && IMG_SIZE=512
    [ "$BATCH_SIZE" = "16" ] && BATCH_SIZE=8
else
    BASE_CONFIG="configs/joint.yaml"
fi
OUTPUT_DIR="ProMoE_B"
MODEL_NAME="ProMoE_TC_L"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo " 文生图 + 编辑 联合训练 ($MODE-step)  $MODEL_NAME"
echo " GPU=$GPU_ID  BS=$BATCH_SIZE  IMG=${IMG_SIZE}  Accum=$GRAD_ACCUM_STEPS"
echo " LR: ${LR[*]}  Steps: ${STEPS[*]}"
echo " CKPT: ${CKPT:-无}"
echo "============================================================"

get_step() {
    python3 -c "import torch; print(torch.load('$1', map_location='cpu').get('step',0))" 2>/dev/null || echo 0
}

for i in $(seq 0 $((MODE-1))); do
    PHASE="phase$((i+1))"
    PHASE_DIR="$OUTPUT_DIR/$MODEL_NAME/joint_$PHASE"

    CKPT_STEP=0
    CUSTOM_CFG_NAME="joint_$PHASE"
    if [ $i -eq 0 ] && [ -n "$CKPT" ]; then
        CKPT_STEP=$(get_step "$CKPT")
        # 从 checkpoint 路径解析目录结构: <output_dir>/<model_name>/<custom_cfg_name>/checkpoints/ckpt_step_N.pth
        CKPT_DIR=$(dirname "$(dirname "$CKPT")")
        CUSTOM_CFG_NAME=$(basename "$CKPT_DIR")
        MODEL_DIR=$(dirname "$CKPT_DIR")
        OUTPUT_DIR=$(basename "$(dirname "$MODEL_DIR")")
        [ "$(basename "$MODEL_DIR")" != "$MODEL_NAME" ] && echo "Warning: checkpoint model dir != $MODEL_NAME"
        echo ">>> 使用原 checkpoint 目录: $OUTPUT_DIR/$MODEL_NAME/$CUSTOM_CFG_NAME"
    elif [ $i -eq 0 ]; then
        mkdir -p "$PHASE_DIR/checkpoints"
    else
        PREV=$(ls -t "$OUTPUT_DIR/$MODEL_NAME/joint_phase$i/checkpoints"/ckpt_step_*.pth 2>/dev/null | head -1)
        [ -z "$PREV" ] && echo "缺少 phase$i checkpoint" && exit 1
        CKPT_STEP=$(get_step "$PREV")
        mkdir -p "$PHASE_DIR/checkpoints"
        cp "$PREV" "$PHASE_DIR/checkpoints/"
    fi

    # 用 Python 覆写 config 中的关键字段
    TMPCFG=$(mktemp /tmp/joint_XXXXX.yaml)
    python3 -c "
import yaml
with open('$BASE_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['lr'] = ${LR[$i]}
cfg['num_steps'] = ${STEPS[$i]}
cfg['resume_checkpoint_step'] = $CKPT_STEP
cfg['gpu_ids'] = [$GPU_ID]
cfg['total_train_batch_size'] = $BATCH_SIZE
cfg['image_size'] = $IMG_SIZE
cfg['custom_cfg_name'] = '$CUSTOM_CFG_NAME'
with open('$TMPCFG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('$TMPCFG')
    " 2>/dev/null

    echo ">>> $PHASE: lr=${LR[$i]} steps=${STEPS[$i]} ckpt_step=$CKPT_STEP"

    python "$TRAIN_SCRIPT" --config "$TMPCFG" --grad_accum_steps "$GRAD_ACCUM_STEPS" \
        2>&1 | tee -a "$LOG_DIR/joint_$PHASE.log"

    rm -f "$TMPCFG"
done

echo "=== 完成 ==="
