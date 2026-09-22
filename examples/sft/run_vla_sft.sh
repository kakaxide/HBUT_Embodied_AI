#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"
export PYTHONWARNINGS="ignore::UserWarning"
#=============================================改的
export COLORFUL_DISABLE=1   #关闭 colorful 着色
export OTEL_SDK_DISABLED=true   # 屏蔽遥测刷屏
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
#==============================================
#==================== 选择用哪张卡训练 ====================
# 想用哪张卡，就把 GPU_IDS 改成哪张卡（物理卡号，从 0 开始，和 ixsmi/nvidia-smi 的编号一致）。
#   单卡：GPU_IDS=3
#   多卡：GPU_IDS="2,3" 或 "2-3"（必须是连续的卡，中间不能跳号）
# 也可以在命令行第 2 个参数传入：bash run_vla_sft.sh libero_sft_openpi 3
# 说明：这里故意不设置 CUDA_VISIBLE_DEVICES，让 Ray 能看到全部显卡，
#       再由 RLinf 按 component_placement 精确地把训练进程绑到指定的卡上。
#       一旦在这里设置 CUDA_VISIBLE_DEVICES，卡号会被重新编号为 0 开头，placement 就对不上了。
GPU_IDS="${2:-${GPU_IDS:-0}}"
export RLINF_GPU_RANKS="${GPU_IDS}"
echo "[run_vla_sft] 训练使用的物理 GPU: ${GPU_IDS}  (component_placement=${RLINF_GPU_RANKS})"
#=========================================================
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}