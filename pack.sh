#!/bin/bash
set -e

# =========================
# 手动指定要打包的日志文件夹
LOG_TO_PACKAGE="logs/rsl_rl/quadcopter_direct/2026-03-29_21-18-57"
# =========================

# 打包输出目录
TMP_DIR=pack

# 清理旧的
rm -rf $TMP_DIR
mkdir -p $TMP_DIR

# -------------------------
# 1. 复制代码文件到 pack/ 根目录
cp src/third_parties/rsl_rl_local/rsl_rl/algorithms/ppo.py $TMP_DIR/
cp src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py $TMP_DIR/
cp scripts/rsl_rl/play_race.py $TMP_DIR/

# -------------------------
# 2. 复制日志文件夹，保持完整目录结构
if [ ! -d "$LOG_TO_PACKAGE" ]; then
    echo "❌ 指定的日志文件夹不存在: $LOG_TO_PACKAGE"
    exit 1
fi

# 日志文件夹名
LOG_NAME=$(basename "$LOG_TO_PACKAGE")
DST_LOG=$TMP_DIR/$LOG_NAME

# 使用 rsync-like 方式复制所有内容
mkdir -p "$DST_LOG"

# 复制所有文件和子文件夹
cp -r "$LOG_TO_PACKAGE"/. "$DST_LOG"/

# 删除所有除了 best_model.pt 的 .pt 文件
find "$DST_LOG" -type f -name "*.pt" ! -name "best_model.pt" -delete

echo "✅ 打包完成，生成目录: $TMP_DIR"