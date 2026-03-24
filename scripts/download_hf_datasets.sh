#!/bin/bash

# 加速网络
if [ -f /etc/network_turbo ]; then
    echo "Enabling network turbo..."
    source /etc/network_turbo
fi

# 关键：开启 Rust 并行下载驱动以提高效率
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://huggingface.co

# 设置错误处理：如果任何命令失败，脚本将立即退出
set -e

# 创建目标目录
mkdir -p ./Data/VLGuard

echo "Starting download for ys-zong/VLGuard..."
# 在最近版本的 hf 中，指定 --local-dir 会自动将文件下载到本地文件夹。
uv run hf download \
    --repo-type dataset ys-zong/VLGuard \
    --local-dir ./Data/VLGuard

echo "All downloads completed."
