#!/usr/bin/env bash
# 一键运行 DynSess 评测流水线：对话生成 → 格式转换 → rubric 评判
#
# 用法：bash run_eval.sh
# 配置：复制 .env.example 为 .env 并填入凭证，或直接 export 环境变量。
#       关键变量：ARK_API_KEY（用户模拟器）、EVAL_API_URL / DYNS_EVAL_API_TOKEN（评判端点）、
#                 LOCAL_VLLM_URL（本地角色扮演模型，ASSISTANT_MODEL=local 时）。
set -euo pipefail

# 切到仓库根目录（本脚本所在目录），路径不依赖当前工作目录
cd "$(dirname "$(readlink -f "$0")")"

# 可选：加载 .env
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
    echo "[run_eval] loaded .env"
fi

echo "[run_eval] starting DynSess pipeline (ASSISTANT_MODEL=${ASSISTANT_MODEL:-local}, GENERATE_MODE=${GENERATE_MODE:-continue})"
python eval/run_dynsess_eval.py
