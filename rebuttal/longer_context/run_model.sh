#!/usr/bin/env bash
# 换模型重跑 baseline（复用已建历史，只重跑评测）。
#
# 用法（全部可省，用默认）：
#   bash longer_context/run_model.sh [MODEL_NAME] [API_URL] [EVAL_TURNS]
# 默认：
#   MODEL_NAME = persona_general
#   API_URL    = ${VLLM_URL:-http://localhost:88/v1/chat/completions}  (之前用的本地 vLLM)
#   EVAL_TURNS = 10
#
# 自动选择请求方式：
#   - 模型名含 "doubao" -> 直连豆包 ARK 端点，用写死的 ARK key，不发 vLLM 专有字段。
#   - 其它(qwen / persona_general / 本地) -> 用传入/默认的 vLLM 端点，不需要 token，
#     并附带 top_p + chat_template_kwargs(enable_thinking=false)。
#   如本地 vLLM 需要鉴权，可另外 export ROLE_TOKEN=xxx。

set -u
cd "$(dirname "$0")/.."
ROOT="longer_context"
SETA="$ROOT/outputs/varied_100p_50to100t_seed20260529"
SETB="$ROOT/outputs/varied_10p_200t_seed20260529"

ARK_URL="https://ark.cn-beijing.volces.com/api/v3/chat/completions"
ARK_KEY="${ARK_API_KEY:-}"
DANLU_URL="${VLLM_URL:-http://localhost:88/v1/chat/completions}"
FUXI_URL="https://aigc-api.fuxi.netease.com/v1/chat/completions"
FUXI_KEY="${DYNS_EVAL_API_TOKEN:-}"

MODEL="${1:-persona_general}"
API_URL="${2:-$DANLU_URL}"
EVAL_TURNS="${3:-10}"

# ---- 按模型类型决定端点 / token / extra_body ----
if [[ "$MODEL" == *doubao* || "$MODEL" == *Doubao* ]]; then
  API_URL="$ARK_URL"
  TOKEN_ARGS="--role-requires-token --role-api-token $ARK_KEY"
  PREFLIGHT_TOKEN="$ARK_KEY"
  EXTRA_BODY='{}'                       # 豆包不认 vLLM 的 chat_template_kwargs
  echo "[mode] doubao 直连 ARK"
elif [[ "$MODEL" == qwen-plus* || "$MODEL" == qwen-max* || "$MODEL" == qwen-turbo* \
        || "$MODEL" == gpt* || "$MODEL" == gemini* || "$MODEL" == claude* ]]; then
  API_URL="$FUXI_URL"                   # 伏羲网关上的云模型(如 qwen-plus-character)
  TOKEN_ARGS="--role-requires-token --role-api-token $FUXI_KEY"
  PREFLIGHT_TOKEN="$FUXI_KEY"
  EXTRA_BODY='{}'
  echo "[mode] 伏羲云模型 ($MODEL)"
else
  if [ -n "${ROLE_TOKEN:-}" ]; then
    TOKEN_ARGS="--role-requires-token --role-api-token $ROLE_TOKEN"
    PREFLIGHT_TOKEN="$ROLE_TOKEN"
  else
    TOKEN_ARGS="--no-role-requires-token"
    PREFLIGHT_TOKEN=""
  fi
  EXTRA_BODY='{"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": false}}'
  echo "[mode] 本地 vLLM ($MODEL)"
fi

SLUG="$(echo "$MODEL" | tr -c 'A-Za-z0-9._-' '_')"
OUT="rc_${SLUG}"
log(){ echo -e "\n[$(date '+%H:%M:%S')] ===== $* ====="; }

# ---------- 预检模型服务 ----------
log "预检: $MODEL @ $API_URL"
python3 - "$MODEL" "$API_URL" "$PREFLIGHT_TOKEN" <<'PY' || { echo "!! 模型服务不可用"; exit 1; }
import requests,sys
m,u=sys.argv[1],sys.argv[2]
tok=sys.argv[3] if len(sys.argv)>3 else ''
h={'Authorization':f'Bearer {tok}'} if tok else {}
try:
    r=requests.post(u,json={'model':m,'messages':[{'role':'user','content':'你好'}],'max_tokens':8},headers=h,timeout=15)
    print('->',r.status_code); sys.exit(0 if r.status_code==200 else 1)
except Exception as e:
    print('ERR',e); sys.exit(1)
PY

# ---------- 评测（先删后跑，不 resume） ----------
run_eval(){  # $1=set_dir  $2=标签
  log "评测 $2 (model=$MODEL, turns=$EVAL_TURNS) -> $1/$OUT"
  rm -rf "$1/$OUT"
  python -u "$ROOT/evaluate_history_length.py" \
    --context-dir "$1" --context-file "$1/context_dialogues.jsonl" --output-dir "$1/$OUT" \
    --role-model "$MODEL" --role-api-url "$API_URL" $TOKEN_ARGS --role-extra-body "$EXTRA_BODY" \
    --user-simulator-style proactive --eval-turns "$EVAL_TURNS" --bucket-size 10 --mode all
}
run_eval "$SETA" "100p 50-100轮"
run_eval "$SETB" "10p 200轮"

# ---------- 汇总 ----------
log "汇总 model=$MODEL"
python3 - "$SETA/$OUT" "$SETB/$OUT" "$MODEL" <<'PY'
import json,sys,statistics,collections
A,B,model=sys.argv[1],sys.argv[2],sys.argv[3]
def load(d):
    try: return json.load(open(f"{d}/role_consistency_by_length.json"))
    except FileNotFoundError: return []
rows=load(A)+load(B)
def coarse(h): return '50-74轮' if h<75 else ('75-100轮' if h<=100 else '200轮')
bins=collections.defaultdict(list)
for r in rows: bins[coarse(r['history_turns'])].append(r['score'])
print(f"== {model} baseline 三档汇总 ==")
print("| 历史长度 | N | 均值 | Std | 分布 |"); print("|---|---:|---:|---:|---|")
for b in ('50-74轮','75-100轮','200轮'):
    s=bins.get(b,[])
    if s: print(f"| {b} | {len(s)} | {statistics.mean(s):.2f} | {statistics.pstdev(s):.2f} | {dict(sorted(collections.Counter(s).items()))} |")
allsc=[r['score'] for r in rows]
if allsc: print(f"\n合计 N={len(allsc)} 均值={statistics.mean(allsc):.3f} 5分占比={sum(x==5 for x in allsc)}/{len(allsc)}")
PY
log "完成 -> $SETA/$OUT , $SETB/$OUT"
