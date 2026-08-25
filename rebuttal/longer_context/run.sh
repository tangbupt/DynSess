#!/usr/bin/env bash
# 全量 baseline：100 personas × 随机 50-100 轮历史 + 10 personas × 200 轮（极端参照）。
# 只跑 baseline(无破坏)，proactive 用户 + 10 轮续写，只评 role_consistency。
#
# 复用已建历史（40 条 50-100 + 10 条 200），只补建其余 ~60 条。评测(step3)全新跑、不 resume。
# 依赖：本地 vLLM persona_general 在线；doubao/gemini key 已写死。
# 用法： bash longer_context/run.sh

set -u
cd "$(dirname "$0")/.."
ROOT="longer_context"
SETA="$ROOT/outputs/varied_100p_50to100t_seed20260529"
SETB="$ROOT/outputs/varied_10p_200t_seed20260529"
OLD40="$ROOT/outputs/varied_40p_50to100t_seed20260529"
EVAL_TURNS=10

log(){ echo -e "\n[$(date '+%H:%M:%S')] ===== $* ====="; }

# ---------- 0. 预检 ----------
log "预检：本地 vLLM persona_general"
python3 - <<'PY' || { echo "!! 本地服务不可用，请先确认 persona_general 在线"; exit 1; }
import requests,sys
try:
    r=requests.post('${VLLM_URL:-http://localhost:88/v1/chat/completions}',
                    json={'model':'persona_general','messages':[{'role':'user','content':'hi'}],'max_tokens':5},timeout=10)
    print('persona_general ->', r.status_code); sys.exit(0 if r.status_code==200 else 1)
except Exception as e:
    print('ERR', e); sys.exit(1)
PY

# ---------- 1. 构建历史（复用旧的 40 条，补建其余） ----------
log "STEP1 复用已建的 40 条 50-100 轮历史，其余用豆包补建"
mkdir -p "$SETA/records"
cp -n "$OLD40/records/"*.json "$SETA/records/" 2>/dev/null || true
python -u "$ROOT/build_varied_contexts.py"

# ---------- 2. 评测：baseline（先删后跑，不 resume） ----------
run_eval(){   # $1=set_dir $2=标签
  log "STEP2 评测 baseline: $2"
  rm -rf "$1/rc_proactive_baseline"
  python -u "$ROOT/evaluate_history_length.py" \
    --context-dir "$1" --context-file "$1/context_dialogues.jsonl" --output-dir "$1/rc_proactive_baseline" \
    --user-simulator-style proactive --eval-turns "$EVAL_TURNS" --bucket-size 10 --mode all
}
run_eval "$SETA" "100p 50-100轮"
run_eval "$SETB" "10p 200轮"

# ---------- 3. 汇总（默认10轮桶 + 50-74/75-100/200 三档） ----------
log "STEP3 汇总"
python3 - "$SETA" "$SETB" <<'PY'
import json,sys,statistics,collections
SA,SB=sys.argv[1],sys.argv[2]
def load(b):
    try: return json.load(open(f"{b}/rc_proactive_baseline/role_consistency_by_length.json"))
    except FileNotFoundError: return []
rows=load(SA)+load(SB)
def coarse(h): return '50-74轮' if h<75 else ('75-100轮' if h<=100 else '200轮')
bins=collections.defaultdict(list)
for r in rows: bins[coarse(r['history_turns'])].append(r['score'])
print("== baseline 三档汇总 ==")
print("| 历史长度 | N | 均值 | Std | 分布 |"); print("|---|---:|---:|---:|---|")
for b in ('50-74轮','75-100轮','200轮'):
    s=bins.get(b,[])
    if s: print(f"| {b} | {len(s)} | {statistics.mean(s):.2f} | {statistics.pstdev(s):.2f} | {dict(sorted(collections.Counter(s).items()))} |")
allsc=[r['score'] for r in rows]
print(f"\n合计 N={len(allsc)} 均值={statistics.mean(allsc):.3f} 5分占比={sum(x==5 for x in allsc)}/{len(allsc)}")
print("10轮细分见各 rc_proactive_baseline/summary_by_length.md")
PY
log "全部完成"
