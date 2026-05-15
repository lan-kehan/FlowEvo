# FlowEvo / Flow AutoTTS

FlowEvo 是一个用于 **flow-matching 采样器 test-time scaling** 的自动搜索库。当前主实验是：

- 模型：Stable Diffusion 3.5 Medium
- 奖励：PickScore
- 任务：在固定 NFE budget 下，让 Codex 迭代改写 controller，寻找比普通 ODE 更好的采样策略

## 核心思路

库里把采样过程包装成一个受限环境，controller 只能调用公开动作：

```text
spawn      创建分支
forward    向前积分一步或多步
preview    解码/打分当前 clean anchor
backward   从某个 anchor 重新加噪，生成局部分支
prune      剪掉弱分支
answer     返回最终结果
```

Workflow 每轮会：

1. 从 `optimal.template.py` 重置 `flow_autotts/controllers/optimal.py`
2. 给 Codex 一个 AutoTTS 风格 prompt、spec、baseline、最近几轮 summary 和 controller snapshot
3. Codex 只改 `optimal.py`
4. harness 在 train prompts 上评估不同 beta/NFE 档
5. 归档 controller、summary、history，进入下一轮

`beta` 控制 compute budget：`0` 接近 10 NFE deterministic ODE，`1` 接近 64 NFE；中间 beta 会分配更多 preview、branch、backward refinement 和 pruning。

## 安装

推荐用 `uv`：

```bash
cd /root/code/FlowEvo
uv sync --group dev --group sd35
```

Codex workflow 还需要 OpenAI Codex CLI 和较新的 Node：

```bash
node --version   # 建议 >= 18
npm install -g @openai/codex
codex exec --help
```

如果本机的 sandbox 不可用，可以在运行 workflow 时加：

```bash
CODEX_EXEC_ARGS="--dangerously-bypass-approvals-and-sandbox"
```

## 数据和模型

默认路径：

```text
SD_3.5_med/                         SD3.5 Medium 本地模型
PickScore_v1/                       PickScore reward model
flow_grpo/dataset/pickscore/train.txt
flow_grpo/dataset/pickscore/test.txt
```

下载模型示例：

```bash
huggingface-cli login
huggingface-cli download stabilityai/stable-diffusion-3.5-medium \
  --local-dir SD_3.5_med
huggingface-cli download yuvalkirstain/PickScore_v1 \
  --local-dir PickScore_v1
```

数据文件是一行一个 prompt。当前仓库已带 `flow_grpo/dataset/pickscore/train.txt` 和 `test.txt`；如果新机器缺失，从 Flow-GRPO 的 `dataset/pickscore` 拷贝到同一路径即可。也可以用环境变量 `FLOW_TTS_DATASET=/path/to/pickscore` 指向自己的数据目录。

## 运行 5 轮搜索

下面命令用 4 张卡并行评估，每轮评估 500 条 train prompts、5 个 beta 档：

```bash
cd /root/code/FlowEvo

RUN_TAG="autotts_prompt_v3_4gpu_r0000_0004_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/flow_autotts/pickscore_sd35/manual_runs

nohup env \
  FLOW_TTS_PROMPT_PROFILE=autotts \
  FLOW_TTS_EVAL_DEVICES="cuda:0,cuda:1,cuda:2,cuda:3" \
  WORKFLOW_RESUME=0 \
  WORKFLOW_ROUNDS=5 \
  WORKFLOW_CONTEXT_HISTORY_ROUNDS=5 \
  FLOW_TTS_SPLIT=train \
  FLOW_TTS_SAMPLE_SIZE=500 \
  FLOW_TTS_SAMPLE_SEED=42 \
  FLOW_TTS_BETAS="0 0.25 0.5 0.75 1.0" \
  FLOW_TTS_BUDGET=64 \
  FLOW_TTS_NUM_STEPS=10 \
  WORKFLOW_HISTORY_DIR="logs/flow_autotts/pickscore_sd35/history_${RUN_TAG}" \
  WORKFLOW_CODEX_LOG_PARENT="/root/code/FlowEvo/logs/flow_autotts/pickscore_sd35/codex_logs_${RUN_TAG}" \
  WORKFLOW_RESULT_DIR="/root/code/FlowEvo/logs/flow_autotts/pickscore_sd35/training_results_${RUN_TAG}" \
  CODEX_EXEC_ARGS="--dangerously-bypass-approvals-and-sandbox" \
  uv run --group sd35 bash flow_autotts/experiments/pickscore_sd35/run_workflow.sh \
  > "logs/flow_autotts/pickscore_sd35/manual_runs/${RUN_TAG}.log" 2>&1 &

echo "${RUN_TAG}"
```

看进度：

```bash
tail -f logs/flow_autotts/pickscore_sd35/manual_runs/${RUN_TAG}.log
nvidia-smi
```

## 输出怎么看

每轮结果在：

```text
logs/flow_autotts/pickscore_sd35/history_${RUN_TAG}/rXXXX_*/proposal_results/summary.json
logs/flow_autotts/pickscore_sd35/history_${RUN_TAG}/rXXXX_*/proposal_results/history.json
logs/flow_autotts/pickscore_sd35/history_${RUN_TAG}/rXXXX_*/flow_autotts/controllers/optimal.py
```

`summary.json` 是给下一轮 Codex 的 compact history，包含：

- 每个 beta 的 reward / NFE / reward_per_nfe
- Pareto frontier
- `action_statistics`：平均 spawn、forward、preview、backward、prune、mean_nfe
- `behavior_summary`：一句话概括该 beta 档 controller 行为

`history.json` 是完整评估日志，包含每个 prompt 的 event log，体积会比较大。

## 快速检查

```bash
python -m py_compile flow_autotts/controllers/optimal.py
python -m pytest tests/test_workflow.py
```

