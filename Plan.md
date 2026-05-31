# MinivLLM 从零手搓计划

**硬件**：RTX 2060 6GB，`tp_size=1` 跑通，代码支持多卡扩展
**目标模型**：Qwen3-0.6B（~1.2GB 权重）
**总代码量**：约 800-1000 行 Python
**预计时间**：18-22 天

---

## 项目结构

```
minivllm/
  __init__.py          # 导出 LLM, SamplingParams
  config.py            # Config 数据类 + AutoConfig 自动读模型参数
  sampling_params.py   # temperature, max_tokens, ignore_eos（禁止 greedy）
  llm.py               # LLM 公共 API（LLMEngine 薄子类）
  engine/
    llm_engine.py      # 主入口，mp.spawn TP worker
    scheduler.py       # waiting/running 队列调度
    model_runner.py    # 模型初始化 + KV cache + CUDA graph
    sequence.py        # 序列状态追踪
    block_manager.py   # 分页 KV cache + 前缀缓存
  models/
    qwen3.py           # Qwen3ForCausalLM + packed_modules_mapping
  layers/
    attention.py       # flash_attn 调用 + Triton KV store 核
    linear.py          # ColumnParallel / RowParallel / QKVParallel / MergedColumnParallel + weight_loader
    embed_head.py      # VocabParallelEmbedding + ParallelLMHead
    layernorm.py       # RMSNorm + fused residual (@torch.compile)
    rotary_embedding.py # RotaryEmbedding + lru_cache
    activation.py      # SiluAndMul
    sampler.py         # @torch.compile 温度采样
  utils/
    context.py         # 全局 Context 单例（prefill/decode 状态跨层共享）
    loader.py          # safetensors 权重加载 + packed_modules_mapping 驱动
```

**依赖**（pyproject.toml）：
- Python >=3.10, <3.13
- torch >=2.4, triton >=3.0, transformers >=4.51, flash-attn, xxhash, safetensors

---

## 核心流程

```
LLM.generate() → LLMEngine.step() → Scheduler.schedule() → ModelRunner.run() → Scheduler.postprocess()
```

---

## 各 Phase 详细计划

### Phase 0 — 项目骨架 (1天)

- 创建上述目录和空文件
- 编写 pyproject.toml（依赖声明）
- Config 数据类框架（字段定义，AutoConfig 预留）
- SamplingParams 数据类框架
- `pip install -e .` 验证安装

**交付**：可安装的空骨架

---

### Phase 1 — 单序列推理 + 张量并行 (5-6天)

**目标**：一条 prompt 跑通，同时理解 TP 分片机制

#### 1.1 Config (0.5天)

```python
@dataclass(slots=True)
class Config:
    model: str                    # 本地模型目录路径（必须）
    tensor_parallel_size: int = 1 # TP 并行数
    max_model_len: int = 8192
    kvcache_block_size: int = 16  # 小显存用小 block
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    max_num_seqs: int = 32        # 6GB 显存限制
    max_num_batched_tokens: int = 512
    # 以下由 AutoConfig 自动填充：
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    vocab_size: int = 0
    dtype: str = "auto"
```

- `__post_init__`：调用 `AutoConfig.from_pretrained(model)` 自动填充模型维度
- 断言 `model` 是目录路径、`kvcache_block_size % 256 的某个因子 == 0`、`tensor_parallel_size` 在 1-8
- 计算 `num_kv_heads_per_rank = num_key_value_heads // tensor_parallel_size`

#### 1.2 Linear 层体系 — TP 核心 (1.5天)

这是张量并行的核心学习内容，需逐层理解：

**LinearBase**：
- 持有 `weight` 参数（in_features × out_features）
- 持有 `weight_loader` 属性（函数，决定如何将 safetensors 权重放入参数）
- 持有 `tp_dim`（0 = 行切分 = input_dim 切分，1 = 列切分 = output_dim 切分）
- 持有 `tp_rank`、`tp_size`

**ColumnParallelLinear**（输出维度分片）：
- `weight.shape = (out_features // tp_size, in_features)`
- 每个 rank 只存 1/tp_size 的输出行
- forward：`F.linear(input, weight)` → 输出是 rank 的分片
- 无 bias（推理不需要）
- 用于：QKV 投影、gate_up 投影

**RowParallelLinear**（输入维度分片）：
- `weight.shape = (out_features, in_features // tp_size)`
- 每个 rank 只存 1/tp_size 的输入列
- forward：`F.linear(input, weight)` → `dist.all_reduce(output)` → 得到完整输出
- 用于：Attention 的 o_proj、MLP 的 down_proj
- all_reduce 是 TP 的通信开销

**QKVParallelLinear**（最复杂）：
- Q/K/V 各有不同 num_heads，分片比例不同
- `weight.shape = ((q_size + kv_size) // tp_size, in_features)`
- `weight_loader(weight, tensor, shard_id)`：
  - shard_id="q" → tensor 切出 q 的部分，放到 weight 的 q offset
  - shard_id="k" → tensor 切出 k 的部分，放到 weight 的 k offset
  - shard_id="v" → tensor 切出 v 的部分，放到 weight 的 v offset
- 每个 rank 只加载 Q/K/V 各自的 1/tp_size 分片

**MergedColumnParallelLinear**（gate/up 融合）：
- `weight.shape = (out_features * 2 // tp_size, in_features)` （gate 和 up 各分片后拼接）
- `weight_loader(weight, tensor, shard_id)`：
  - shard_id=0 → gate 分片，放到 weight 的前半部分
  - shard_id=1 → up 分片，放到 weight 的后半部分

**TP 学习要点**：
- `tp_size=1` 时每层是完整权重，等价于普通 Linear —— 这是调试基准
- ColumnParallel = 切 output_dim，RowParallel = 切 input_dim + all_reduce
- QKVParallel 是最复杂的：Q/K/V 各有不同 num_heads，分片比例不同，offset 计算要精确
- weight_loader 是"延迟分片"：加载时才按 shard_id 切出对应 chunk 放到参数正确位置

#### 1.3 Embedding / LMHead (0.5天)

- **VocabParallelEmbedding**：词表按 tp_size 分片，每个 rank 存 vocab_size // tp_size 行
- **ParallelLMHead**：同 VocabParallel 结构，forward 时各 rank 算自己的 logits 分片 → gather 到 rank-0

#### 1.4 Qwen3ForCausalLM (1天)

- `Qwen3Config` 直接用 HuggingFace 的
- `Qwen3Attention`：
  - QKVParallelLinear（融合 q/k/v_proj）
  - RotaryEmbedding
  - flash_attn（prefill 用 varlen_func，decode 用 with_kvcache）
  - RowParallelLinear（o_proj）
- `Qwen3MLP`：
  - MergedColumnParallelLinear（融合 gate_proj + up_proj）
  - SiluAndMul
  - RowParallelLinear（down_proj）
- `Qwen3DecoderLayer`：
  - input_layernorm（RMSNorm + fused residual）
  - attention
  - post_attention_layernorm
  - MLP
- `Qwen3ForCausalLM`：
  - VocabParallelEmbedding
  - N 个 DecoderLayer
  - ParallelLMHead
  - `forward(input_ids, positions)` — positions 作为显式参数
  - `packed_modules_mapping`：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

#### 1.5 Attention (0.5天)

- prefill：`flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)`
- decode：`flash_attn_with_kvcache(q, k_cache, v_cache, block_table, cache_seqlens)`
- Triton `store_kvcache_kernel`：1D grid，flat slot 索引，每个 thread 写 1 个 token 的 KV 到 paged cache

#### 1.6 RoPE / RMSNorm / SiLU / Sampler (0.5天)

- **RotaryEmbedding**：预计算 cos/sin cache，`@torch.compile` forward
- **RMSNorm**：两个路径 — `rms_forward` 和 `add_rms_forward`（fuse residual + normalization），都用 `@torch.compile`
- **SiluAndMul**：`F.silu(x[..., :half]) * x[..., half:]`
- **Sampler**：`@torch.compile` — logits / temperature → softmax → exponential sampling

#### 1.7 LLMEngine + Worker 通信 (1天)

- **mp.spawn**：启动 tp_size 个 worker 进程
- **NCCL 初始化**：`tcp://localhost:29500`，rank 由 mp.spawn 自动分配
- **SharedMemory 通信**：
  - rank-0 写 input_ids/positions/block_tables 到 `/dev/shm/minivllm`（pickle）
  - 其他 rank 通过 Event 信号等待读取
  - 各 rank 独立 forward → all_reduce 汇合
- **generate()**：add_request → step 循环 → tqdm → 返回 `list[dict(text, token_ids)]`

#### 1.8 Loader (0.5天)

- 遍历 safetensors 文件
- 对每个权重名，查 `packed_modules_mapping`
  - 若命中：替换子名 → 得到融合参数名 → 调 `param.weight_loader(tensor, shard_id)`
  - 若未命中：调 `default_weight_loader(param, tensor)`
- 每个 rank 的 weight_loader 自动只加载自己的分片

**验证标准**：`tp_size=1` 单序列推理输出与 transformers 一致

---

### Phase 2 — 分页 KV Cache + BlockManager (2-3天)

**目标**：内存高效管理，支持多序列共存

| 任务 | 说明 |
|------|------|
| BlockManager | Block(block_id, ref_count, hash, token_ids)，allocate/free/append |
| Sequence.block_table | 每序列维护 block_table，class-level block_size |
| allocate_kv_cache | GPU 剩余内存算 num_blocks，shape `(2, num_layers, num_blocks, block_size, num_kv_heads_per_rank, head_dim)` |
| Triton store_kvcache | 1D grid，flat slot 索引 |
| decode 传 block_tables | `flash_attn_with_kvcache(block_tables, cache_seqlens)` |

**验证**：单序列推理仍正确

---

### Phase 3 — 连续批处理 + Scheduler (3-4天)

**目标**：多 prompt 同时推理，吞吐量提升

| 任务 | 说明 |
|------|------|
| Scheduler | waiting/running 双队列 |
| schedule() | token budget 选 prefill 序列，running 做 decode |
| postprocess() | EOS/max_tokens → FINISHED，释放 block |
| prepare_prefill | 拼 batch input_ids, positions, cu_seqlens, slot_mapping, block_tables |
| prepare_decode | 拼 batch input_ids(1/seq), positions, slot_mapping, block_tables, context_lens |
| LLMEngine.step() | schedule → run_model → postprocess 循环 |
| generate() | add_request → step 循环 → tqdm 进度条 |

**TP 注意**：所有 rank 共享 Scheduler 决策，rank-0 写 SharedMemory 传调度结果给其他 rank

**验证**：多条 prompt 并行推理输出正确

---

### Phase 4 — 分块 Prefill (2天)

**目标**：长 prompt 超预算也能处理，不卡死

| 任务 | 说明 |
|------|------|
| Sequence.num_scheduled_tokens | 跟踪已调度 token 数 |
| Scheduler.schedule() | 超预算时只调度部分 token，seq 仍在 waiting |
| WAITING→RUNNING | 仅当全部 prompt tokens 调度完才转换 |
| prepare_prefill | 支持部分 prompt 的 cu_seqlens 计算 |

**验证**：2048-token prompt 在 budget=512 下分多步完成

---

### Phase 5 — 抢占 + Prefix Cache (2-3天)

**目标**：内存不足时抢占序列，共享前缀复用 KV cache

| 任务 | 说明 |
|------|------|
| preempt() | running 队列尾部回退到 waiting，释放 block |
| can_allocate() | 返回 `num_cached_blocks`（-1 = 不足） |
| hash_blocks() | xxhash 计算每 block hash，注册到 `hash_to_block_id` |
| allocate(seq, num_cached) | 复用 cached blocks（ref_count++），仅新分配未缓存部分 |
| prefill 带 block_tables | flash_attn 读 cached KV，prefix cache 生效 |

**验证**：相似前缀 prompt 第二条明显更快

---

### Phase 6 — CUDA Graph Decode (2天)

**目标**：消除 decode 阶段 kernel launch 开销

**原理**：
- 问题：decode 每步只算 1 token，但要 launch 30+ 个 kernel，CPU→GPU 指令开销 ~0.3ms，比 GPU 计算时间还长
- 解法：
  1. 捕获阶段：跑一次完整 decode step，CUDA 记录所有 kernel 调用顺序和依赖，形成"图"
  2. 回放阶段：后续 step 把新数据拷进预分配 tensor，`graph.replay()` 一次指令触发整张图
- 约束：input/output tensor 地址必须固定（预分配），batch_size 固定（需为多个 batch 各捕获一张图）
- 只适合 decode（输入模式稳定）；prefill 不适合

| 任务 | 说明 |
|------|------|
| 捕获 decode graph | batch_sizes `[1,2,4,8,16,32]`，共享 graph_pool |
| graph_vars | 预分配 input_ids, positions, block_tables, context_lens 等 |
| positions 作为 graph 输入 | model.forward(input_ids, positions) — positions 是 graph input |
| replay | 拷数据 → graph.replay() → 取 logits |
| enforce_eager 开关 | Config 可选禁用 CUDA graph |

**验证**：eager vs graph TPS 对比（预期提升 ~30%）

---

### Phase 7 — 收尾打磨 (1-2天)

| 任务 | 说明 |
|------|------|
| __init__.py | 导出 LLM, SamplingParams |
| LLM 类 | LLMEngine 薄子类 |
| example.py | 示例推理脚本 |
| bench.py | 吞吐量基准测试 |

---

## 2060 关键参数配置

| 参数 | 值 | 原因 |
|------|-----|------|
| tensor_parallel_size | 1 | 单卡 |
| max_num_seqs | 32 | 6GB VRAM |
| kvcache_block_size | 16 | 小显存更细粒度 |
| max_num_batched_tokens | 512 | 限制 prefill 内存峰值 |
| gpu_memory_utilization | 0.9 | 尽量多用 |
| 可跑模型 | Qwen3-0.6B | ~1.2GB 权重 + ~3GB KV cache |

---

## 时间估算

| Phase | 天数 | TP 学习密度 |
|-------|------|------------|
| 0 骨架 | 1 | — |
| 1 单序列+TP | 5-6 | **最高** |
| 2 分页 KV | 2-3 | 中 |
| 3 批处理 | 3-4 | 中 |
| 4 分块 prefill | 2 | 低 |
| 5 抢占+前缀缓存 | 2-3 | 低 |
| 6 CUDA Graph | 2 | 低 |
| 7 收尾 | 1-2 | — |
| **总计** | **~18-22天** | |

---

## 踩坑预警

1. **禁止 greedy 采样** — temperature 必须 > 1e-10
2. **SharedMemory 通信** — rank-0 写 `/dev/shm/minivllm` pickle，其他 rank Event 等待读取
3. **Context 是全局单例** — 每个 step 修改，非线程安全
4. **CUDA graph** — positions 必须作为 graph input（不能在模型内部计算）
5. **packed_modules_mapping** — 必须正确映射 HF 名到融合名 + shard_id
6. **Config.model** 必须是本地目录路径，不能是 HF repo ID
7. **QKVParallel weight_loader** — offset 计算要精确考虑 num_heads / tp_size 的分片
8. **Block.reset()** — ref_count 应设为 1（正在分配），不是 0