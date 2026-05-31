# 学习计划：由浅入深理解 MinivLLM

以 Qwen3-0.6B 为例，由浅入深逐步理解关键技术点。穿插小练习验证理解。

---

## 第1天 — 基础层组件 + 线性层体系

上午 — 简单组件（快速过，建立直觉）：

| 文件 | 关键点 | 一句话 |
|------|--------|-------|
| `activation.py` | SiluAndMul, `@torch.compile` | SwiGLU = silu(gate) * up |
| `sampler.py` | 温度采样，exponential sampling | softmax(logits/temp) / Exp(1) → argmax |
| `layernorm.py` | RMSNorm + fused residual | Pre-Norm: residual_rms_forward 融合残差+归一化 |
| `rotary_embedding.py` | RoPE, cos/sin cache, Llama3缩放 | 位置编码通过旋转矩阵注入 q/k |

**小练习**：用 Qwen3-0.6B 参数（head_dim=128, base=1e6）手动算 position=0 和 position=1 的 cos/sin 前 4 个值，对照代码输出验证。

下午 — 线性层体系（TP 核心，重点理解）：

| 类 | 关键点 |
|-----|--------|
| `LinearBase + ReplicatedLinear` | weight_loader 概念，tp_rank/tp_size |
| `ColumnParallelLinear` | 输出维度分片，forward 无通信，weight_loader 按行 narrow |
| `RowParallelLinear` | 输入维度分片 + `all_reduce`，bias 只在 rank-0 加 |
| `QKVColumnParallelLinear` | Q/K/V 各有不同 num_heads，offset 计算：shard_id='q'/'k'/'v' 定位 |
| `MergedColumnParallelLinear` | gate+up 融合，shard_id=0/1 定位 offset |
| `VocabParallelEmbedding + ParallelLMHead` | 词表分片，`dist.gather` 到 rank-0，`contiguous()` 保证内存连续 |

**重点问题**：
- `tp_size=1` 时等价普通 Linear — 调试基准
- QKV offset：Qwen3-0.6B 16 qo_heads + 8 kv_heads，`tp_size=2` 时每 GPU 分 8+4+4 heads
- RowParallel 的 `all_reduce` 是 TP 通信开销点
- `dist.gather` vs `dist.all_gather`：前者只有 dst GPU 收全量，后者所有 GPU 都收

---

## 第2天 — Qwen3 模型组装 + Attention接口 + 权重加载

上午 — Qwen3 模型组装：

| 类 | 关键点 |
|-----|--------|
| `Qwen3Attention` | QKV投影 → q_norm/k_norm → RoPE → Attention → o_proj |
| `Qwen3MLP` | gate_up 融合 → SiluAndMul → down_proj |
| `Qwen3DecoderLayer` | fused residual 连接：`residual_rms_forward` 一步完成 |
| `Qwen3Model + Qwen3ForCausalLM` | embed → 28层 → final_norm → lm_head |

**重点问题**：
- 为什么 `num_heads` 是 per-GPU？每 GPU 独立算不同 head，attention 内部无通信
- 为什么 RMS 只作用在 Q/K？Q/K 参与 score 计算，大值导致 softmax 不稳定；V 不影响 score
- 为什么 gate_up 用 MergedColumnParallelLinear？为了与 HF checkpoint 兼容，不能简单用 `intermediate_size * 2`
- positions 从 Context 动态计算（prefill 按 cu_seqlens，decode 按 context_lens-1）
- `packed_modules_mapping`：HF 名 → 融合名 + shard_id

下午 — Attention 接口（只学怎么用，不深入 kernel）：

| 要点 | 说明 |
|------|------|
| `Attention` 类的两条路径 | prefill: store_kvcache → flash_attention_varlen；decode: store_kvcache → paged_attention_decode |
| k_cache/v_cache 注入 | 由 `ModelRunner.allocate_kv_cache()` 分配后注入到各层 |
| Context 依赖 | slot_mapping, cu_seqlens, block_tables, context_lens 从 `get_context()` 读 |
| 输入模式 | 3D varlen（prefill）vs 4D batched（decode） |

下午 — 权重加载 + Sequence：

| 文件 | 关键点 |
|------|--------|
| `loader.py` | safetensors 遍历，遇 `q_proj.weight` → 找 `k_proj/v_proj` → 拼接写入 `qkv_proj` |
| `sequence.py` | SequenceStatus(WAITING/RUNNING/FINISHED), token_ids 追踪, block_table, `__getstate__/__setstate__` TP 优化 |

**重点问题**：
- loader 硬编码合并：遇到 gate_proj → 找 up_proj → 拼接写入 gate_up_proj
- Sequence 的 `token_ids = copy(token_ids)`：必须拷贝，否则外部修改会影响内部
- `__getstate__/__setstate__`：prefill 传全 token_ids，decode 只传 last_token（减少 TP 通信量）

---

## 第3天 — 分页 KV Cache + 前缀缓存

上午 — Block 类 + hash：

| 部分 | 关键点 |
|------|--------|
| `Block` | block_id, hash(-1=未计算), ref_count, token_ids |
| `compute_hash` | xxhash 增量 hash，参数包含 prefix_hash 保证唯一性 |
| 前缀缓存原理 | 相同前缀 → hash 匹配 → ref_count++ → 跳过写入；`[prefix_hash_1][1,2,3]` ≠ `[prefix_hash_2][1,2,3]` |

下午 — BlockManager 核心方法：

| 方法 | 关键点 |
|------|--------|
| `can_allocate(seq)` | 检查空闲 block 是否足够 |
| `allocate(seq)` | 逐 block 计算 hash → 缓存命中(复用) vs 未命中(分配新) → 碰撞检测(`block_id!=-1` 但 `token_ids!=...`) |
| `can_append(seq)` | 检查新 block 需要时是否有空闲 block |
| `append(seq)` | block 满 → 计算 hash + 注册；新 block → 分配 + 追加 block_table |
| `deallocate(seq)` | ref_count 递减，归零时回收 block |

**重点问题**：
- 物理 KV cache 在 GPU tensor 上，BlockManager 只管逻辑映射
- hash 碰撞：即使 hash 匹配，还要检查 token_ids 是否一致（防碰撞）
- `Block.reset()` 中 `ref_count=0`（Plan.md 说应为 1，这是个 bug）

**小练习**：模拟 3 个有相同前缀的序列，画出 allocate 过程中 ref_count 的变化。

---

## 第4天 — FlashAttention 数学原理 + Prefill Kernel（核心难点①）

上午 — 数学原理（不看代码，先理解为什么）：

| 主题 | 关键点 |
|------|--------|
| 标准 Attention 内存问题 | O(N²) 内存存完整 attention matrix → 长 prompt 爆显存 |
| Tiling（分块） | Q/K/V 分成小块，逐块计算，不存完整矩阵 |
| Online Softmax | 逐块更新 numerator/denominator（m, l, o），O(N) 内存 |
| Causal Mask | `(offs_m + seq_start) >= (offs_n + seq_start)` 只算下半三角 |
| GQA | `kv_head_idx = head_idx // group_size`（Qwen3 group_size=2） |
| GPU 架构 | 每个 3D grid 有 4 WARP，每 WARP 32 线程，共 128 线程 |
| Triton 自动提取指针 | PyTorch tensor 传给 Triton kernel 时自动提取内存地址 |

下午 — store_kvcache + flash_attention_varlen kernel：

| 部分 | 关键点 |
|------|--------|
| `store_kvcache_kernel` | Grid: `(num_tokens, num_kv_heads)`，slot_mapping → block_idx + offset 定位写入位置，slot=-1 跳过 padding |
| `flash_attention_varlen_kernel` | Grid: `(num_blocks_M, num_heads, num_seqs)`，逐 block 做 online softmax，BLOCK_M/N 根据 head_dim 自适应 |
| `cu_seqlens` | `[0, len_seq1, len_seq1+len_seq2, ...]` 标记变长序列边界 |
| stride() | tensor 在内存中是 1D 连续数组，stride 描述沿某维度移动需跳过多少元素 |

**小练习**：用纸笔画 2-sequence prefill 的 flash attention 计算过程，标注每个 program block 的 m/l/o 更新。

---

## 第5天 — PagedAttention Decode Kernel + Attention 类串联（核心难点②）

上午 — PagedAttention：

| 部分 | 关键点 |
|------|--------|
| `paged_attention_decode_kernel` | Grid: `(batch_size, num_heads)`，每 program 处理 1 batch × 1 head |
| block_tables 查找 | 物理 block id → 从 k_cache/v_cache 按 block 读取 KV |
| 逐 chunk online softmax | BLOCK_N=64/32，每 chunk 更新 numerator/denominator |
| 内循环逐 token 加载 | 性能瓶颈点，与 flash-attn 库的优化对比 |

下午 — Attention 类串联 + Context 全局单例：

| 部分 | 关键点 |
|------|--------|
| `Attention` 类完整流程 | prefill 路径 vs decode 路径，k_cache/v_cache 注入时机 |
| `Context` 单例 | `is_prefill, cu_seqlens, slot_mapping, block_tables, context_lens` 跨层共享 |
| 每步生命周期 | `set_context` → forward → `reset_context`，非线程安全 |
| positions 动态计算 | prefill: 按 cu_seqlens 重建位置；decode: context_lens-1 |

**对比思考**：MinivLLM 自研 Triton kernel vs nano-vLLM 用 flash-attn 库，优劣各是什么？

---

## 第6天 — Scheduler + 连续批处理

上午 — 调度逻辑：

| 部分 | 关键点 |
|------|--------|
| 双队列 | waiting（新请求）+ running（正在 decode） |
| 优先 prefill > decode | waiting 有请求时先处理，即使 running 不空 |
| token budget 控制 | `max_num_batched_tokens` + `max_num_sequences` 限制每步吞吐 |
| 抢占 `preempt` | 内存不足 → running 尾部回退 waiting → 释放 block |

下午 — 后处理 + 数据准备串联：

| 部分 | 关键点 |
|------|--------|
| `postprocess` | append_token → 检查 EOS / max_tokens / max_model_length → FINISHED 释放 block |
| `prepare_prefill` | 展平 input_ids（FlashAttention 要求单次 kernel）+ `cu_seqlens` 标边界 + slot_mapping 只含未缓存 token + `pin_memory`/`non_blocking` |
| `prepare_decode` | 每 seq 1 token + slot = `block_table[-1] * block_size + last_block_num_tokens - 1` |

**重点问题**：
- `cu_seqlens` 没有 `cu_seqlens_v`？因为 K 和 V 序列结构一致
- `pin_memory=True`？锁定物理内存页，DMA 直传 GPU，省 1 次拷贝
- `slot_mapping` 只含未缓存 token？已缓存 KV 不需重写
- 为什么不用担心 decode slot 重叠？`append()` 保证 block 不会重叠

---

## 第7天 — ModelRunner + CUDA Graph + SharedMemory

上午 — ModelRunner 核心：

| 部分 | 关键点 |
|------|--------|
| `warmup_model` | 空序列预热，`torch.cuda.memory_stats()['allocated_bytes.all.peak']` 测峰值（不含 KV cache） |
| `allocate_kv_cache` | 剩余显存 → block 大小 → 可分配 block 数 → 多卡 `all_reduce(MIN)` 同步 → `torch.zeros` 分配 → 注入各层 |
| 权重加载顺序 | 先 `model.cuda(rank)` 再 `load_weights_from_checkpoint`（GPU 上加载），CPU 加载可能有问题 |

下午 — CUDA Graph + SharedMemory：

| 部分 | 关键点 |
|------|--------|
| `capture_cudagraph` | 预分配固定 tensor，`[1,2,4,8]+range(16,max+1,16)` 反序捕获共享 graph_pool |
| 为什么只用于 decode？ | decode 输入模式固定（1 token/seq）；prefill 输入长度可变 |
| warmup → capture | CUDA graph 要求 capture 前完成所有内存分配，warmup 触发惰性分配 |
| `torch.compile` vs CUDA Graph | compile 融合 kernel（减少数量）；graph 消除 launch 开销（减少 CPU 参与）。组合 = 双重加速 |
| SharedMemory 通信 | rank-0 写 `/dev/shm/myvllm` pickle（4字节头+数据），其他 rank `Event.wait()` 读 |
| `self.event` vs `self.events` | Worker: 单个 Event；Master: Event 列表，逐一 set 通知每个 worker |
| `call(method_name, *args)` | rank-0 写 shm + set events；其他 rank 读 shm → 调用方法 |

**重点问题**：
- `torch.cuda.synchronize()` 在 `reset_context()` 前：确保 capture 完成
- rank-0 采样：所有 rank 计算相同 logits（或 gather 到 rank-0），只需采样一次
- `atexit.register(self.exit)`：防止 worker 僵尸进程

---

## 第8天 — LLMEngine + 全流程串联

上午 — LLMEngine：

| 部分 | 关键点 |
|------|--------|
| `__init__` | `mp.spawn` 启动 worker → ModelRunner(rank-0) → NCCL barrier → Scheduler |
| 为什么 Scheduler 在 ModelRunner 之后？ | NCCL `init_process_group` 是 collective barrier，所有 rank 汇合后才返回 |
| `step` | schedule → model_runner.call("run") → postprocess → 返回 (finished_outputs, num_processed_tokens, is_prefill) |
| `generate` | add_prompt → step 循环 → is_finished → 打印 throughput → 返回 {text, token_ids} |
| `atexit.register(self.exit)` | 程序退出时自动清理 worker 进程，防僵尸 |

下午 — 用 main.py 跑一遍 Qwen3-0.6B，对照输出验证理解。

---

## 第9天 — 总结回顾 + 完整数据流图

上午 — 画出完整数据流：

```
generate("你好")
  → tokenize → Sequence(token_ids=[151644, ...])
  → scheduler.add_sequence → waiting 队列

  → step 循环 (prefill):
    → scheduler.schedule() → (scheduled_seqs, is_prefill=True)
    → block_manager.allocate(seqs) → 计算hash → 缓存命中/未命中 → block_table
    → model_runner.prepare_prefill(seqs):
        展平 input_ids, cu_seqlens, slot_mapping(跳过已缓存), block_tables → set_context
    → model_runner.run_model:
        Qwen3ForCausalLM.forward:
          VocabParallelEmbedding → 28层 DecoderLayer → final_norm → compute_logits
          每层: layernorm → QKV投影 → q_norm/k_norm → RoPE → Attention → o_proj(all_reduce) → layernorm → MLP → residual
        Attention.prefill: store_kvcache → flash_attention_varlen → 输出
    → sampler(logits, temperature) → sampled_token_ids (rank-0 only)
    → scheduler.postprocess: append_token → 检查 EOS → 未结束 → 状态=RUNNING

  → step 循环 (decode):
    → scheduler.schedule() → (running_seqs, is_prefill=False)
    → block_manager.append(seqs) → block满了就计算hash+分配新block
    → model_runner.prepare_decode(seqs):
        input_ids(1token/seq), slot_mapping, context_lens, block_tables → set_context
    → model_runner.run_model → CUDA graph replay → logits
    → sampler → token → postprocess → 检查EOS → 结束 → deallocate block → FINISHED
  → decode tokens → 返回文本
```

下午 — MinivLLM vs nano-vLLM 对比 + 知识点清单复盘：

| 对比维度 | MinivLLM | nano-vLLM |
|----------|----------|-----------|
| Attention 实现 | 自研 Triton kernel | flash-attn 库 |
| 模型支持 | Qwen3 + Llama3.2 | 仅 Qwen3 |
| 权重加载 | 硬编码合并 | packed_modules_mapping 驱动 |
| TP 通信 | shm `myvllm`，端口 `12345` | shm `nanovllm`，端口 `2333` |
| KV cache 分配 | `torch.zeros` + 跨卡 MIN | 无 MIN 同步 |

---

## Qwen3-0.6B 关键参数

| 参数 | 值 |
|------|-----|
| hidden_size | 1024 |
| num_heads (qo) | 16 |
| num_kv_heads | 8 (GQA, group_size=2) |
| head_dim | 128 |
| num_layers | 28 |
| RoPE base | 1,000,000 |
| q_norm/k_norm | 有（qkv_bias=False 时启用） |