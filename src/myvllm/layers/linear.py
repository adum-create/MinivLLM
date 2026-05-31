import torch.nn as nn 
import torch
import torch.distributed as dist

class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim 
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""

# the simpliest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
class ReplicatedLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# columnsplit Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter
class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the output column
        full_data_output_size = loaded_weights.size(0)
        # dim size after sharding
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# an extension of ColumnParallelLinear by merging several matrices
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, 
        input_size: int, 
        output_sizes: list[int], # e.g. merge QKV matrices to compute MM together and then split
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),  
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        load to 
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        ) which is also sharded by tp_size
        """
        param_data = param.data
        # compute offset 
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        # compute size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # find the correct slice to be loaded in the sharded parameter
        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,  # 128
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads // self.tp_size  # Q头数
        self.num_kv_heads = num_kv_heads // self.tp_size
        # Calculate per-GPU output size
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # Pass TOTAL output size to parent (it will divide by tp_size)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    # load_weight_id: q, k, v
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        # batch_size * num_heads * num_token * head_size
        param_data = param.data
        # loaded_weights: batch_size * num_token * (head_size*num_heads)
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # compute offset
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        if param_data.ndim == 1:
            param_data.copy_(loaded_weights)
            return
        # full_dim on the input row
        full_data_input_size = loaded_weights.size(1)
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result


def _run_all_tests(rank: int, world_size: int):
    import os
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29501", rank=rank, world_size=world_size)

    passed = 0

    # ======================== 1. ReplicatedLinear ========================
    torch.manual_seed(1)
    input_size, output_size = 8, 6
    full_w = torch.arange(output_size * input_size, dtype=torch.float32).reshape(output_size, input_size)
    full_b = torch.arange(output_size, dtype=torch.float32)

    layer = ReplicatedLinear(input_size, output_size, bias=True)
    layer.weight.weight_loader(layer.weight, full_w)
    layer.bias.weight_loader(layer.bias, full_b)

    assert torch.allclose(layer.weight.data, full_w), f"rank {rank}: ReplicatedLinear weight mismatch"
    assert torch.allclose(layer.bias.data, full_b), f"rank {rank}: ReplicatedLinear bias mismatch"

    torch.manual_seed(10)
    x = torch.randn(2, input_size)
    result = layer(x)
    expected = nn.functional.linear(x, full_w, full_b)
    assert torch.allclose(result, expected, atol=1e-5), f"rank {rank}: ReplicatedLinear forward mismatch"
    passed += 1
    if rank == 0:
        print(f"✓ [1/5] ReplicatedLinear: weight_loader copies full weight, forward = standard linear")

    # ======================== 2. ColumnParallelLinear ========================
    torch.manual_seed(2)
    input_size, output_size = 8, 6
    shard_output = output_size // world_size  # 3

    full_w = torch.arange(output_size * input_size, dtype=torch.float32).reshape(output_size, input_size)
    full_b = torch.arange(output_size, dtype=torch.float32)

    layer = ColumnParallelLinear(input_size, output_size, bias=True)
    assert layer.weight.shape == (shard_output, input_size), \
        f"rank {rank}: ColumnParallelLinear weight shape {layer.weight.shape}, expected ({shard_output}, {input_size})"

    layer.weight.weight_loader(layer.weight, full_w)
    layer.bias.weight_loader(layer.bias, full_b)

    start_row = rank * shard_output
    expected_w = full_w.narrow(0, start_row, shard_output)
    expected_b = full_b.narrow(0, start_row, shard_output)
    assert torch.allclose(layer.weight.data, expected_w), f"rank {rank}: ColumnParallelLinear weight shard mismatch"
    assert torch.allclose(layer.bias.data, expected_b), f"rank {rank}: ColumnParallelLinear bias shard mismatch"

    torch.manual_seed(10)
    x = torch.randn(2, input_size)
    result = layer(x)
    expected = nn.functional.linear(x, expected_w, expected_b)
    assert torch.allclose(result, expected, atol=1e-5), f"rank {rank}: ColumnParallelLinear forward mismatch"
    passed += 1
    if rank == 0:
        print(f"✓ [2/5] ColumnParallelLinear: each rank gets row shard [rank*3:(rank+1)*3], forward = partial output (no all_reduce)")

    # ======================== 3. MergedColumnParallelLinear ========================
    torch.manual_seed(3)
    input_size = 8
    output_sizes = [6, 6]  # gate=6, up=6, total=12
    total_output = sum(output_sizes)  # 12
    shard_total = total_output // world_size  # 6 per rank (gate_shard=3, up_shard=3)

    gate_full = torch.arange(output_sizes[0] * input_size, dtype=torch.float32).reshape(output_sizes[0], input_size)
    up_full = torch.arange(output_sizes[1] * input_size, dtype=torch.float32).reshape(output_sizes[1], input_size) + 100

    layer = MergedColumnParallelLinear(input_size, output_sizes, bias=False)
    assert layer.weight.shape == (shard_total, input_size), \
        f"rank {rank}: MergedColumnParallelLinear weight shape {layer.weight.shape}, expected ({shard_total}, {input_size})"

    layer.weight.weight_loader(layer.weight, gate_full, loaded_weight_id=0)
    layer.weight.weight_loader(layer.weight, up_full, loaded_weight_id=1)

    gate_shard_size = output_sizes[0] // world_size  # 3
    up_shard_size = output_sizes[1] // world_size    # 3
    gate_offset = 0
    up_offset = gate_shard_size  # 3

    expected_gate_shard = gate_full.narrow(0, rank * gate_shard_size, gate_shard_size)
    expected_up_shard = up_full.narrow(0, rank * up_shard_size, up_shard_size)
    expected_merged = torch.cat([expected_gate_shard, expected_up_shard], dim=0)

    assert torch.allclose(layer.weight.data, expected_merged), \
        f"rank {rank}: MergedColumnParallelLinear merged weight mismatch\n  got:      {layer.weight.data}\n  expected: {expected_merged}"

    torch.manual_seed(10)
    x = torch.randn(2, input_size)
    result = layer(x)
    expected = nn.functional.linear(x, expected_merged, None)
    assert torch.allclose(result, expected, atol=1e-5), f"rank {rank}: MergedColumnParallelLinear forward mismatch"
    passed += 1
    if rank == 0:
        print(f"✓ [3/5] MergedColumnParallelLinear: gate+up weights loaded with offset, each rank gets [gate_shard | up_shard]")

    # ======================== 4. QKVColumnParallelLinear ========================
    torch.manual_seed(4)
    input_size, head_size = 8, 2
    num_heads, num_kv_heads = 8, 2  # GQA: 8 Q heads, 2 KV heads
    # Per GPU: num_heads=4, num_kv_heads=1
    # Per GPU output = 2*(4 + 2*1) = 12
    # Total output = 2*(8 + 2*2) = 24

    q_full = torch.arange(num_heads * head_size * input_size, dtype=torch.float32).reshape(num_heads * head_size, input_size)
    k_full = torch.arange(num_kv_heads * head_size * input_size, dtype=torch.float32).reshape(num_kv_heads * head_size, input_size) + 100
    v_full = torch.arange(num_kv_heads * head_size * input_size, dtype=torch.float32).reshape(num_kv_heads * head_size, input_size) + 200

    layer = QKVColumnParallelLinear(input_size, head_size, num_heads, num_kv_heads, bias=False)
    per_gpu_heads = num_heads // world_size  # 4
    per_gpu_kv = num_kv_heads // world_size  # 1
    per_gpu_output = head_size * (per_gpu_heads + 2 * per_gpu_kv)  # 2*(4+2) = 12
    assert layer.weight.shape == (per_gpu_output, input_size), \
        f"rank {rank}: QKVColumnParallelLinear weight shape {layer.weight.shape}, expected ({per_gpu_output}, {input_size})"

    layer.weight.weight_loader(layer.weight, q_full, load_weight_id='q')
    layer.weight.weight_loader(layer.weight, k_full, load_weight_id='k')
    layer.weight.weight_loader(layer.weight, v_full, load_weight_id='v')

    q_shard = q_full.narrow(0, rank * (per_gpu_heads * head_size), per_gpu_heads * head_size)
    k_shard = k_full.narrow(0, rank * (per_gpu_kv * head_size), per_gpu_kv * head_size)
    v_shard = v_full.narrow(0, rank * (per_gpu_kv * head_size), per_gpu_kv * head_size)
    expected_qkv = torch.cat([q_shard, k_shard, v_shard], dim=0)

    assert torch.allclose(layer.weight.data, expected_qkv), \
        f"rank {rank}: QKVColumnParallelLinear weight mismatch\n  got:      {layer.weight.data}\n  expected: {expected_qkv}"

    torch.manual_seed(10)
    x = torch.randn(2, input_size)
    result = layer(x)
    expected = nn.functional.linear(x, expected_qkv, None)
    assert torch.allclose(result, expected, atol=1e-5), f"rank {rank}: QKVColumnParallelLinear forward mismatch"

    q_size = per_gpu_heads * head_size
    kv_size = per_gpu_kv * head_size
    q_out, k_out, v_out = result.split([q_size, kv_size, kv_size], dim=-1)
    expected_q, expected_k, expected_v = expected.split([q_size, kv_size, kv_size], dim=-1)
    assert torch.allclose(q_out, expected_q, atol=1e-5), f"rank {rank}: QKV q split mismatch"
    assert torch.allclose(k_out, expected_k, atol=1e-5), f"rank {rank}: QKV k split mismatch"
    assert torch.allclose(v_out, expected_v, atol=1e-5), f"rank {rank}: QKV v split mismatch"
    passed += 1
    if rank == 0:
        print(f"✓ [4/5] QKVColumnParallelLinear: Q/K/V weights loaded with correct offsets, forward split matches")

    # ======================== 5. RowParallelLinear ========================
    torch.manual_seed(5)
    input_size, output_size = 8, 4
    shard_input = input_size // world_size  # 4

    full_w = torch.arange(output_size * input_size, dtype=torch.float32).reshape(output_size, input_size)
    full_b = torch.arange(output_size, dtype=torch.float32)

    layer = RowParallelLinear(input_size, output_size, bias=True)
    assert layer.weight.shape == (output_size, shard_input), \
        f"rank {rank}: RowParallelLinear weight shape {layer.weight.shape}, expected ({output_size}, {shard_input})"

    layer.weight.weight_loader(layer.weight, full_w)
    layer.bias.weight_loader(layer.bias, full_b)

    expected_w_shard = full_w.narrow(1, rank * shard_input, shard_input)
    assert torch.allclose(layer.weight.data, expected_w_shard), f"rank {rank}: RowParallelLinear weight shard mismatch"
    assert torch.allclose(layer.bias.data, full_b), f"rank {rank}: RowParallelLinear bias not replicated"

    torch.manual_seed(10)
    x = torch.randn(2, input_size)
    x_shard = x.narrow(1, rank * shard_input, shard_input)
    result = layer(x_shard)
    expected = nn.functional.linear(x, full_w, full_b)
    assert torch.allclose(result, expected, atol=1e-5), f"rank {rank}: RowParallelLinear forward+all_reduce mismatch"
    passed += 1
    if rank == 0:
        print(f"✓ [5/5] RowParallelLinear: weight sliced by columns, bias replicated, all_reduce = full linear result")

    if rank == 0:
        print(f"\n=== {passed}/5 ALL TESTS PASSED ===\n")

    dist.destroy_process_group()


def run_all_tests():
    import os
    import torch.multiprocessing as mp
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    mp.spawn(_run_all_tests, args=(2,), nprocs=2, join=True)


if __name__ == "__main__":
    run_all_tests()