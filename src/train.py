"""
train_gpt_simple.py

Fork of `records/track_3_optimization/train_gpt_simple.py` from modded-nanogpt.
    repo:   https://github.com/KellerJordan/modded-nanogpt
    commit: 73ba75775fb134157cc7a8d61f3b0d1b524b74e6
"""

import argparse
import json
import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


def measure_noise_scale(model, params, loader, b_small_seqs, mbs, seq_len):
    """McCandlish (2018) two-independent-batch gradient noise scale.

    For two disjoint micro-batches of b_small_seqs sequences (split across ranks),
    compute the mean per-sequence gradient G1, G2 on `params`. Then
      tr(Sigma)        ~= (b_small_seqs / 2) * |G1 - G2|^2
      |G_true|^2       ~= G1 . G2
      B_noise_examples  = tr(Sigma) / |G_true|^2
      B_noise_tokens    = B_noise_examples * seq_len

    Caller's grads are zeroed before return. Loader is a separate generator
    (different shards from training) yielding (b_small_seqs / world_size)
    sequences per call.
    """
    grads = []
    for _ in range(2):
        for p in params:
            p.grad = None
        inputs, targets = next(loader)
        assert len(inputs) % mbs == 0
        for i in range(0, len(inputs), mbs):
            model(inputs[i:i+mbs], targets[i:i+mbs]).backward()
        for p in params:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        grads.append(torch.cat([p.grad.detach().float().flatten() for p in params]) / float(b_small_seqs))
    g1, g2 = grads
    dot = float(torch.dot(g1, g2))
    diff_sq = float(torch.sum((g1 - g2) ** 2))
    tr_sigma = (b_small_seqs / 2.0) * diff_sq
    if dot > 0:
        B_noise_tokens = tr_sigma / dot * seq_len
    else:
        B_noise_tokens = float("nan")
    # Zero ALL model grads, not just Muon-managed `params`. Backward() inside the probe
    # populates .grad on every trainable param (incl. AdamW-managed embed/proj/scalars);
    # leaving those stale would contaminate the next training step's gradient on those params.
    for p in model.parameters():
        p.grad = None
    return {"dot": dot, "diff_sq": diff_sq, "tr_sigma": tr_sigma, "B_noise_tokens": B_noise_tokens}


########################################
#               CLI args               #
########################################

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="JSON file to override CLI args")
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=524288, help="global batch in tokens")
parser.add_argument("--mu", type=float, default=0.95, help="Muon momentum")
parser.add_argument("--lr_muon", type=float, default=0.035)
parser.add_argument("--total_tokens", type=int, default=1_000_000_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mbs", type=int, default=8, help="per-rank micro-batch size in sequences")
parser.add_argument("--noise_target_samples", type=int, default=100, help="target number of noise scale probes per run, 0 disables")
parser.add_argument("--b_small_seqs", type=int, default=128, help="sequences per noise-scale micro-batch")
args = parser.parse_args()
if args.config is not None:
    with open(args.config) as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        assert hasattr(args, k), f"unknown config key: {k}"
        setattr(args, k, v)
run_id = args.run_id or f"B{args.batch_size}_mu{args.mu:.2f}_lr{args.lr_muon:.5f}_seed{args.seed}"

# Constants kept in-file (not CLI):
val_tokens = 20 * 524288
batch_size = args.batch_size
mbs = args.mbs
seq_len = 1024


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0
torch.manual_seed(args.seed + dist.get_rank())
torch.cuda.manual_seed(args.seed + dist.get_rank())

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"run_id: {run_id}")
print0(f"args: {json.dumps(vars(args), indent=2)}")
# print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
#        + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))
noise_loader = None
if args.noise_target_samples > 0:
    noise_loader = distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", args.b_small_seqs * seq_len)

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# train for a fixed token budget
train_steps = max(1, args.total_tokens // batch_size)
val_step_freq = max(1, 25_000_000 // batch_size)  # ~25M tokens between val passes
# Noise probes are uniform-in-step-space with `noise_target_samples` per run,
# so overhead is constant in fraction-of-training compute across B.
noise_K = max(1, train_steps // args.noise_target_samples) if args.noise_target_samples > 0 else 0
print0(f"train_steps={train_steps}  val_step_freq={val_step_freq}  noise_K={noise_K}", console=True)

# initialize model parameters
for name, p in model.named_parameters():
    w = p.data
    if name.endswith("weight"):
        if "proj" in name:
            w.zero_()
        elif "embed" in name:
            w.normal_()  # default torch init
        else:
            w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
    elif name.endswith("bias"):
        w.zero_()
    elif name.endswith("gains"):
        w.normal_(mean=1, std=0)
    else:
        raise Exception(f"Uninitialized parameter: {name}")

# create the optimizer(s)
muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
optimizer2 = Muon(muon_params, lr=args.lr_muon, weight_decay=0.025, mu=args.mu)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def set_hparams(step, cooldown_frac=0.7):
    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < 1 - cooldown_frac:
        eta = 1.0
    else:
        eta = (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


########################################
#        Training and Validation       #
########################################

val_loss_trace = []
noise_scale_trace = []
train_step_trace = []  # per-step: train_loss, tokens, examples, lr, time
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
last_val_step = 0
dist.barrier()
run_start = time.perf_counter()
t0 = run_start
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    if step == train_steps or step % val_step_freq == 0:
        # stop the clock
        dist.barrier()
        time_since_last_val = time.perf_counter() - t0
        step_avg = time_since_last_val / (step - last_val_step) if step > last_val_step else float("nan")
        last_val_step = step
        training_time += time_since_last_val
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss = float(val_loss) / val_tokens
        val_loss_trace.append({"step": step, "tokens": step * batch_size,
                               "val_loss": val_loss, "training_time_s": training_time})
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*step_avg:.2f}ms", console=True)
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    # --------------- NOISE SCALE SECTION -----------------
    if noise_K > 0 and 0 < step < train_steps and step % noise_K == 0:
        dist.barrier()
        ns_t0 = time.perf_counter()
        ns = measure_noise_scale(model, muon_params, noise_loader,
                                 args.b_small_seqs, mbs, seq_len)
        ns.update({"step": step, "tokens": step * batch_size,
                   "measure_time_s": time.perf_counter() - ns_t0})
        noise_scale_trace.append(ns)
        print0(f"step:{step}/{train_steps} B_noise_tokens:{ns['B_noise_tokens']:.1f}"
               + f" dot:{ns['dot']:.3e} diff_sq:{ns['diff_sq']:.3e}", console=True)
        dist.barrier()
        t0 = time.perf_counter()  # don't count noise measurement as training time

    if step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    train_loss_sum = torch.zeros((), device="cuda")
    for i in range(len(inputs) // mbs):
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        loss.backward()
        train_loss_sum = train_loss_sum + loss.detach()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
    train_loss_val = float(train_loss_sum) / batch_size
    # set optimization hyperparameters and take a step
    set_hparams(step)
    lr_muon_now = optimizer2.param_groups[0]["lr"]
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time = training_time + (time.perf_counter() - t0)
    train_step_trace.append({
        "step": step + 1,
        "tokens": (step + 1) * batch_size,
        "examples": (step + 1) * batch_size // seq_len,
        "train_loss": train_loss_val,
        "lr_muon": lr_muon_now,
        "training_time_s": approx_training_time,
    })
    print0(f"step:{step+1}/{train_steps} train_loss:{train_loss_val:.4f}"
           + f" train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms",
           console=True, log=False)

wallclock_seconds = time.perf_counter() - run_start

# dump JSON on rank 0
if dist.get_rank() == 0:
    with open(f"results/{run_id}.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "config": vars(args),
            "world_size": dist.get_world_size(),
            "train_steps": train_steps,
            "val_step_freq": val_step_freq,
            "wallclock_seconds": wallclock_seconds,
            "final_loss": val_loss_trace[-1]["val_loss"] if val_loss_trace else None,
            "val_loss_trace": val_loss_trace,
            "noise_scale_trace": noise_scale_trace,
            "train_step_trace": train_step_trace,
        }, f, indent=2)
    print0(f"wrote results/{run_id}.json", console=True)

dist.destroy_process_group()
