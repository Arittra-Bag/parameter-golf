# Activation Ablation + EMA + GPTQ-lite

**val_bpb: TBD** | **~15.9 MB** | 8×H100 SXM

## Key Innovations

### 1. Configurable Activation Functions
Novel implementation supporting multiple activation functions for systematic ablation:

```python
# Supported activations (via ACTIVATION env var):
# - "relu2": torch.relu(x).square()          [baseline]
# - "leaky_relu2": F.leaky_relu(x).square()  [recommended by PR #493]
# - "swish2": (torch.sigmoid(x) * x).square() [novel to test]
# - "gelu2": F.gelu(x).square()              [alternative]

class MLP(nn.Module):
    def __init__(self, ..., activation: str = "leaky_relu2", leaky_slope: float = 0.5):
        ...
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.activation == "leaky_relu2":
            x = F.leaky_relu(x, negative_slope=self.leaky_slope).square()
        elif self.activation == "swish2":
            x = (torch.sigmoid(x) * x).square()
        elif self.activation == "gelu2":
            x = F.gelu(x).square()
        else:
            x = torch.relu(x).square()
        return self.proj(x)
```

### 2. LeakyReLU Slope Configuration
The LeakyReLU² activation allows configurable negative slope via `LEAKY_RELU_SLOPE`:
- 0.3: Stronger negative gradient flow
- 0.5: Default (confirmed -0.003 BPB in PR #493)
- 0.6-0.7: Stronger positive bias

### 3. Int6 + lzma Compression
GPTQ-lite int6 quantization for MLP and attention layers + lzma compression:
- Per-row optimal clip search
- 31-level quantization (-31 to 31)
- lzma preset 9 for best compression

## Architecture

Built on PR #414 stack with the following configuration:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with configurable activation |
| BigramHash | 2048 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | Int6 (MLP/Attn) + Int8 (Embeddings) + lzma |
| Optimizer | Muon with momentum warmup |

## Run Commands

**LeakyReLU² baseline (recommended first):**
```bash
ACTIVATION=leaky_relu2 LEAKY_RELU_SLOPE=0.5 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Swish² activation (novel to test):**
```bash
ACTIVATION=swish2 LEAKY_RELU_SLOPE=0.5 \
... rest of parameters same as above
```

**Slope sweep:**
```bash
for slope in 0.3 0.4 0.5 0.6 0.7; do
  ACTIVATION=leaky_relu2 LEAKY_RELU_SLOPE=$slope ... && echo "Slope $slope done"
done
```

## Expected Results

| Activation | Slope | Expected BPB |
|------------|-------|-------------|
| relu² | N/A | 1.1234 (baseline) |
| leaky_relu² | 0.5 | 1.1204 (-0.003 confirmed) |
| leaky_relu² | TBD | ~1.120 (slope sweep) |
| swish² | N/A | TBD (novel) |
| gelu² | N/A | TBD (alternative) |

## Ablation Plan

1. **Baseline**: relu² activation → record BPB
2. **LeakyReLU(0.5)²**: Known improvement → expect ~1.120
3. **Slope sweep**: Test 0.3, 0.4, 0.5, 0.6, 0.7 → find optimal
4. **Swish²**: Novel test → may improve further
5. **If Swish² works**: Combine with TTT from #1 submission

## Credits

- LeakyReLU² activation from [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- EMA + GPTQ-lite from [PR #374](https://github.com/openai/parameter-golf/pull/374) by @signalrush
- Base model from [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
