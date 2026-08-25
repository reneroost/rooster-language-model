# Language Model in Kotlin — Roadmap

A from-scratch small language model in Kotlin. Phase 1 trains on CPU; phase 2 moves the
compute to an RX 570 via LWJGL + Vulkan compute shaders.

**The design principle:** don't abstract everything. *Localize the compute behind a small set
of named primitive functions*, and keep model / autograd / training loop unaware of where the
numbers actually live. Extract the `Backend` interface at the **end** of phase 1, once you know
which ops you actually needed.

**The layering:**

| Layer | Phase 1 | Phase 2 | Fate |
|---|---|---|---|
| 1. Storage | `FloatArray` | GPU buffer handle | **swap line** |
| 2. Backend ops | CPU loops | kernel dispatches | rewritten |
| 3. Autograd tape | closures over tensors | unchanged | reused |
| 4. Model + training loop | pure composition | unchanged | reused |

**The thing people miss:** phase 1 is the *correctness oracle* for phase 2. When the Vulkan
matmul returns garbage (it will), you diff it against the CPU matmul on the same inputs. That
alone justifies building phase 1 carefully.

---

## Table of contents

- [Design notes and corrections](#design-notes-and-corrections)
- [Roadmap overview](#roadmap-overview)
- [Step 0 — Skeleton and determinism](#step-0--skeleton-and-determinism)
- [Step 1 — Data pipeline](#step-1--data-pipeline)
- [Step 2 — Tensor](#step-2--tensor)
- [Step 3 — Ops (the seam)](#step-3--ops-the-seam)
- [Step 4 — Bigram with hand-rolled backward](#step-4--bigram-with-hand-rolled-backward)
- [Step 5 — The autograd tape](#step-5--the-autograd-tape)
- [Step 6 — MLP and real training infrastructure](#step-6--mlp-and-real-training-infrastructure)
- [The next cliff: attention shapes](#the-next-cliff-attention-shapes)
- [Invariants worth pinning to the wall](#invariants-worth-pinning-to-the-wall)

---

## Design notes and corrections

Changes made to the original roadmap, and why.

### 1. Autograd needs a slot in the ladder

It was named as a "reused" layer but never scheduled. The decision is load-bearing: a
micrograd-style **scalar** tape is a trap at 1M params (hundreds of millions of `Node` objects,
GC death). You need a **tensor-level** tape, and it should land immediately after the bigram —
hand-roll backward exactly once, then never again.

### 2. "Validate to the last float" isn't achievable

Different summation order and FMA contraction mean CPU and GPU will never match bit-exactly.
Decide tolerances now, and make the CPU deliberately **more** accurate (fp32 in/out, fp64
accumulator) so that when the two disagree, the CPU is right by construction.

### 3. Determinism is a phase-2 requirement built in phase 1

Own the RNG. The highest-value phase-2 test isn't "diff one matmul" — it's *"run 100 seeded
training steps on both backends, loss curves agree to ~1e-5."* That test only exists if init and
batch sampling are pure functions of a seed.

### 4. Gradient checking was missing

Finite differences against the tape is the single highest-value test in the project.

### 5. Batched matmul was missing

2D matmul plus row-flattening covers linear layers but **not** QKᵀ and attn·V. You need
`(B, M, K) @ (B, K, N)`. Discovering this at rung 6 means reopening the seam.

### 6. Config, metrics, checkpointing were missing

Ten lines each in phase 1; painful to retrofit and mandatory before any multi-day GPU run.

### 7. Phase 2 steps 1–2 underestimate the real problem

The hard part isn't the matmul kernel, it's the **dispatch model**. One dispatch + fence per op
and you will be slower than your CPU backend. The architecture — sub-allocated buffers, all ops
for a step recorded into one command buffer, one submit, one fence, host readback only for
logging and sampling — is a decision that belongs *before* kernel #1.

### 8. The 100M target needs a reality check

- **Memory:** 100M params fp32 = 400 MB weights + 400 MB grads + 800 MB Adam moments =
  **1.6 GB before a single activation**. On a 4 GB card with a desktop compositor alive, that
  leaves roughly 2–2.5 GB for activations, workspace, and staging.
- **Compute:** ~6·N·D FLOPs. 100M params × 2B tokens ≈ 1.2e18. RX 570 peak is ~5 TFLOP/s fp32;
  a good hand-written tiled GLSL matmul reaches maybe 25–40% of peak, and whole-model
  utilization (attention, softmax, norms are memory-bound) lands nearer 15–25% → **~1 TFLOP/s
  effective, i.e. ~14 days of continuous compute** for one Chinchilla-ish run.
- **Therefore:** make **25–50M params the "it works and it's good" milestone**, and treat 100M
  as the memory-management exercise it actually is.
- Polaris supports fp16 *storage* but not packed 2× fp16 math — halving weight memory buys
  space, not speed.
- ROCm dropped Polaris years ago. That's not a limitation of this plan, it's the *justification*
  for it: Vulkan compute is the only serious path on this card.

### 9. tiny-shakespeare is a phase-1 dataset only

~1.1M chars; a 30M-param model memorizes it in minutes. Phase 2 needs a real corpus and
byte-level BPE (~8–16k vocab), which also shrinks embedding and softmax cost. Put the tokenizer
behind an interface in phase 1 so the swap is a config change.

### 10. Added phase 1.5: a second CPU backend

Before betting on Vulkan, implement a multithreaded, cache-blocked CPU backend behind the freshly
extracted interface. It's a cheap dress rehearsal that proves the seam, and it makes every
phase-1 experiment 4–8× faster.

---

## Roadmap overview

### Phase 0 — Foundations
Project skeleton, owned RNG, config objects, metrics sink, test harness. No ML.

### Phase 1 — CPU model (~1–10M params)
1. Data pipeline
2. Tensor + ops (the seam)
3. Bigram, hand-rolled backward — *first generated text*
4. Autograd tape + gradient check — *bigram rewritten, loss curve must match rung 3*
5. MLP + AdamW + training infrastructure
6. Single-head causal self-attention
7. Multi-head + output projection
8. Full pre-norm block, stacked ×N
9. Training quality: warmup + cosine LR, grad clipping, eval split, checkpointing,
   top-k/temperature sampling
10. Consolidation: extract `Backend`, freeze golden tests

> **Rung discipline:** every rung generates text end-to-end before the next one is started.

### Phase 1.5 — Second CPU backend
Multithreaded + cache-blocked. Proves the seam works before taking on Vulkan risk.

### Phase 2 — Vulkan (~25–100M params)
1. Bring-up: instance with validation layers, device selection, compute queue, VMA allocator,
   `FloatArray` roundtrip
2. **Dispatch architecture** — command buffer strategy, descriptor management, sync model.
   Before any kernel.
3. Kernel 1: matmul, naive → tiled (LDS) → register-blocked, validated at each stage
4. Port remaining primitives one at a time, each dual-run against its CPU twin
5. Full GPU training step; 100-step seeded loss curve vs CPU
6. Memory campaign: gradient accumulation, activation checkpointing, fused Adam
7. Scale: bigger corpus, BPE, 25 → 50 → 100M

---

## Step 0 — Skeleton and determinism

- [ ] **0.1** Gradle/Kotlin JVM project, single module, packages
      `data / tensor / ops / autograd / model / train`.
- [ ] **0.2 Own the RNG.** `class Rng(seed: Long)` — PCG32 or xorshift128+, with `nextFloat()`,
      `nextInt(bound)`, `nextGaussian()` (Box–Muller, cache the second value).

  > Not `kotlin.random.Random`: you want a bit sequence you control, can reproduce across JVM
  > versions, and could reimplement in GLSL later if you ever add dropout.

- [ ] **0.3** `ModelConfig` / `TrainConfig` data classes. No numeric literals at call sites.
      Serialize the config into every checkpoint.
- [ ] **0.4** Metrics sink appending `step,loss,valLoss,tokensPerSec,elapsedMs` to CSV.
- [ ] **0.5** `assertAllClose(expected, actual, rtol, atol)` using `|a-b| <= atol + rtol*|b|`,
      reporting the **worst offending index** and both neighbourhoods.

  > Write it now — it is your debugging surface for the next year.

---

## Step 1 — Data pipeline

- [ ] **1.1** Vendor `input.txt` into the repo; assert a checksum on load. Never download at
      runtime.
- [ ] **1.2** `Vocab`: sorted distinct chars + an `IntArray(65536)` lookup indexed by char code
      (256 KB, faster and simpler than a `HashMap<Char, Int>`).
- [ ] **1.3** `encode(s: String): IntArray` / `decode(ids: IntArray): String`, round-trip
      asserted over the **entire** corpus. If this fails, nothing downstream can work.
- [ ] **1.4** Split 90/10 **before** any sampling exists, so val can't leak.
- [ ] **1.5** Batch sampler: `getBatch(split, batchSize, blockSize, rng): Batch` returning
      `x: IntArray(batchSize * blockSize)` and `y`, row-major, with
      `y[b, t] = data[start_b + t + 1]` and `start_b ∈ [0, n - blockSize - 1]`.
      That `-1` is the off-by-one.

  > **Keep tokens as `IntArray`, not `Tensor`.** Token ids are indices into an embedding table,
  > not numbers you do arithmetic on. Making them floats is the classic early mistake and it
  > poisons the embedding backward, which is a scatter-add over integer indices.

- [ ] **1.6** Determinism test: same seed → byte-identical batches. Pass `Rng` explicitly; no
      global generator anywhere in the project.

---

## Step 2 — Tensor

- [ ] **2.1** `class Tensor(val shape: IntArray, val data: FloatArray)` with
      `require(shape.product() == data.size)`.
- [ ] **2.2** Compute `strides` once at construction even if only contiguous row-major is
      supported — you'll want them the first time you transpose. Keep an inlined
      `at(r, c) = data[r * cols + c]` fast path for the 2D case you live in.
- [ ] **2.3** Factories: `zeros`, `full`, `randn(shape, std, rng)`, `fromRows`.

  > **Init scaling is not a TODO.** Use `std = 0.02` for embeddings and linear weights, and scale
  > residual-projection weights by `1 / sqrt(2 * nLayer)`. Bad init is a silent trainer-killer
  > that presents as a bug somewhere else entirely.

- [ ] **2.4** Grid `toString()` truncating past 8 rows/cols, plus `stats()` →
      min/max/mean/std/**nanCount/infCount**.

  > The NaN counter earns its keep the first time loss goes to NaN at step 300.

- [ ] **2.5** `expectShape(vararg s: Int)` helper, used liberally.

---

## Step 3 — Ops (the seam)

> **Rule:** one file, `CpuOps.kt`, all top-level functions. **No `FloatArray` arithmetic anywhere
> else in the codebase, ever.** That rule is greppable, and it is the entire phase-2 strategy.

- [ ] **3.1** `matmul(a, b)` — 2D. Write **two** versions:
  - `matmulRef`: `i, j, k` loop order with a **`double` accumulator**. Slow. This is the oracle.
  - `matmul`: `i, k, j` with `aik` hoisted out of the inner loop — sequential access on both `b`
    and `out`, typically 3–5× faster for free, but it accumulates into the output array so it
    can't use a double accumulator.

  Test `matmul` against `matmulRef`. You've now practiced the exact oracle pattern phase 2
  depends on, months early, at zero cost.

- [ ] **3.2** Golden test: hand-computed 2×3 @ 3×2. Keep forever.
- [ ] **3.3** `matmulBatched`: `(B, M, K) @ (B, K, N) -> (B, M, N)`. Attention needs it.
- [ ] **3.4** `transA` / `transB` flags rather than materializing transposes.

  > Backward needs `aᵀ @ g` and `g @ bᵀ`, and materializing doubles memory traffic. Ugly,
  > correct, and exactly what the GLSL kernel will want as specialization constants.

- [ ] **3.5** Elementwise: `add`, `addInPlace`, `mul`, `scale`, `addBias`, and `sumRows`
      (which is `addBias`'s gradient — write them as a pair).
- [ ] **3.6** `softmaxRows` — **subtract the row max before exp**, non-negotiable;
      `exp(89f)` is already `inf`.
- [ ] **3.7** `rmsNorm(x, gamma, eps)` returning the cached `rstd`.

  > RMSNorm over LayerNorm here: one fewer parameter, materially simpler backward, and it's what
  > modern models use. Only pick LayerNorm if bit-matching GPT-2 matters to you.

- [ ] **3.8** `gelu` (tanh approximation) or `silu`.
- [ ] **3.9** `embeddingLookup(table, ids)` — gather rows.

  > Its backward is **scatter-add**, and it's the one op whose GPU port is genuinely awkward
  > (atomics, or sort-then-segment-reduce). Flag it in a comment now.

- [ ] **3.10** `crossEntropyFromLogits(logits, targets)` returning `(loss, dLogits)` **fused**.

  > Compute via log-softmax; never softmax-then-log. The fused backward is the two-line closed
  > form `dLogits = (softmax(logits) - onehot(targets)) / N`, which skips an intermediate tensor
  > and a whole family of numerical problems.

- [ ] **3.11** Forward test per op against a hand-computed small case.

At the end of Step 3 you have: data you can batch, a tensor type, and tested ops — exactly enough
to build the bigram.

---

## Step 4 — Bigram with hand-rolled backward

- [ ] **4.1** Model is a single `Tensor(vocabSize, vocabSize)`; forward is a row gather.
- [ ] **4.2** Backward by hand: `dW[id] += dLogits[row]`. Ten lines. Do it once so the tape is
      never magic to you.
- [ ] **4.3** SGD, 1000 steps.

  > **Sanity anchor: step-0 loss must be ≈ ln(65) = 4.174.** If it isn't, your init or your loss
  > is wrong — stop and fix it before anything else. This check ("expected initial loss =
  > ln(vocabSize)") applies to every model you build after this one, so internalize it now.
  > Expect to land around 2.45.

- [ ] **4.4** Sampler: multinomial from softmax using your `Rng`, fed back autoregressively,
      500 chars. The output will be gibberish with correct letter frequencies and plausible
      spacing. That's success.
- [ ] **4.5** **Freeze the first-100-step loss curve to a golden file.** Step 5 has to reproduce
      it.

---

## Step 5 — The autograd tape

The hard one.

- [ ] **5.1 Shape:**

  ```kotlin
  class Node(val value: Tensor, val requiresGrad: Boolean) {
      var grad: Tensor? = null
      var backward: (() -> Unit)? = null
  }
  ```

  Each differentiable op takes `Node`s, calls the layer-2 function on `.value`, allocates the
  output `Node`, and registers a closure that reads `out.grad!!` and accumulates into its inputs.

- [ ] **5.2 Accumulate, never assign.**
      `parent.grad = parent.grad?.let { add(it, contrib) } ?: contrib`

  > If you assign, **residual connections silently drop half their gradient**. The model still
  > trains, just worse, and there is no error message. This is bug #1 in every from-scratch
  > autograd ever written.

- [ ] **5.3 Ordering.** Skip the DFS topological sort: maintain an append-only
      `tape: MutableList<Node>` in creation order and walk it backwards.

  > Forward execution order *is* a valid topological order, so this is simpler, allocation-free,
  > and impossible to get subtly wrong.

- [ ] **5.4 Every broadcast has a sum in its shadow.** `addBias` broadcasts `(1, N)` across
      `(M, N)`; its backward must `sumRows` the incoming gradient back down to `(1, N)`.

  > Bug #2, and it fails quietly because the shapes still look plausible if you're careless.

- [ ] **5.5 Lifecycle.** Clear the tape and null all grads after `optimizer.step()`.

  > Forget, and memory grows linearly with step count — you'll OOM around step 4000 and it will
  > read exactly like a leak in your data loader.

- [ ] **5.6 `noGrad { }` scope** — a flag that makes ops skip closure registration.

  > Without it, generating 500 tokens builds a 500-node tape you never free, on every sample call.

- [ ] **5.7 Gradient check.** Tiny config (vocab 7, nEmbd 4, blockSize 3, batch 2). For each
      parameter tensor, sample ~20 entries and compare the analytic gradient to a central
      difference `(L(θ+ε) − L(θ−ε)) / 2ε`.

  > **The subtlety: fp32 central differences are nearly useless.** With ε = 1e-3 in fp32 you get
  > maybe 1e-2 relative agreement on a good day, which catches sign errors and missing terms but
  > hides scaling bugs — exactly the ones that cost you a week.
  >
  > Two options: **(a)** write a test-only `DoubleTensor` and a fp64 path for the tiny model,
  > giving a real threshold of relative error < 1e-5; or **(b)** stay in fp32 and accept < 1e-2
  > as passing. **Do (a).** The duplication is maybe 150 lines and it's the difference between
  > "my gradients are probably right" and "my gradients are right."
  >
  > Use relative error `|a−b| / max(|a|, |b|, 1e-8)`, and check *every* parameter tensor
  > including biases and norm gains — the ones you skip are the ones that are broken.

- [ ] **5.8** Rewrite the bigram on the tape. Its first-100-step loss curve must match Step 4.5's
      golden file to ~1e-5.

  > That test is what buys you the right to trust the tape for the rest of the project.

---

## Step 6 — MLP and real training infrastructure

- [ ] **6.1** `Module` interface exposing `parameters(): List<Node>`.
      `Linear(inF, outF)` = `matmul` + `addBias`.
- [ ] **6.2** Token embedding + learned positional embedding `(blockSize, nEmbd)`, summed.
- [ ] **6.3 Reshape `(B, T, C)` ↔ `(B*T, C)` is free** for contiguous row-major tensors — same
      `FloatArray`, different shape. Make `reshape` a zero-copy view with an `isView` flag that
      all in-place ops `require(!isView)`.

  > You'll do this reshape constantly; paying a copy each time is a real cost, and unguarded
  > views are a real bug.

- [ ] **6.4** Model: embed → reshape → `Linear(C, 4C)` → GELU → `Linear(4C, vocab)` → fused CE.
- [ ] **6.5 AdamW.** Two `FloatArray`s per parameter (m, v); bias correction `m̂ = m/(1−β₁ᵗ)`,
      `v̂ = v/(1−β₂ᵗ)`; update `θ -= lr * (m̂/(√v̂ + ε) + wd·θ)`.

  > The decoupling is the point: the `wd·θ` term sits **outside** the moment estimates. Fold it
  > into the gradient instead and you've written Adam-with-L2, which is measurably worse and
  > looks identical in a code review.
  >
  > Also: **build the `decay` / `noDecay` parameter groups now** — biases, norm gains, and
  > embeddings shouldn't be decayed, and retrofitting grouping means touching every module
  > you've written.

- [ ] **6.6 Overfit a single batch.** Train on one fixed batch for 200 steps; loss must approach
      ~0.

  > The highest-yield integration test in machine learning. It catches wrong backward passes,
  > unattached parameters, optimizers silently not updating something, and stale gradients. If it
  > plateaus, something is broken and no amount of hyperparameter tuning will save you.
  > **Put it in CI.**

- [ ] **6.7** LR warmup + cosine decay, gradient clipping by global norm at 1.0, periodic
      val-loss eval under `noGrad`, checkpoint save/load (weights + optimizer state + config +
      RNG state + step).

---

## The next cliff: attention shapes

Worth reading before you get there, because it's where the seam gets stress-tested.

```text
x:    (B*T, C)
qkv = Linear(C, 3C)(x)        → (B*T, 3C), split into q, k, v each (B*T, C)
reshape (B, T, H, hs) → transpose to (B, H, T, hs) → flatten (B*H, T, hs)
att = matmulBatched(q, k, transB = true) * (1 / sqrt(hs))   → (B*H, T, T)
mask upper triangle to -inf, softmaxRows
out = matmulBatched(att, v)   → (B*H, T, hs)
transpose back → (B*T, C) → output projection
```

Three things to know going in:

- **The head transpose is not free.** `(B, T, H, hs) → (B, H, T, hs)` is genuine data movement in
  a contiguous-only world, and its backward is the inverse permutation. Write it as a real op
  with a real backward and test it in isolation — a wrong axis permutation produces output that
  trains, slowly, to a mediocre loss.
- **Use `-inf` for the mask, not `-1e9`.** With row-max subtraction, `-inf` is safe here because a
  causal row always retains at least one valid entry, so you never get an all-masked row and
  never get `0 * inf = NaN`. `-1e9` "works" but leaks a small probability mass into the future,
  which is a real, subtle correctness bug.
- **The `1/√hs` scale exists in the backward too.** It's just a scalar multiply, which is
  precisely why it gets forgotten. Your gradient check will catch it — if you remembered to
  gradient-check attention.

---

## Invariants worth pinning to the wall

1. All arithmetic goes through `CpuOps.kt`. No exceptions, ever.
2. Every gradient **accumulates**; nothing assigns.
3. Every broadcast forward has a sum backward.
4. Step-0 loss ≈ `ln(vocabSize)`.
5. A single batch must overfit to ~0.
6. Same seed → same batches → same loss curve.
7. `matmulRef` (fp64 accumulator) is the oracle; everything else is checked against it.
8. Golden files are never regenerated to make a test pass.
