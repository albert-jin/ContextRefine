# ContextRefine

**Topology-Aware Crack Segmentation via Frozen Context Features and Consistency Regularization**

ContextRefine combines frozen local and contextual representations with a trainable residual refinement head. This repository contains the v8 implementation used for the archived TopoMortar and Crack500 experiments, together with configurations, dataset manifests, evaluation results, and reproduction instructions. The two datasets have separate training entry points but share the context refinement method.

## Results

All results below use seed 0. Metrics are arithmetic means of per-image scores. Dice, foreground IoU, and clDice are percentages; lower Betti errors are better.

| Dataset | Test images | Dice | Foreground IoU | clDice | Betti-0 error | Betti-1 error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TopoMortar | 350 | 82.1688 | 71.8639 | 88.3723 | 11.4543 | 9.0714 |
| Crack500 | 200 | 71.1897 | 56.3653 | 81.6750 | 8.8150 | 6.5050 |

The matched ConvNeXt SCNP base achieves 81.9754% Dice on TopoMortar and 71.0372% on Crack500. ContextRefine improves these means by **0.1933 and 0.1524 percentage points**, respectively. Crack500 local refinement without DINO achieves 71.1408%, giving a matched context-versus-local difference of 0.0489 points.

The TopoMortar no-DINO control is available only on a fixed 70-image subset. ContextRefine improves over that control by 0.1430 points on the same subset; this is not a comparison over all 350 test images. Crack500 also has a fixed 50-image ablation subset whose differences are distinct from full-test means. See [`results/`](results/) for the per-image evidence and [`verified_main_results.json`](results/verified_main_results.json) for matched comparisons. The historical Crack500 score of 71.6089% belongs to a different gradient-budget method, not ContextRefine v8.

## Method

```text
RGB image
  |-- frozen ConvNeXt-Tiny U-Net --> local features (32 channels)
  |                                base logits (2 channels)
  `-- frozen DINOv2-S/14 ---------> last four layer features (1536 channels)
                                     |
                              trainable 1x1 projection (64 channels)
                                     |
[pooled local features, pooled base probabilities, aligned context]
                       98 channels
                            |
                 trainable residual head
                            |
                      delta (1 channel)

z = z_base + concat(-delta / 2, +delta / 2)
foreground = softmax(z)[1] >= 0.5
```

- A 512 x 512 input is symmetrically padded to 518 x 518 for DINO. The last four layers each have a 37 x 37 patch grid: these are different depths, not different spatial scales. Their concatenated 1536 channels are projected to 64 channels, upsampled, unpadded, and pooled to 128 x 128.
- Both backbones are frozen and remain in evaluation mode. The projection and residual head have **192,065 trainable parameters**. The no-DINO local refinement control has 93,633 effective trainable parameters; the full context model stores 54,215,731 parameters in total.
- The output layer is initialized to zero, so initial predictions equal the local base model. Version 8 uses `delta = raw`, without the confidence gate or tanh clipping used in v7.
- Paired views remain geometrically aligned. The second view receives color perturbations and local occlusion. Both views use SCNP cross-entropy plus Dice supervision:

```text
L = 0.5 * (L_SCNP(view1) + L_SCNP(view2)) + lambda(t) * L_structure_JS
lambda(t) = 0.5 * min(t / 600, 1)
```

- Fixed structure weights are derived from training ground truth. For foreground and background regions R, compute `maxfilter_3x3(skeleton(R) / (1 + EDT(R))) * R`, sum the two contributions into G, and zero the full cached image boundary. The raw weight is `w = 1 + 4 * clip(G, 0, 1)`. Weighted JS is normalized within each ground-truth class present in each image, then averaged over classes and images. These are neither Gaussian boundary weights nor learned attention maps. Test-time prediction does not use ground truth.

## Repository layout

| Location | Purpose |
| --- | --- |
| `exp_SCNP_topo_free_context_v8/research/` | TopoMortar implementation |
| `exp_SCNP_crack500_context_v8_transfer/research/` | Crack500 implementation |
| `research/context_residual.py` in each project | Frozen context and residual refinement head |
| `research/convnext_unet.py` | Local ConvNeXt model |
| `research/chroma_consistency.py` | Paired views and structure-weighted JS |
| `research/amodal_completion.py` | Occlusion augmentation |
| `research/losses.py`, `SCNP/experiments/Detectron2/loss.py` | SCNP adaptation and upstream loss |
| TopoMortar `research/train_pretrained.py` | TopoMortar training entry point |
| Crack500 `research/train_crack500_context.py` | Crack500 training and test entry point |
| Crack500 `research/crack500_data.py` | Cropping, sliding windows, and native-resolution evaluation |
| `scripts/evaluate_topo_free_context.py` | TopoMortar evaluation |
| `scripts/test_metrics.py`, `scripts/gpu_guard.py` | Metrics and GPU admission checks |
| `results/` | Archived evaluation summaries and per-image scores |
| `data/`, `auto_res_logs/` | Dataset manifests and experiment records |

Some inherited modules remain because other code imports or extends them. The entry points above define the v8 workflow; v9/v10 experiments are not included.

## Installation and required resources

```bash
git clone https://github.com/albert-jin/ContextRefine.git
cd ContextRefine
python3.12 -m venv env
source env/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_core.txt
git clone https://github.com/facebookresearch/dinov2.git upstream_dinov2
git -C upstream_dinov2 checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

Use Linux with CUDA for training. Recorded core versions are Python 3.12, PyTorch 2.9.1+cu128, TorchVision 0.24.1+cu128, MONAI 1.6.0, and NumPy 2.2.5. SciPy, scikit-image, Pillow, and PyYAML are also required. [`requirements_core.txt`](requirements_core.txt) records core dependencies, not a complete historical environment lock. These setup commands are reproduction instructions, not a claim that a fresh installation has been validated.

**Datasets, model checkpoints, and the original Python environment are not bundled.** [`CHECKPOINTS.json`](CHECKPOINTS.json) lists expected checkpoint locations, public pretrained-weight URLs, and checksums. Restore the following resources relative to the repository root:

```text
data/TopoMortar/dataset/{train,val,test}/{images,accurate}/
data/TopoMortar/dataset/splits.yaml
data/icassp_downloads/crack500/raw/...
data/icassp/crack500/cache/{train,val,test}/...
data/pretrained_weights/convnext_tiny-983f1562.pth
data/pretrained_weights/dinov2_vits14_pretrain.pth
upstream_dinov2/
exp_SCNP_topo_visibility_budget_v6/auto_res_logs/runs/topo_visibility_pair_base_s0/best.pt
exp_SCNP_crack500_context_v8_transfer/auto_res_logs/runs/crack_context_base_s0/best.pt
```

Code checks the DINO revision, clean checkout, and weight checksums. Preserve the relative directory layout. DINO uses additional LVD-142M pretraining; the local backbone uses ImageNet-1K pretraining. Historical records retain original server paths; these are provenance records and may need the original filesystem layout for exact replay.

### Dataset protocol

- **TopoMortar:** 50 training, 20 validation, and 350 test images. The test set contains one in-distribution group and six out-of-distribution groups of 50 images each.
- **Crack500:** 243 training, 49 validation, and 200 test images after excluding seven training images and one validation image from overlapping families. Derived caches for three training images are rotated to match their labels. Manifests and correction records are included under `data/` and `auto_res_logs/`.

The fixed Crack500 manifest checksum is `8044bc6505e0a82b17faa538bb718b3d3df27d4a99586dbc4eefebdef0d79ddb`. Regenerating JSON may change timestamps and checksums. Exact replay uses the original manifest and matching caches. Dataset preparation and download scripts are included under `scripts/`.

## Training

The refinement head uses batch size 8, AdamW with learning rate 3e-4 and weight decay 0.01, 3,000 steps, validation every 300 steps, cosine decay to 1% of the initial rate, gradient clipping at 12, FP16 model computation, and FP32 losses. Only the head is trained.

TopoMortar starts from an existing v6 local checkpoint whose history includes 12,000 base-training steps, 3,600 v4 steps, and 3,000 v6 steps. The final 3,000 refinement steps are not the total training cost. This package does not include the complete code for all preceding TopoMortar stages. Crack500 trains its local base independently for 12,000 steps, then trains refinement; it does not transfer TopoMortar segmentation weights.

Run from the repository root after restoring data and external models. Use new run IDs to keep the archived results intact:

```bash
python scripts/gpu_guard.py --reserve-mib 17920 --output auto_res_logs/reproduce_topo_gpu.json --launch -- \
  python exp_SCNP_topo_free_context_v8/research/train_pretrained.py \
  --method dino_context --seed 0 --steps 3000 --batch 8 --workers 4 --val-every 300 --run-id reproduce_topo_v8_s0

python scripts/gpu_guard.py --reserve-mib 18000 --output auto_res_logs/reproduce_crack_gpu.json --launch -- \
  python exp_SCNP_crack500_context_v8_transfer/research/train_crack500_context.py \
  --stage dino_context --run-id reproduce_crack_v8_s0 --steps 3000 --batch 8 --workers 4 --val-every 300 --seed 0 \
  --warm exp_SCNP_crack500_context_v8_transfer/auto_res_logs/runs/crack_context_base_s0/best.pt
```

To train the Crack500 local base, use `--stage base --steps 12000 --val-every 1200`, omit `--warm`, and choose a new run ID. Pass its checkpoint to subsequent refinement runs and retain its configuration and test lock. TopoMortar checks a fixed warm-start checksum, so an arbitrary retrained checkpoint cannot directly replace the expected warm start.

The GPU guard considers visible devices 0–3, excludes devices already at or above 70% memory use, and requires projected usage below 80% after reservation. It defers launch when resources are insufficient. Original queues are in `auto_res_logs/topo_free_context_v8_20260918/queue.json` and `auto_res_logs/crack500_context_v8_20260918/queue.json`. They contain historical paths and short diagnostic runs; use individual run configurations rather than rerunning an entire queue.

## Model selection and evaluation

**TopoMortar.** Selection requires clean validation Dice of at least 0.936 and maximizes an equally weighted combination of clean and fixed grayscale/RGB-cycle/warm-shadow validation Dice. The v8 checkpoint was selected at step 2,700. Test inputs are 512 x 512, with threshold 0.5 and no test-time augmentation or post-processing.

**Crack500.** Selection uses mean Dice on 49 validation images at native ground-truth resolution. Selected steps are 2,400 for the base, 300 for local refinement, and 2,100 for v8. Inference limits the longest side to 1,024 pixels, uses 512-pixel windows with stride 384, averages overlapping probabilities, and bilinearly resizes to native ground-truth dimensions before thresholding at 0.5. There is no test-time augmentation or post-processing.

Metrics are implemented in `scripts/test_metrics.py`. Dice, foreground IoU, and clDice are averaged per image. Foreground uses 8-connectivity and background uses 4-connectivity. Betti errors are absolute differences in topological counts, not percentage scores.

For TopoMortar, first run `scripts/evaluate_topo_free_context.py --id topo_free_context_dino_context_s0_best --validation` through the GPU guard, then evaluate the same ID without `--validation`. Restore the checkpoint and all records referenced by its test lock; exact replay may require original server records outside this source package.

For Crack500, use `train_crack500_context.py --mode test` with the stage, run ID, warm start, and required arguments from the corresponding run configuration. Original commands are in `auto_res_logs/crack500_context_v8_20260918/finalize_commands.jsonl`. Evaluation checks weights, manifests, source hashes, and validation replay. Existing evaluation outputs are not overwritten; new training runs need their own locks.

## Source verification and English localization

```bash
python verify_source.py
```

The verifier checks release files and the four archived run source inventories. Training algorithms and recorded results are unchanged. The English release translates report text in `research/finalize_transfer.py` and writes `RESULTS.md` instead of `RESULTS_ZH.md`. [`LOCALIZATION.json`](LOCALIZATION.json) records the original and translated file checksums. Archived run hashes and test locks retain their original values; verification explicitly distinguishes an exact archived match from a documented localization change.

The historical test entry points retain strict source checks. Exact replay of an old Crack500 lock therefore requires the original report script as well as the missing original resources; this release does not disable those checks. Newly trained runs record their own source hashes. The upstream `SCNP/scnp_example.py` files contain the original non-executable `loss.backward()...` placeholder and are examples, not training entry points.

## Manuscript correspondence

The archived v8 implementation differs from several descriptions in the supplied manuscript draft, including backbone names, training losses, data splits, and ablation claims. See [Manuscript correspondence](docs/manuscript-correspondence.md) before using draft tables or descriptions as a specification for this code. Recorded experiment results have not been changed to match the draft.

## Upstream code and licenses

- [SCNP](https://github.com/jmlipman/SCNP-SameClassNeighborPenalization), revision `1649403189f51c97db8629dbdf1984c184646270`. Its licenses are retained in both experiment directories.
- [TopoMortar](https://github.com/jmlipman/TopoMortar), revision `b1f31f20a41b20aa775c21078b2c432c6dcc26b4`. Reference implementations and licenses are retained under each `reference/` directory.
- [DINOv2](https://github.com/facebookresearch/dinov2), revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, Apache-2.0. Obtain it separately as described above. TorchVision weights retain their original attribution and terms.

Later development was informed by earlier test observations, so these are post-development benchmark results. Only seed 0 is archived; the reported means alone do not establish statistical significance or a general state-of-the-art claim.
