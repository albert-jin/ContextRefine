# Manuscript correspondence

The supplied draft, `ContextRefine(3).pdf`, combines actual v8 results and examples with descriptions of a different implementation. The following distinctions are retained from the original package documentation so that readers can interpret the code and evidence accurately.

| Draft location | Draft description | Archived v8 implementation or evidence |
| --- | --- | --- |
| Section 3.2; Figure 1 | DINOv2 ViT-B/14 with multiple spatial scales | Last four DINOv2-S/14 layers at the same patch resolution |
| Section 3.3; Eq. (2) | Trainable ResNet-34 | Previously trained, frozen ConvNeXt-Tiny U-Net |
| Section 3.3; Eq. (3) | Context injected at each decoder stage; alpha initialized to 0.01 | One final refinement head with a zero-initialized output layer; no such alpha |
| Eq. (4); Section 3.5 | Sigmoid and BCE | Two-class softmax and paired SCNP cross-entropy plus Dice |
| Eqs. (6)–(7) | Gaussian distance weights and a full-pixel average | Skeleton/inverse-width geometry G, w = 1 + 4G, and normalization within ground-truth classes |
| Sections 3.4–3.5 | Separate affine views compared directly; fixed lambda | Geometrically aligned paired views; lambda warm-up over 600 steps |
| Section 4.1 | 4,200 TopoMortar training images; Crack500 split 300/200 | TopoMortar 50/20/350; cleaned Crack500 243/49/200 |
| Section 4.1 | Learning rate 1e-4, weight decay 1e-4, 100 epochs, 50% sliding-window overlap | Head learning rate 3e-4, weight decay 0.01, 3,000 steps; Crack500 window 512, stride 384 |
| Table 1 | Final ContextRefine rows | Supported by this package; the other draft baseline rows are not verified by these records |
| Table 2 | Complete no-JS, uniform-JS, and full-DINO-fine-tuning ablations | Not supported by the available archives; planned or short diagnostic runs are not completed ablations |
| Figure 2; Section 4.4 | Learned weights with red/blue colors | Fixed ground-truth weights w = 1 + 4G, shown with a viridis color scale |
| Section 4.5; Figures 3–5 | ConvNeXt-Tiny/S14 cases | Consistent with v8 and the exported cases; positive single-image gains are not dataset-average gains |
| Table 3; Section 4.6 | 5.9M full-fine-tuning parameters, 30x reduction, H100/45 FPS | No corresponding measurements in the package; 192,065 is the trainable head count and does not establish inference speed |

The draft SCNP Dice values 81.69 and 70.63 are not the matched ConvNeXt baselines in this package. Consequently, draft improvements of +0.48 and +0.56 percentage points do not describe the matched comparison here. The earlier TopoMortar ResNet-34 SCNP score was 75.9538%; differences from v8 also include backbone and training changes. SCNP is a training-time loss transformation and does not justify a claim of test-time neighborhood-sampling overhead.

This document records the correspondence of the supplied draft to the archived implementation. It does not modify manuscript text or experimental results.
