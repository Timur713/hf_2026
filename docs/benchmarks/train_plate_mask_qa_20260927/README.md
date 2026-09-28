# Visual QA of the plate control — 27.09.2026

**The current geometric control is not a full plate-area fill.** Inspection of
eight regularly spaced train fold0 queries found obvious misses or incomplete
coverage. In particular, the side-view van's plate remains on the right while
the rectangle covers a door; the Toyota's plate remains above the rectangle;
the blue SUV's plate remains above it. Some overlap in the white crossover
does not cover the full blurred plate region either.

These are deterministic train examples, not test inspection or error-selected
cases. The sample is qualitative and small: no dataset-wide coverage percentage
is claimed. The rectangle covers a vehicle-relative estimate, not an annotated
or detected plate. Therefore a small mAP drop under this control **does not
prove compliance with an organizer control that covers the actual plate**.

Reproduce from the repository root:

```bash
PYTHONPATH=. python scripts/preview_train_plate_mask.py \
  --data-root dataset --out outputs/train_mask_qa
```

The exact selected IDs, BBoxes and mask coordinates are in `selection.json`.
Each row shows the original train vehicle on the left and the control on the
right. Source images and the shared preprocessing function are not modified.

![Train sample 1](page1.png)
![Train sample 2](page2.png)

Next gate: evaluate a train-only subset with verified actual plate rectangles,
including the blurred border, in both embedder and verifier. A broad lower-body
occlusion may be a separate stress test, but must not be labelled precise
plate masking. Changes to training or shared C3 need agreement with track A;
no unvalidated preprocessing or test-set tuning was introduced here.
