# COCO Filtered Dataset for Class Balancing

## Overview

This directory contains a filtered subset of the COCO 2017 dataset, restricted to the following object detection classes:

- **car**
- **person**
- **bicycle**

The filtered COCO data is converted into YOLO format and is intended to be used as an auxiliary dataset to balance class distributions in the Huawei and PascalRaw datasets used in this project.

**The goal is not to replace Huawei or PascalRaw, but to supplement underrepresented classes (especially pedestrian and bicycle) during YOLO-based object detection training.**

---

## Objective

In the Huawei and PascalRaw datasets, object class distributions are highly imbalanced:

- **car** instances are abundant
- **person** and **bicycle** instances are comparatively scarce

This imbalance causes:

- Domination of classification and box regression loss by majority classes
- Reduced recall and localization accuracy for minority classes

To address this, we selectively introduce COCO data to:

1. Increase the number of pedestrian and bicycle instances
2. Improve training stability and recall for minority classes
3. Preserve Huawei + PascalRaw as the primary domain of interest

**COCO is used as a balancing patch, not as a primary training dataset.**

---

## Why COCO (and Why Filtering)

COCO provides:

- High-quality bounding box annotations
- Diverse real-world scenes
- Standardized detection benchmarks

However, the full COCO dataset is:

- Large
- Domain-diverse
- Easily dominant if added without control

Therefore, we filter COCO to only the relevant detection classes and only include the minimum amount needed to improve class balance.

---

## Filtering Strategy

### Classes Kept

Only the following COCO categories are retained:

- `car`
- `person`
- `bicycle`

All other categories are ignored.

### Image Selection Logic

Images are selected if they contain **at least one instance** of any target class (union logic, not intersection).

**Formally:**

```
image ∈ dataset if (car OR person OR bicycle) appears in image
```

This avoids over-filtering and ensures realistic detection distributions.

### Dataset Split Choice

**Start with `val2017` only (current setup):**

- COCO `val2017` contains ~5,000 images total
- After filtering, ~2,889 images remain
- Small enough to iterate quickly
- Fast to download and preprocess
- Sufficient to validate the full pipeline:
   - Filtering
   - YOLO label generation
   - Training
   - Evaluation metrics

This is intentionally chosen as the initial balancing dataset.

**When to add `train2017` later:**

The much larger `train2017` split should only be included if necessary.

Add `train2017` only when:

- The final target class balance cannot be achieved with Huawei + PascalRaw + COCO `val2017`
- Pedestrian and/or bicycle instances remain underrepresented
- Additional diversity is required for final experiments or ablations

**Caution:**

- `train2017` is large (storage and preprocessing cost)
- It can easily dominate the combined dataset if not carefully capped
- Overuse may introduce domain mismatch relative to driving-oriented data

---

## Output Structure

After filtering and conversion, the dataset is organized as:

```
coco_filtered_3cls/
├── annotations/
│   └── instances_val2017.json
├── images/
│   └── val2017/                # symlinks to COCO images
├── labels/
│   └── val2017/                # YOLO-format labels
├── classes.txt
└── dataset.yaml
```

- Images are **symlinked** (not copied) to save disk space
- Labels follow standard YOLO format:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```

---

## Intended Usage in Training

The filtered COCO dataset is used to:

- Supplement Huawei and PascalRaw during training
- Reduce object-level class imbalance
- Improve minority-class recall and localization

It is **not** intended to:

- Dominate the training distribution
- Replace domain-specific datasets
- Introduce unrelated object categories

---

## Notes

- COCO `person` includes many non-driving contexts (indoor, sports, etc.). The amount of COCO data injected should therefore remain controlled.
- All preprocessing is designed to be reproducible on the computing cluster with no internet access during training.

---

## Summary

This COCO filtering pipeline provides a controlled, reproducible way to improve class balance for YOLO detection while preserving the integrity and domain relevance of the Huawei and PascalRaw datasets.
