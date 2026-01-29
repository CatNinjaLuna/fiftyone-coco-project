# FiftyOne COCO Project

This project uses **FiftyOne** with a **micromamba environment** for COCO dataset loading and visualization.

## Prerequisites

The following setup has been completed:

- ✅ Micromamba is installed
- ✅ Shell is initialized for zsh
- ✅ Environment `fiftyone` exists with FiftyOne installed

## Getting Started

### Activating the Environment

**Option 1: Terminal activation (recommended)**

```bash
micromamba activate fiftyone
```

Verify the installation:

```bash
python -c "import fiftyone as fo; print(fo.__version__)"
```

Run the COCO loader:

```bash
python load_coco_fiftyone.py
```

**Option 2: One-liner (no activation required)**

```bash
micromamba run -n fiftyone python load_coco_fiftyone.py
```

### VS Code Python Interpreter

To avoid `ModuleNotFoundError`, configure VS Code to use the correct Python interpreter:

1. Press `Cmd + Shift + P` (macOS) or `Ctrl + Shift + P` (Windows/Linux)
2. Select **Python: Select Interpreter**
3. Choose: `~/.local/share/mamba/envs/fiftyone/bin/python`

## Troubleshooting

| Issue                                             | Solution                                                                                                                                                                                                         |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Terminal prompt shows `(base)`                    | Wrong environment activated. Run `micromamba activate fiftyone`                                                                                                                                                  |
| `ModuleNotFoundError: No module named 'fiftyone'` | **Always activate the fiftyone environment before running the script!** Use `micromamba activate fiftyone` then `python load_coco_fiftyone.py`, OR use `micromamba run -n fiftyone python load_coco_fiftyone.py` |
| `import fiftyone` fails                           | VS Code interpreter not set to `fiftyone` environment                                                                                                                                                            |
| Changes don't take effect                         | Restart VS Code after environment or shell changes                                                                                                                                                               |

### Important: Running the Script

To avoid module import errors, **always run the script within the fiftyone environment**:

**Method 1: Activate first (recommended)**

```bash
micromamba activate fiftyone
python load_coco_fiftyone.py
```

**Method 2: Use micromamba run**

```bash
micromamba run -n fiftyone python load_coco_fiftyone.py
```

## Project Structure

```
.
├── load_coco_fiftyone.py         # Quickstart dataset loader (200 samples)
├── load_coco_full_fiftyone.py    # Full COCO 2017 validation analysis
├── load_coco_all_tasks.py        # Multi-task analysis (detection/segmentation/keypoints)
├── download_coco_filtered.py     # ⭐ PRIMARY: HPC-ready YOLO format downloader
├── README.md                      # This file
├── DOWNLOAD_INSTRUCTIONS.md       # Detailed HPC usage guide
├── classes_result_200.txt         # Analysis: Quickstart subset
├── coco_full_classes_result.txt   # Analysis: Validation set (2,889 images)
├── coco_all_tasks_result.txt      # Analysis: All COCO tasks
└── quickstart/                    # Quickstart dataset directory
    ├── info.json
    ├── metadata.json
    ├── samples.json
    └── data/
```

## 🎯 For Dataset Balancing (Car/Person/Bicycle)

**Focus on DETECTION task only** - it contains all the annotations you need!

### Primary Script: `download_coco_filtered.py`

This script is designed for your use case:

- ✅ Downloads **detection task only** (not segmentation/keypoints)
- ✅ Filters for **car, person, bicycle** classes
- ✅ Converts to **YOLO format** (ready for training)
- ✅ **HPC-ready** (no FiftyOne dependency, uses pycocotools)
- ✅ Resumable downloads

**Quick Start:**

```bash
# 1. Test locally with validation set (2,889 images, ~1-2GB)
python download_coco_filtered.py --split val2017 --output ./coco_filtered

# 2. On HPC: Download full training set (~40-60K images, 15-20GB)
python download_coco_filtered.py --split train2017 --output ./coco_filtered
```

### COCO Detection Task Statistics (Validation Set)

Based on analysis in `coco_full_classes_result.txt`:

- **Total samples**: 2,889 images
- **Total detections**: 27,260 objects
- **Target class detections**: 13,252 (48.61% of all detections)
   - person: 11,004 (40.37%)
   - car: 1,932 (7.09%)
   - bicycle: 316 (1.16%)

**Images per class:**

- Images with person: 2,693
- Images with car: 535
- Images with bicycle: 149

### Why Detection Task Only?

Analysis (`coco_all_tasks_result.txt`) confirms:

- **Detection task**: Contains all car/person/bicycle annotations ✅
- **Keypoints task**: Only human pose annotations (no object labels) ❌
- **Segmentation task**: Same objects as detection but with masks (not needed for YOLO) ❌

**Next Steps:**

1. Review `DOWNLOAD_INSTRUCTIONS.md` for HPC setup
2. Use `download_coco_filtered.py` to get detection data in YOLO format
3. Merge with Huawei + PascalRAW datasets for balanced training
