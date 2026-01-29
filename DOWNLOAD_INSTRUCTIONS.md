# Download Filtered COCO Dataset

This script downloads COCO images filtered for specific classes (car, person, bicycle) and converts annotations to YOLO format for dataset balancing.

## Requirements

```bash
pip install pycocotools tqdm
```

## Usage

### Step 1: Download COCO Annotations (one-time)

```bash
# Download and extract annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
mkdir -p coco_filtered/annotations
mv annotations/instances_*.json coco_filtered/annotations/
```

### Step 2: Run the Script

**For validation set (~2-3K images, 1-2GB):**

```bash
python download_coco_filtered.py --split val2017 --output ./coco_filtered
```

**For training set (~40-60K images, 15-20GB):**

```bash
python download_coco_filtered.py --split train2017 --output ./coco_filtered
```

**Custom classes:**

```bash
python download_coco_filtered.py --split val2017 --classes car person bicycle truck
```

## Output Structure

```
coco_filtered/
├── images/
│   ├── val2017/          # or train2017
│   │   ├── 000000000139.jpg
│   │   └── ...
├── labels/
│   ├── val2017/          # YOLO format annotations
│   │   ├── 000000000139.txt
│   │   └── ...
├── annotations/
│   └── instances_val2017.json
├── classes.txt           # Class names in order
└── dataset.yaml          # YOLO config file
```

## YOLO Format

Each `.txt` file contains annotations in YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized to [0, 1].

## Class Mapping

The script creates a 0-indexed class mapping:

- 0: bicycle
- 1: car
- 2: person

## For HPC Usage

1. Copy this script to your HPC cluster
2. Run in a job or interactive session with internet access
3. Download annotations first (they're small, ~250MB)
4. Run the script to download images

### Example SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=coco_download
#SBATCH --time=4:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=4

# Load Python environment
module load python/3.10

# Install dependencies
pip install pycocotools tqdm --user

# Download COCO filtered data
python download_coco_filtered.py --split train2017 --output /scratch/$USER/coco_filtered
```

## Next Steps: Merging with Your Datasets

After downloading, you can:

1. Count class distributions in all datasets (COCO + Huawei + PascalRAW)
2. Calculate how many samples to use from each dataset for balance
3. Merge datasets using symlinks or copying
4. Use class weights in YOLO if needed

Example class counting:

```python
from pathlib import Path
from collections import Counter

def count_classes(labels_dir):
    counts = Counter()
    for label_file in Path(labels_dir).glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                class_id = int(line.split()[0])
                counts[class_id] += 1
    return counts

# Count in each dataset
coco_counts = count_classes("coco_filtered/labels/train2017")
huawei_counts = count_classes("huawei/labels")
pascal_counts = count_classes("pascalraw/labels")
```
