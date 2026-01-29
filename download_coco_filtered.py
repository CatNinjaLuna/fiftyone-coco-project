"""
Download and filter COCO dataset for car, person, and bicycle classes.
Converts annotations to YOLO format for dataset balancing.
"""

import os
import json
import urllib.request
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse


def download_coco_annotations(split, output_dir):
    """Download COCO annotation file if not exists."""
    annotations_dir = Path(output_dir) / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    annotation_file = annotations_dir / f"instances_{split}.json"
    
    if not annotation_file.exists():
        print(f"Downloading COCO {split} annotations...")
        url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        print(f"Please download annotations manually from: {url}")
        print(f"Extract and place instances_{split}.json in {annotations_dir}")
        return None
    
    return str(annotation_file)


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] normalized."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]


def download_filtered_coco(split="val2017", output_dir="./coco_filtered", target_classes=None):
    """
    Download COCO images filtered by specific classes and convert to YOLO format.
    
    Args:
        split: 'val2017' or 'train2017'
        output_dir: Directory to save filtered dataset
        target_classes: List of class names to filter (default: ['car', 'person', 'bicycle'])
    """
    if target_classes is None:
        target_classes = ['car', 'person', 'bicycle']
    
    output_path = Path(output_dir)
    images_dir = output_path / "images" / split
    labels_dir = output_path / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations
    annotation_file = download_coco_annotations(split, output_dir)
    if annotation_file is None:
        # Create dummy annotation file path for manual download
        annotation_file = f"{output_dir}/annotations/instances_{split}.json"
    
    if not Path(annotation_file).exists():
        print(f"\nERROR: Annotation file not found at {annotation_file}")
        print("\nTo download COCO annotations:")
        print("1. wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print("2. unzip annotations_trainval2017.zip")
        print(f"3. Move instances_{split}.json to {output_dir}/annotations/")
        return
    
    print(f"Loading COCO annotations from {annotation_file}...")
    coco = COCO(annotation_file)
    
    # Get category IDs for target classes
    cat_ids = coco.getCatIds(catNms=target_classes)
    print(f"\nTarget classes and their IDs:")
    for cat_id in cat_ids:
        cat_info = coco.loadCats(cat_id)[0]
        print(f"  {cat_info['name']}: {cat_id}")
    
    # Create class mapping for YOLO (0-indexed)
    coco_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted(cat_ids))}
    yolo_to_name = {idx: coco.loadCats(cat_id)[0]['name'] for cat_id, idx in coco_to_yolo.items()}
    
    # Save class names for reference
    classes_file = output_path / "classes.txt"
    with open(classes_file, 'w') as f:
        for idx in sorted(yolo_to_name.keys()):
            f.write(f"{yolo_to_name[idx]}\n")
    print(f"\nClass names saved to {classes_file}")
    
    # Get all images containing target classes
    img_ids = coco.getImgIds(catIds=cat_ids)
    print(f"\nFound {len(img_ids)} images containing target classes")
    
    # Statistics
    stats = {name: 0 for name in target_classes}
    downloaded_count = 0
    skipped_count = 0
    
    # Download images and create YOLO annotations
    print(f"\nDownloading images and creating YOLO annotations...")
    for img_id in tqdm(img_ids, desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_path = images_dir / img_filename
        
        # Skip if image already exists
        if img_path.exists():
            skipped_count += 1
        else:
            try:
                # Download image
                url = img_info['coco_url']
                urllib.request.urlretrieve(url, str(img_path))
                downloaded_count += 1
            except Exception as e:
                print(f"\nError downloading {img_filename}: {e}")
                continue
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        
        # Convert to YOLO format
        label_filename = img_filename.replace('.jpg', '.txt')
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in anns:
                if ann['category_id'] in cat_ids:
                    # Update statistics
                    cat_name = coco.loadCats(ann['category_id'])[0]['name']
                    stats[cat_name] += 1
                    
                    # Convert bbox to YOLO format
                    bbox = ann['bbox']
                    yolo_bbox = convert_bbox_to_yolo(
                        bbox, 
                        img_info['width'], 
                        img_info['height']
                    )
                    
                    # YOLO format: class_id x_center y_center width height
                    yolo_class_id = coco_to_yolo[ann['category_id']]
                    f.write(f"{yolo_class_id} {' '.join(map(str, yolo_bbox))}\n")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Download Summary")
    print(f"{'='*50}")
    print(f"Total images: {len(img_ids)}")
    print(f"Downloaded: {downloaded_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"\nAnnotation counts:")
    for class_name, count in stats.items():
        print(f"  {class_name}: {count}")
    print(f"\nData saved to: {output_path}")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  Classes: {classes_file}")
    
    # Create dataset.yaml for YOLO
    yaml_content = f"""# COCO Filtered Dataset
path: {output_path.absolute()}
train: images/{split}
val: images/{split}

nc: {len(target_classes)}
names: {target_classes}
"""
    
    yaml_file = output_path / "dataset.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    print(f"\nYOLO dataset config saved to: {yaml_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download filtered COCO dataset")
    parser.add_argument('--split', type=str, default='val2017', 
                        choices=['val2017', 'train2017'],
                        help='Dataset split to download')
    parser.add_argument('--output', type=str, default='./coco_filtered',
                        help='Output directory for filtered dataset')
    parser.add_argument('--classes', nargs='+', default=['car', 'person', 'bicycle'],
                        help='Classes to filter')
    
    args = parser.parse_args()
    
    print(f"Downloading COCO {args.split} filtered for: {args.classes}")
    download_filtered_coco(
        split=args.split,
        output_dir=args.output,
        target_classes=args.classes
    )
