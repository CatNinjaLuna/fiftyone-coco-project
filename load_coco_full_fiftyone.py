import fiftyone as fo
import fiftyone.zoo as foz

name = "coco-2017-detection-full"

# Delete existing dataset if it exists
if name in fo.list_datasets():
    print(f"Deleting existing dataset: {name}")
    fo.delete_dataset(name)

print("="*60)
print("Loading COCO 2017 Detection Dataset")
print("Filtering for: car, person, bicycle")
print("="*60)

# Try to load existing dataset first, or download if needed
try:
    # Try loading existing dataset
    dataset = fo.load_dataset(name)
    print(f"\nLoaded existing dataset: {name}")
except:
    # Download COCO 2017 detection dataset with filtering
    print("\nDownloading COCO 2017 validation set...")
    print("(This will resume if partially downloaded)")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",  # Use "train" for full training set
        label_types=["detections"],
        classes=["car", "person", "bicycle"],
        dataset_name=name,
    )
    print(f"\nDataset downloaded: {name}")

print(f"\nDataset loaded: {name}")
print(f"Total samples downloaded: {len(dataset)}")

# Target classes to analyze
target_labels = ["car", "person", "bicycle"]

# Count all detections by label
output_file = "coco_full_classes_result.txt"

with open(output_file, 'w') as f:
    # Write to both console and file
    def write_output(text):
        print(text)
        f.write(text + '\n')
    
    write_output("\n" + "="*60)
    write_output("COCO 2017 DETECTION DATASET - FULL ANALYSIS")
    write_output("="*60)
    
    write_output(f"\nDataset: {name}")
    write_output(f"Split: validation")
    write_output(f"Total samples: {len(dataset)}")
    
    # Print schema to see field names
    write_output(f"\nDataset fields:")
    for field_name, field in dataset.get_field_schema().items():
        write_output(f"  {field_name}: {field}")
    
    # Count all labels in the dataset
    # The field name is 'ground_truth' not 'detections'
    write_output("\n" + "="*60)
    write_output("ALL LABELS IN DATASET")
    write_output("="*60)
    label_counts = dataset.count_values("ground_truth.detections.label")
    
    total_detections = sum(label_counts.values())
    write_output(f"\nTotal detections: {total_detections}")
    write_output(f"\nLabel breakdown (sorted by count):")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_detections) * 100
        write_output(f"  {label:20s}: {count:6d} ({percentage:5.2f}%)")
    
    # Target label statistics
    write_output("\n" + "="*60)
    write_output("TARGET LABELS (car, person, bicycle)")
    write_output("="*60)
    
    target_detections = {}
    target_total = 0
    for label in target_labels:
        count = label_counts.get(label, 0)
        target_detections[label] = count
        target_total += count
        percentage = (count / total_detections) * 100 if total_detections > 0 else 0
        write_output(f"  {label:20s}: {count:6d} ({percentage:5.2f}%)")
    
    write_output(f"\nTotal target detections: {target_total}")
    write_output(f"Target detections ratio: {target_total}/{total_detections} ({(target_total/total_detections)*100:.2f}%)")
    
    # Sample statistics
    write_output("\n" + "="*60)
    write_output("SAMPLE STATISTICS")
    write_output("="*60)
    write_output(f"Total images with target classes: {len(dataset)}")
    
    # Calculate images per class
    for label in target_labels:
        # Count samples containing this label
        view = dataset.filter_labels("ground_truth", fo.ViewField("label") == label)
        samples_with_label = len(view.match(fo.ViewField("ground_truth.detections").length() > 0))
        write_output(f"  Images with '{label}': {samples_with_label}")
    
    # Dataset info
    write_output("\n" + "="*60)
    write_output("DATASET INFORMATION")
    write_output("="*60)
    write_output(str(dataset))

print(f"\n{'='*60}")
print(f"Results saved to: {output_file}")
print(f"{'='*60}")

# Instructions for downloading training set
print("\nNOTE: This script downloaded the VALIDATION set.")
print("To download the FULL TRAINING SET (~20GB with 40k-60k target images):")
print("  1. Change split='validation' to split='train'")
print("  2. Run the script again")
print("  3. Be prepared for a longer download time")
