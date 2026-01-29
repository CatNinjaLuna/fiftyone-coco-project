import fiftyone as fo

name = "coco-local"
dataset_dir = "/Users/carolina1650/fiftyone-coco-project/quickstart"

if name in fo.list_datasets():
    fo.delete_dataset(name)

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.FiftyOneDataset,
    name=name,
)

# Filter for samples with car, person, or bicycle annotations
target_labels = ["car", "person", "bicycle"]

# Filter samples that have at least one detection with the target labels
filtered_view = dataset.filter_labels(
    "ground_truth", 
    fo.ViewField("label").is_in(target_labels)
).match(
    fo.ViewField("ground_truth.detections").length() > 0
)

# Count all detections by label in the original dataset
output_file = "classes_result_200.txt"

with open(output_file, 'w') as f:
    # Write to both console and file
    def write_output(text):
        print(text)
        f.write(text + '\n')
    
    write_output("=== Label Counts in Original Dataset ===")
    label_counts = dataset.count_values("ground_truth.detections.label")
    write_output(f"Total samples: {len(dataset)}")
    write_output(f"\nAll labels found:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        write_output(f"  {label}: {count}")
    
    # Count target label detections
    total_target_detections = sum(label_counts.get(label, 0) for label in target_labels)
    write_output(f"\n=== Target Labels (car, person, bicycle) ===")
    for label in target_labels:
        count = label_counts.get(label, 0)
        write_output(f"  {label}: {count}")
    write_output(f"Total target detections: {total_target_detections}")
    write_output(f"Filtered samples (images with target labels): {len(filtered_view)}")
    
    write_output("\n=== Filtered Dataset Info ===")
    write_output(str(filtered_view))

print(f"\nResults saved to: {output_file}")
print("\nSample data:")
print(filtered_view.head())
