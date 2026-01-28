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

print(f"Original dataset: {len(dataset)} samples")
print(f"Filtered dataset: {len(filtered_view)} samples with car/person/bicycle annotations")
print("\nFiltered dataset info:")
print(filtered_view)
print("\nSample data:")
print(filtered_view.head())
