import fiftyone as fo

name = "coco-local"
dataset_dir = "/absolute/path/to/coco"

if name in fo.list_datasets():
    fo.delete_dataset(name)

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset,
    name=name,
)

print(dataset)
print(dataset.head())
