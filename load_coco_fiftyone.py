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

print(dataset)
print(dataset.head())
