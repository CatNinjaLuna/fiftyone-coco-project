import fiftyone as fo
import fiftyone.zoo as foz

# Target classes
target_labels = ["car", "person", "bicycle"]

# COCO 2017 tasks to analyze
tasks = {
    "detection": "detections",
    "segmentation": "segmentations", 
    "keypoints": "keypoints"
}

output_file = "coco_all_tasks_result.txt"

with open(output_file, 'w') as f:
    def write_output(text):
        print(text)
        f.write(text + '\n')
    
    write_output("="*70)
    write_output("COCO 2017 ALL TASKS ANALYSIS")
    write_output("Analyzing: Detection, Segmentation, and Keypoints tasks")
    write_output("Target classes: car, person, bicycle")
    write_output("="*70)
    
    all_stats = {}
    
    for task_name, label_field in tasks.items():
        write_output(f"\n{'='*70}")
        write_output(f"TASK: {task_name.upper()}")
        write_output(f"{'='*70}")
        
        dataset_name = f"coco-2017-{task_name}"
        
        try:
            # Try to load existing dataset first
            try:
                dataset = fo.load_dataset(dataset_name)
                write_output(f"Loaded existing dataset: {dataset_name}")
            except:
                # Download if not exists
                write_output(f"Downloading COCO 2017 {task_name} (validation split)...")
                write_output("Filtering for: car, person, bicycle")
                
                dataset = foz.load_zoo_dataset(
                    "coco-2017",
                    split="validation",
                    label_types=[label_field],
                    classes=target_labels,
                    dataset_name=dataset_name,
                )
                write_output(f"Dataset downloaded: {dataset_name}")
            
            write_output(f"\nTotal samples: {len(dataset)}")
            
            # Get field name for labels
            schema = dataset.get_field_schema()
            label_field_name = None
            
            # Find the label field in the schema
            for field_name in schema.keys():
                if 'ground_truth' in field_name or label_field in field_name:
                    label_field_name = field_name
                    break
            
            if not label_field_name:
                # Try common field names
                possible_fields = ['ground_truth', 'detections', 'segmentations', 'keypoints']
                for pf in possible_fields:
                    if pf in schema:
                        label_field_name = pf
                        break
            
            if not label_field_name:
                write_output(f"Warning: Could not find label field for {task_name}")
                write_output(f"Available fields: {list(schema.keys())}")
                continue
            
            write_output(f"Label field: {label_field_name}")
            
            # Count labels based on task type
            try:
                if task_name == "detection" or task_name == "segmentation":
                    label_path = f"{label_field_name}.detections.label"
                elif task_name == "keypoints":
                    label_path = f"{label_field_name}.keypoints.label"
                else:
                    label_path = f"{label_field_name}.label"
                
                label_counts = dataset.count_values(label_path)
                
                write_output(f"\nLabel counts:")
                total_detections = sum(label_counts.values())
                
                # Show all labels
                for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                    write_output(f"  {label:20s}: {count:6d} ({percentage:5.2f}%)")
                
                # Target class statistics
                write_output(f"\nTarget class statistics:")
                task_stats = {}
                for label in target_labels:
                    count = label_counts.get(label, 0)
                    task_stats[label] = count
                    percentage = (count / total_detections) * 100 if total_detections > 0 else 0
                    write_output(f"  {label:20s}: {count:6d} ({percentage:5.2f}%)")
                
                all_stats[task_name] = {
                    'total_samples': len(dataset),
                    'total_detections': total_detections,
                    'target_counts': task_stats
                }
                
                # Count images per target class
                write_output(f"\nImages containing target classes:")
                for label in target_labels:
                    view = dataset.filter_labels(label_field_name, fo.ViewField("label") == label)
                    try:
                        if task_name == "detection" or task_name == "segmentation":
                            samples_with_label = len(view.match(fo.ViewField(f"{label_field_name}.detections").length() > 0))
                        elif task_name == "keypoints":
                            samples_with_label = len(view.match(fo.ViewField(f"{label_field_name}.keypoints").length() > 0))
                        else:
                            samples_with_label = len(view)
                        write_output(f"  Images with '{label}': {samples_with_label}")
                    except:
                        write_output(f"  Images with '{label}': Unable to count")
                
            except Exception as e:
                write_output(f"Error counting labels: {e}")
                write_output(f"Skipping detailed analysis for {task_name}")
        
        except Exception as e:
            write_output(f"Error loading {task_name} dataset: {e}")
            write_output(f"Skipping {task_name} task")
            continue
    
    # Summary across all tasks
    write_output(f"\n{'='*70}")
    write_output("SUMMARY ACROSS ALL TASKS")
    write_output(f"{'='*70}")
    
    if all_stats:
        total_across_all = {label: 0 for label in target_labels}
        total_samples_all = 0
        total_detections_all = 0
        
        for task_name, stats in all_stats.items():
            total_samples_all += stats['total_samples']
            total_detections_all += stats['total_detections']
            for label, count in stats['target_counts'].items():
                total_across_all[label] += count
        
        write_output(f"\nTotal samples across all tasks: {total_samples_all}")
        write_output(f"Total detections across all tasks: {total_detections_all}")
        write_output(f"\nTarget class totals:")
        
        grand_total = sum(total_across_all.values())
        for label in target_labels:
            count = total_across_all[label]
            percentage = (count / total_detections_all) * 100 if total_detections_all > 0 else 0
            write_output(f"  {label:20s}: {count:6d} ({percentage:5.2f}%)")
        
        write_output(f"\nGrand total target detections: {grand_total}")
        
        # Breakdown by task
        write_output(f"\nBreakdown by task:")
        for task_name, stats in all_stats.items():
            write_output(f"  {task_name:20s}: {stats['total_detections']} detections in {stats['total_samples']} samples")
    else:
        write_output("No statistics collected. All tasks failed to load.")

print(f"\n{'='*70}")
print(f"Results saved to: {output_file}")
print(f"{'='*70}")
print("\nNOTE: This analyzed the VALIDATION sets.")
print("To analyze TRAINING sets:")
print("  1. Change split='validation' to split='train' in the script")
print("  2. Re-run for much larger datasets")
