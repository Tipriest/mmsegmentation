#!/usr/bin/env python3
"""
Script to remap label values in TIB_GROUND1 dataset to match the model expectations.
Original labels: {4, 34, 83, 104, 144, 188, 216} -> New labels: {0, 1, 2, 3, 4, 5, 6}
"""

import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm


def remap_labels():
    """Remap label values to match model expectations"""
    
    # Define the mapping from original labels to new labels
    # Based on the labels_map.txt file
    original_to_new = {
        4: 0,    # concreteroad
        34: 1,   # road_curb  
        83: 2,   # redbrickroad
        104: 3,  # zebracrossing
        144: 4,  # stone_pier
        188: 5,  # soil
        216: 6   # yellowbrick_road
    }
    
    # Define data paths
    data_root = '/home/tipriest/Documents/mmsegmentation/data/tib_ground1'
    train_label_dir = os.path.join(data_root, 'labels', 'train')
    val_label_dir = os.path.join(data_root, 'labels', 'val')
    
    # Create backup directories
    train_backup_dir = os.path.join(data_root, 'labels', 'train_backup')
    val_backup_dir = os.path.join(data_root, 'labels', 'val_backup')
    
    os.makedirs(train_backup_dir, exist_ok=True)
    os.makedirs(val_backup_dir, exist_ok=True)
    
    def process_directory(label_dir, backup_dir, split_name):
        """Process all label files in a directory"""
        if not os.path.exists(label_dir):
            print(f"Directory {label_dir} does not exist, skipping...")
            return
            
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
        
        print(f"Processing {split_name} labels: {len(label_files)} files")
        
        for filename in tqdm(label_files, desc=f"Processing {split_name}"):
            file_path = os.path.join(label_dir, filename)
            backup_path = os.path.join(backup_dir, filename)
            
            # Load the original label image
            label_img = Image.open(file_path)
            label_array = np.array(label_img)
            
            # Create backup
            Image.fromarray(label_array).save(backup_path)
            
            # Create new label array
            new_label_array = np.zeros_like(label_array)
            
            # Apply mapping
            for original_val, new_val in original_to_new.items():
                mask = label_array == original_val
                new_label_array[mask] = new_val
            
            # Set all other values to ignore_index (255)
            # Find pixels that don't match any of our target classes
            valid_mask = np.zeros_like(label_array, dtype=bool)
            for original_val in original_to_new.keys():
                valid_mask |= (label_array == original_val)
            
            # Set invalid pixels to ignore_index
            new_label_array[~valid_mask] = 255
            
            # Save the remapped label image
            Image.fromarray(new_label_array.astype(np.uint8)).save(file_path)
    
    # Process train and validation directories
    process_directory(train_label_dir, train_backup_dir, 'train')
    process_directory(val_label_dir, val_backup_dir, 'val')
    
    # Update the labels_map.txt file
    new_labels_map = {
        "concreteroad": 0,
        "road_curb": 1,
        "redbrickroad": 2,
        "zebracrossing": 3,
        "stone_pier": 4,
        "soil": 5,
        "yellowbrick_road": 6
    }
    
    labels_map_path = os.path.join(data_root, 'labels', 'labels_map.txt')
    labels_map_backup_path = os.path.join(data_root, 'labels', 'labels_map_backup.txt')
    
    # Create backup of original labels_map.txt
    if os.path.exists(labels_map_path):
        with open(labels_map_path, 'r') as f:
            original_content = f.read()
        with open(labels_map_backup_path, 'w') as f:
            f.write(original_content)
    
    # Write new labels_map.txt
    with open(labels_map_path, 'w') as f:
        json.dump(new_labels_map, f, indent=4)
    
    print("Label remapping completed!")
    print(f"Original labels backed up to: {train_backup_dir} and {val_backup_dir}")
    print(f"Original labels_map.txt backed up to: {labels_map_backup_path}")
    print("New label mapping:")
    for class_name, label_id in new_labels_map.items():
        print(f"  {class_name}: {label_id}")


if __name__ == "__main__":
    remap_labels()
