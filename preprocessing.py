import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import gc
from PIL import Image
import zipfile
import kagglehub

# Set dataset path
path = kagglehub.dataset_download("piele1/tiny-persons-upscaled-x4")
dataset_path = path
print(f"Dataset path: {dataset_path}")

# IMAGE PREPROCESSING ONLY 

class ImagePreprocessor:
    
    def __init__(self, img_size=224, save_format='png'):
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.save_format = save_format
        
        print(f"\n IMAGE PREPROCESSOR CONFIGURED:")
        print(f"  • Target size: {img_size}x{img_size}")
        print(f"  • Normalization: ImageNet stats")
        print(f"  • Save format: {save_format.upper()}")
    
    def preprocess_image(self, image_path):

        #Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return Nonen
        
        #Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]
        
        #Calculate scale factor (maintain aspect ratio)
        scale = min(self.img_size / original_w, self.img_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        #Resize image (preserve aspect ratio)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Free original image
        del img
        
        #Add padding to create square image
        square_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2
        square_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
        
        # Free resized image
        del img_resized
        
        # Normalize pixels with ImageNet stats (for tensor)
        img_normalized = (square_img / 255.0 - self.mean) / self.std
        
        #Convert to tensor (C, H, W) for PyTorch
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
        
        # Free normalized array
        del img_normalized
        
        # Force garbage collection
        gc.collect()
        
        return {
            'tensor': img_tensor,  # For training
            'visualization_img': square_img,  # For saving as image
            'original_shape': (original_h, original_w),
            'processed_shape': (self.img_size, self.img_size),
            'scale': scale,
            'padding': (pad_x, pad_y),
            'filename': os.path.basename(image_path),
            'filename_without_ext': os.path.splitext(os.path.basename(image_path))[0]
        }

# FIND ALL IMAGES IN DATASET

def find_all_images(dataset_path):
    """Find all images in the dataset"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    all_images = []
    
    print("\n SCANNING FOR IMAGES...")
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                all_images.append(os.path.join(root, file))
    
    print(f"  Found {len(all_images)} images total")
    return all_images

# SAVE IMAGES IN BATCHES

def save_images_in_batches(processed_items, output_dir, batch_num, save_format='png'):
    """Save processed images as actual image files"""
    
    # Create batch directory
    batch_dir = os.path.join(output_dir, f'batch_{batch_num:04d}')
    os.makedirs(batch_dir, exist_ok=True)
    
    saved_count = 0
    for item in processed_items:
        # Get the visualization image (already in RGB, 0-255 range)
        img_to_save = item['visualization_img']
        
        # Create filename
        base_name = item['filename_without_ext']
        # Clean filename 
        base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        img_filename = f"{base_name}_224x224.{save_format}"
        img_path = os.path.join(batch_dir, img_filename)
        
        # Save as image using PIL
        Image.fromarray(img_to_save).save(img_path)
        
        # Also save metadata as JSON
        import json
        metadata = {
            'original_filename': item['filename'],
            'original_shape': item['original_shape'],
            'processed_shape': item['processed_shape'],
            'scale': item['scale'],
            'padding': item['padding']
        }
        meta_path = os.path.join(batch_dir, f"{base_name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_count += 1
    
    return saved_count

# PROCESS ALL IMAGES IN BATCHES

def process_all_images(image_paths, preprocessor, batch_size=50, output_dir='/kaggle/working/processed_images'):
    """
    Process all images and save as actual image files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING ALL {len(image_paths)} IMAGES")
    print(f"{'='*60}")
    
    stats = {
        'total': 0,
        'successful': 0,
        'failed': 0,
        'batches_saved': 0
    }
    
    # Process in batches
    for batch_idx in range(0, len(image_paths), batch_size):
        batch = image_paths[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx//batch_size + 1
        total_batches = (len(image_paths)-1)//batch_size + 1
        
        print(f"\n Batch {batch_num}/{total_batches}")
        
        batch_results = []
        
        for img_path in tqdm(batch, desc="  Preprocessing"):
            try:
                # Apply steps 2.1-2.8
                processed = preprocessor.preprocess_image(img_path)
                
                if processed is not None:
                    batch_results.append(processed)
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                
                stats['total'] += 1
                
            except Exception as e:
                print(f"  ✗ Error on {img_path}: {e}")
                stats['failed'] += 1
                stats['total'] += 1
        
        # Save batch results as actual images
        if batch_results:
            saved = save_images_in_batches(
                batch_results, 
                output_dir, 
                batch_num,
                save_format=preprocessor.save_format
            )
            stats['batches_saved'] += 1
            print(f"  Saved {saved} images to batch_{batch_num:04d}/")
        
        # Clear memory
        del batch_results
        gc.collect()
    
    return stats, output_dir


# CREATE DATASET STRUCTURE (Train/Val/Test folders)

def organize_by_split(structure, preprocessor, output_dir='/kaggle/working/transformer_dataset'):
    """Process and organize images by train/val/test splits"""
    
    print(f"\n{'='*60}")
    print(" ORGANIZING BY TRAIN/VAL/TEST SPLITS")
    print(f"{'='*60}")
    
    stats = {'train': 0, 'val': 0, 'test': 0, 'failed': 0}
    
    for split_name, split_images in structure.items():
        if not split_images:
            continue
            
        # Create split directory
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\n Processing {split_name.upper()} split: {len(split_images)} images")
        
        for img_path in tqdm(split_images, desc=f"  {split_name}"):
            try:
                processed = preprocessor.preprocess_image(img_path)
                
                if processed:
                    # Save image
                    base_name = processed['filename_without_ext']
                    base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    
                    img_filename = f"{base_name}_224x224.{preprocessor.save_format}"
                    img_path_saved = os.path.join(split_dir, img_filename)
                    Image.fromarray(processed['visualization_img']).save(img_path_saved)
                    
                    stats[split_name] += 1
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                stats['failed'] += 1
    
    return stats, output_dir


# EXECUTE PREPROCESSING

# Step 1: Initialize preprocessor
preprocessor = ImagePreprocessor(img_size=224, save_format='png')

# Step 2: Find all images
all_images = find_all_images(dataset_path)

# OPTION A: Process all images in batches (no split organization)
print("\n" + "="*60)
print(" OPTION A: PROCESS ALL IMAGES IN BATCHES")
print("="*60)

stats, output_dir = process_all_images(
    all_images, 
    preprocessor, 
    batch_size=50,
    output_dir='/kaggle/working/processed_images'
)




# SUMMARY
print("\n" + "="*60)
print(" IMAGE PREPROCESSING COMPLETE!")
print("="*60)
print(f"\n STATISTICS:")
print(f"  • Total images processed: {stats['total']}")
print(f"  • Successfully saved: {stats['successful']}")
print(f"  • Failed: {stats['failed']}")
print(f"  • Batches saved: {stats['batches_saved']}")
print(f"\n Output saved to: {output_dir}")


# VERIFY SAVED IMAGES


def verify_saved_images(output_dir, num_samples=3):
    """Verify that images were saved correctly"""
    
    print("\n VERIFYING SAVED IMAGES:")
    
    # Find first few images
    image_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
        if len(image_files) >= num_samples:
            break
    
    if not image_files:
        print("  No image files found to verify")
        return
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, min(num_samples, len(image_files)), figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i, img_path in enumerate(image_files[:num_samples]):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(os.path.basename(img_path))
        axes[i].axis('off')
        print(f"  ✓ Verified: {os.path.basename(img_path)}")
    
    plt.tight_layout()
    plt.show()

# Verify the saved images
verify_saved_images(output_dir, num_samples=3)


# CREATE ZIP FILE FOR DOWNLOAD


print("\n" + "="*60)
print(" PREPARING FOR DOWNLOAD")
print("="*60)

# Create zip file
zip_path = '/kaggle/working/processed_images_dataset.zip'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
            zipf.write(file_path, arcname)
            print(f"  📄 Added: {file}")

size_mb = os.path.getsize(zip_path) / (1024*1024)
print(f"\n Created zip file: {zip_path}")
print(f" Size: {size_mb:.2f} MB")
print(f" Contains: {len(os.listdir(output_dir))} batches/folders")
