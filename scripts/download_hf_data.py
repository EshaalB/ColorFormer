import os
from datasets import load_dataset
from PIL import Image
import io

def prepare_hf_data(dataset_name='pcuenq/oxford-pets', num_images=50, output_dir='datasets/hf_images'):
    print(f"Loading dataset {dataset_name}...")
    # Using streaming=True as requested by user
    dataset = load_dataset(dataset_name, split='train', streaming=True)
    
    # Matching user style: print the first item
    print("Sample from dataset:")
    print(next(iter(dataset)))
    
    os.makedirs(output_dir, exist_ok=True)
    train_list_path = 'train_list.txt'
    
    paths = []
    print(f"Downloading {num_images} images...")
    
    count = 0
    for i, item in enumerate(dataset):
        if count >= num_images:
            break
            
        # The 'image' field usually contains a PIL image or bytes
        img = item['image']
        if not isinstance(img, Image.Image):
            # If it's bytes, convert to PIL
            img = Image.open(io.BytesIO(img))
            
        img_filename = f"img_{count:04d}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        
        # Convert to RGB if necessary (ColorFormer expects BGR/RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(img_path)
        paths.append(img_path)
        count += 1
        if count % 10 == 0:
            print(f"  Downloaded {count}/{num_images}...")

    with open(train_list_path, 'w') as f:
        for p in paths:
            f.write(f"{p}\n")
            
    print(f"Created {train_list_path} with {len(paths)} entries.")

if __name__ == "__main__":
    prepare_hf_data()
