import os
import argparse
import re
from PIL import Image

def cleanup_and_rename_images(input_folder):
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
    filtered_files = [f for f in jpg_files if not int(re.findall(r'\d+', f)[-1]) % 2 != 0]
    filtered_files.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    for index, filename in enumerate(filtered_files, 0):
        input_path = os.path.join(input_folder, filename)
        output_filename = f"{index:05d}.png"
        output_path = os.path.join(input_folder, output_filename)
        
        try:
            with Image.open(input_path) as img:
                img.save(output_path, 'PNG')
            os.remove(input_path)
            print(f"execute: {filename} -> {output_filename}")
        
        except Exception as e:
            print(f"execute {filename} error: {e}")
    
    print(f"\nprocess done. Totally have {len(filtered_files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video and save them as images.")
    parser.add_argument("--images_dir", type=str, required=True, help="cleanup_and_rename_images")
    args = parser.parse_args()
    cleanup_and_rename_images(args.images_dir)