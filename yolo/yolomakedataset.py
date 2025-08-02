import os
import shutil
import random


# using other hdd for large dataset
dataset_dir = 'D:/dataset'

split_ratio = 0.9

def prepare_dataset(batch_dirs, dataset_dir, split_ratio):
    train_images_dir = os.path.join(dataset_dir, 'train', 'images')
    train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    val_images_dir = os.path.join(dataset_dir, 'val', 'images')
    val_labels_dir = os.path.join(dataset_dir, 'val', 'labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    image_files = []
    label_files = []
    for batch_dir in batch_dirs:
        for filename in os.listdir(batch_dir):
            filepath = os.path.join(batch_dir, filename)
            if os.path.isfile(filepath):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
                    image_files.append(filepath)
                elif filename.lower().endswith('.txt'):
                    label_files.append(filepath)

    image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    label_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

    matched_files = []
    for base_name in image_dict.keys():
        if base_name in label_dict:
            matched_files.append(base_name)
        else:
            print(f"Skipping {image_dict[base_name]} no matching label...")

    print(f"\nTotal matched files: {len(matched_files)}")

    random.shuffle(matched_files)

    split_index = int(len(matched_files) * split_ratio)
    train_files = matched_files[:split_index]
    val_files = matched_files[split_index:]

    def copy_files(file_list, dest_images_dir, dest_labels_dir):
        for base_name in file_list:
            img_src = image_dict[base_name]
            label_src = label_dict[base_name]

            img_ext = os.path.splitext(img_src)[1]
            label_ext = os.path.splitext(label_src)[1]

            img_dst = os.path.join(dest_images_dir, base_name + img_ext)
            label_dst = os.path.join(dest_labels_dir, base_name + label_ext)

            if os.path.exists(img_src) and os.path.exists(label_src):
                shutil.copy2(img_src, img_dst)
                shutil.copy2(label_src, label_dst)
            else:
                print(f"Missing files for {base_name}. Skipping.")

    copy_files(train_files, train_images_dir, train_labels_dir)

    copy_files(val_files, val_images_dir, val_labels_dir)

    print("\Dataset organization complete")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")

if __name__ == '__main__':
    batch_dirs = [os.path.join('D:/outputcaptchas', f'batch_{i}') for i in range(1, 15)]

    prepare_dataset(batch_dirs, dataset_dir, split_ratio)
