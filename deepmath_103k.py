import os
import random
import argparse
from datasets import load_dataset

BASEDIR = os.environ['HOME']
TEST_SIZE = 128
TRAIN_SIZE = 16384

DATASET_NAME = "zwhe99/DeepMath-103K"
TARGET_DIR = BASEDIR + "/data/deepmath-16k"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample DeepMath-103K dataset for AReaL")

    parser.add_argument('--target_dir', type=str, default=TARGET_DIR,
                        help="Directory to save the output parquet files")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    print(f"Loading original dataset {DATASET_NAME}...")

    total_data = load_dataset(DATASET_NAME, split='train')
    total_size = len(total_data)
    print(f"Original dataset size: {total_size}")

    train_size = TRAIN_SIZE
    test_size = TEST_SIZE

    if train_size + test_size > total_size:
        raise ValueError(f"Requested sizes ({train_size} + {test_size}) exceed total dataset size ({total_size})!")

    print(f"Sampling {train_size} for train and {test_size} for test...")

    all_indices = list(range(total_size))
    random.seed(args.seed)
    random.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    test_indices = all_indices[train_size : train_size + test_size]

    train_dataset = total_data.select(train_indices)
    test_dataset = total_data.select(test_indices)

    train_path = os.path.join(args.target_dir, 'train-00000-of-00001.parquet')
    test_path = os.path.join(args.target_dir, 'test-00000-of-00001.parquet')

    print(f"Saving sampled datasets to {train_path} and {test_path}...")
    
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print("\n--- Done ---")
    print(f"Train dataset saved to: {train_path} (Size: {len(train_dataset)})")
    print(f"Test dataset saved to:  {test_path} (Size: {len(test_dataset)})")
    print(f"Dataset Features (Columns): {list(train_dataset.features.keys())}")
