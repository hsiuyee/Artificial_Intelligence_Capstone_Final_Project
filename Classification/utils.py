import os
import shutil
import pandas as pd


def prepare_glas_imagefolder(src_root: str, dst_root: str):
    """
    Converts Warwick_QU_Dataset to torchvision.datasets.ImageFolder format.
    
    Args:
        src_root (str): Path to raw dataset containing 'Grade.csv' and images.
        dst_root (str): Path to save ImageFolder-style dataset (e.g., glas_imagefolder/train).

    Result:
        Creates dst_root/<label>/*.bmp for each image.
    """
    csv_path = os.path.join(src_root, 'Grade.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    os.makedirs(dst_root, exist_ok=True)

    # Load label CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # clean column names

    df['filename'] = df['name'].astype(str)
    df['label'] = df['grade (GlaS)'].str.strip().str.lower()  # 'benign', 'malignant'

    # Create subfolders for each label
    for cls in df['label'].unique():
        os.makedirs(os.path.join(dst_root, cls), exist_ok=True)

    # Copy images to appropriate folders
    copied = 0
    for _, row in df.iterrows():
        basename = row['filename']
        label = row['label']
        bmp_file = f"{basename}.bmp"
        src_path = os.path.join(src_root, bmp_file)
        dst_path = os.path.join(dst_root, label, bmp_file)

        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            copied += 1
        else:
            print(f"[WARN] Missing image: {src_path}")

    print(f"✅ Prepared {copied} images into '{dst_root}' (ImageFolder format)")


if __name__ == "__main__":
    prepare_glas_imagefolder(
        src_root=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../Preprocess/Warwick_QU_Dataset/") 
        ),
        dst_root="dataset/train"
    ) 
    # prepare_glas_imagefolder(
    #     src_root=os.path.abspath(
    #         os.path.join(os.path.dirname(__file__), "../Preprocess/Warwick_QU_Dataset/test") 
    #     ),
    #     dst_root="dataset/test" 
    # ) 