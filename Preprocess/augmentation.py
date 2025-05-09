import os
import csv
from glob import glob
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import random
import shutil
from tqdm import tqdm

# 1. Parameters
DATA_DIR = 'Preprocess/Warwick_QU_Dataset'
SAVE_DIR = 'Preprocess/Warwick_QU_Dataset_augmentation'
IMG_SIZE = 256  # Adjust as needed

# 2. Dataset definition
"""
class NoisyImageDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.files = sorted(glob(os.path.join(root_dir, '*.bmp')))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        img = self.transform(img)
        # Add noise
        noise = torch.randn_like(img) * 0.2   # Change 0.2 for desired noise strength
        noisy_img = (img + noise).clamp(0, 1)
        return noisy_img, img
"""

# 3. image transform functions
def mask_transform(original_mask:Image)->Image:
    # transform all mask with value >= 1 to be 1
    mask_arr = np.asarray(original_mask, dtype=np.uint8)
    newmask_arr = np.where(mask_arr >= 1, 1, 0).astype(np.uint8)
    original_mask.close()
    newmask = Image.fromarray(newmask_arr)
    return newmask

def gray_scale(original_image:np.ndarray)->np.ndarray:
    # turn the original bmp to grayscale bmp
    new_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    return new_image


# 4. image augmentation functions (RandAugment)
# reference: https://github.com/NVIDIA/semantic-segmentation/blob/main/datasets/randaugment.py

def random_crop(original_image:Image, original_mask:Image, N:int=5,
                size:int=256)->list: # list of Image
    # crop the image into given size, default is 256*256
    width, height = original_image.size
    rangeW = width - size
    rangeH = height - size
    new_images = list()
    new_masks = list()
    for i in range(N):
        randW = np.random.randint(0, rangeW)
        randH = np.random.randint(0, rangeH)
        new_image = original_image.crop((randW, randH, randW+size, randH+size))
        new_mask = original_mask.crop((randW, randH, randW+size, randH+size))
        new_images.append(new_image)
        new_masks.append(new_mask)
    return new_images, new_masks

FILL_COLOR = (0, 0, 0)
IGNORE_LABEL = 255

def affine_transform(pair, affine_params):
    img, mask = pair
    img = img.transform(img.size, Image.AFFINE, affine_params,
                        resample=Image.BILINEAR, fillcolor=FILL_COLOR)
    mask = mask.transform(mask.size, Image.AFFINE, affine_params,
                          resample=Image.NEAREST, fillcolor=IGNORE_LABEL)
    return img, mask


def ShearX(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if np.random.rand() > 0.5:
        v = -v
    return affine_transform(pair, (1, v, 0, 0, 1, 0))


def ShearY(pair, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if np.random.rand() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, v, 1, 0))


def TranslateX(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if np.random.rand() > 0.5:
        v = -v
    img, _ = pair
    v = v * img.size[0]
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateY(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if np.random.rand() > 0.5:
        v = -v
    img, _ = pair
    v = v * img.size[1]
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def TranslateXAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if np.random.rand() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, v, 0, 1, 0))


def TranslateYAbs(pair, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    if np.random.rand() > 0.5:
        v = -v
    return affine_transform(pair, (1, 0, 0, 0, 1, v))


def Rotate(pair, v):  # [-30, 30]
    assert -30 <= v <= 30
    if np.random.rand() > 0.5:
        v = -v
    img, mask = pair
    img = img.rotate(v, fillcolor=FILL_COLOR)
    mask = mask.rotate(v, resample=Image.NEAREST, fillcolor=IGNORE_LABEL)
    return img, mask


def AutoContrast(pair, _):
    img, mask = pair
    return ImageOps.autocontrast(img), mask


def Invert(pair, _):
    img, mask = pair
    return ImageOps.invert(img), mask


def Equalize(pair, _):
    img, mask = pair
    return ImageOps.equalize(img), mask


def Flip(pair, _):  # not from the paper
    img, mask = pair
    return ImageOps.mirror(img), ImageOps.mirror(mask)


def Solarize(pair, v):  # [0, 256]
    img, mask = pair
    assert 0 <= v <= 256
    return ImageOps.solarize(img, v), mask


def Posterize(pair, v):  # [4, 8]
    img, mask = pair
    assert 4 <= v <= 8
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Posterize2(pair, v):  # [0, 4]
    img, mask = pair
    assert 0 <= v <= 4
    v = int(v)
    return ImageOps.posterize(img, v), mask


def Contrast(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Contrast(img).enhance(v), mask


def Color(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Color(img).enhance(v), mask


def Brightness(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Brightness(img).enhance(v), mask


def Sharpness(pair, v):  # [0.1,1.9]
    img, mask = pair
    assert 0.1 <= v <= 1.9
    return ImageEnhance.Sharpness(img).enhance(v), mask


def Cutout(pair, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return pair
    img, mask = pair
    v = v * img.size[0]
    return CutoutAbs(pair, v)


def CutoutAbs(pair, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    img, mask = pair
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    img = img.copy()
    ImageDraw.Draw(img).rectangle(xy, FILL_COLOR)
    mask = mask.copy()
    ImageDraw.Draw(mask).rectangle(xy, IGNORE_LABEL)
    return img, mask


def Identity(pair, _):
    return pair


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    l = [
        (Identity, 0., 1.0), # 0
        (ShearX, 0., 0.3),  # 1
        (ShearY, 0., 0.3),  # 2
        (TranslateX, 0., 0.33),  # 3
        (TranslateY, 0., 0.33),  # 4
        (Rotate, 0, 30),  # 5
        (AutoContrast, 0, 1),  # 6
        (Invert, 0, 1),  # 7
        (Equalize, 0, 1),  # 8
        (Solarize, 0, 110),  # 9
        (Posterize, 4, 8),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (Cutout, 0, 0.2),  # 14
        (Flip, 1, 1) #15
    ]
    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img:Image, mask:Image):
        pair = img, mask
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            pair = op(pair, val)

        return pair

# 5. zip and unzip
def compress_dataset(path):
    shutil.make_archive(path, 'zip', path)
    shutil.rmtree(path)

def uncompress_dataset(path):
    os.makedirs(path, exist_ok=True)
    shutil.unpack_archive(path+".zip", path)


# 6. Process
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    files = list()
    seq_num = 1
    rand_augmentor_list = list()
    # One can modify this to get more augmented training dataset
    rand_augmentor_list.append(RandAugment(3, 8))
    rand_augmentor_list.append(RandAugment(4, 13))
    rand_augmentor_list.append(RandAugment(5, 19))

    with open(DATA_DIR+"/Grade.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            files.append(row)

    with open(SAVE_DIR+"/Grade.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "grade1", "grade2"])

        for r in tqdm(files, desc='Augmenting and saving'):
            if r[0] is None:
                break
            fn, _, l1, l2 = r
            if fn.find('test') >= 0:
                continue
            else:
                f = DATA_DIR+'/'+fn+'.bmp'
                mf = DATA_DIR+'/'+fn+'_anno.bmp'
            img = Image.open(f).convert('RGB')
            mask = Image.open(mf).convert('L')
            mask = mask_transform(mask)
            img_list, mask_list = random_crop(img, mask, N=5, size=IMG_SIZE) # modify N to get more crop
            for timg, tmask in zip(img_list, mask_list):
                for rdaug in rand_augmentor_list:
                    newimg, newmask = rdaug(timg, tmask)
                    newimg.save(SAVE_DIR+"/train_"+str(seq_num)+".bmp")
                    newmask.save(SAVE_DIR+"/train_"+str(seq_num)+"_anno.bmp")
                    augname = "train_"+str(seq_num)
                    writer.writerow([augname, l1, l2])
                    seq_num += 1

                timg.save(SAVE_DIR+"/train_"+str(seq_num)+".bmp")
                tmask.save(SAVE_DIR+"/train_"+str(seq_num)+"_anno.bmp")
                augname = "train_"+str(seq_num)
                writer.writerow([augname, l1, l2])
                seq_num += 1

            img.close()
            mask.close()

        print('All done! Augmented images are saved in:', SAVE_DIR)
