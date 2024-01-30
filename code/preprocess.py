import os
import json
from tqdm import tqdm
from PIL import Image

import numpy as np
import pickle

from dataset import filter_vertices, resize_img, adjust_height, rotate_img, crop_img


def preprocessing(
        root_dir,
        image_size=2048,
        crop_size=1024,
        ignore_under_threshold=10,
        drop_under_threshold=1,
):
    json_dir = os.path.join(root_dir, 'ufo/train.json')
    with open(json_dir, "r") as f:
        anno = json.load(f)

    image_fnames = sorted(anno["images"].keys())
    image_dir = os.path.join(root_dir, 'img/train')

    save_images = []
    save_vertices = []
    save_labels = []
    for idx in tqdm(range(len(image_fnames))):
        image_fname = image_fnames[idx]
        image_fpath = os.path.join(image_dir, image_fname)

        vertices, labels = [], []
        for word_info in anno["images"][image_fname]["words"].values():
            num_pts = np.array(word_info["points"]).shape[0]
            if num_pts > 4:
                continue

            vertices.append(np.array(word_info["points"]).flatten())
            labels.append(int(not word_info["illegibility"]))
        vertices = np.array(vertices, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=ignore_under_threshold,
            drop_under=drop_under_threshold,
        )
        image = Image.open(image_fpath)

        image, vertices = resize_img(image, vertices, image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        save_images.append(image)
        save_vertices.append(vertices)
        save_labels.append(labels)

    save_images = np.array(save_images)
    path = os.path.join(root_dir, 'ufo/image.npy')
    np.save(path, save_images)
    path = os.path.join(root_dir, 'ufo/vertices.pickle')
    with open(path, 'wb') as f:
        pickle.dump(save_vertices, f)
    path = os.path.join(root_dir, 'ufo/labels.pickle')
    with open(path, 'wb') as f:
        pickle.dump(save_labels, f)


preprocessing(
        root_dir='../data/medical',
        image_size=2048,
        crop_size=1024,
        ignore_tags=["masked", "excluded-region", "maintable", "stamp"],
        ignore_under_threshold=10,
        drop_under_threshold=1,
    )
