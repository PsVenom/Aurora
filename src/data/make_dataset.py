# -*- coding: utf-8 -*-
import click
import logging

from functools import partial
import pandas as pd
import tensorflow as tf
batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}
import collections
import random
# We adjust the train step accordingly
train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}
#
# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True),  default = os.get_cwd()+'caption_dataframe.csv')
# @click.argument('output_filepath', type=click.Path(), default = os.get_cwd()+'caption_dataframe.csv')
def return_string_array(caption):
    arr = []
    c = caption.replace("[", "")
    c = c.replace("]", "")
    c = c.replace("'", "")
    pointer = 0
    for i, letter in enumerate(c):
        if letter == ",":
            string = c[pointer:i]
            arr.append(string)
            pointer = i + 2
    return arr
def set_data(caption_dataframe, dir_path):
    img_name_vector = []
    train_captions = []
    for row in caption_dataframe.itertuples(index = True):
        captions_array = return_string_array(row.text_captions)
        for cap in captions_array:
            img_name_vector.append(dir_path + "/" + row.ImagePath)
            train_captions.append(cap)
    encode_train = sorted(set(img_name_vector))
    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)
    cap_vector = caption_dataset

    #splitting dataset
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])
    img_dataset = tf.data.Dataset.from_tensor_slices(img_name_train)
    img_vectors = img_dataset.map(load_image)
    cap_dataset = tf.data.Dataset.from_tensor_slices(cap_train)
    cap_vector = cap_dataset
    return cap_vector, img_vectors
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img
def resize_image(res, image):
    # only donwsampling, so use nearest neighbor that is faster to run
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image

def create_dataloader(res, img_vectors, cap_vector):
    batch_size = 16
    dl = img_vectors.map(partial(resize_image, res), num_parallel_calls=tf.data.AUTOTUNE)
    dl = dl.batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    dl2 = cap_vector.batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl, dl2
def make_dataset(dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    caption_dataframe = pd.read_csv(dir+"/caption_dataframe.csv")
    caption_dataframe['ImagePath'] = caption_dataframe['ImagePath'].apply(lambda x: x.split('/')[-1])
    # caption_dataframe.to_cav(output_filepath)
    cap_vector, img_vectors = set_data(caption_dataframe, dir_path= dir)
    return cap_vector, img_vectors


# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)
#
#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]
#
#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())
#
#     main()
