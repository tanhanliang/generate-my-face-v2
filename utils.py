from PIL import Image
import os
import numpy as np
from PIL import ImageDraw 


TRANSL_PROP = .05

def flip_all_images(path='data/', extension='.jpg'):
    for filename in os.listdir(path):
        if filename.endswith(extension):
            image = Image.open(path + filename)
            rotated_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            new_filename = 'flipped_' + filename
            rotated_image.save(path + new_filename)


def translate_all_images(path='data/', extension='.jpg'):
    for filename in os.listdir(path):
        if filename.endswith(extension):
            image = Image.open(path + filename)
            width, height = image.size
            x_trans = (int)(TRANSL_PROP*width)
            y_trans = (int)(TRANSL_PROP*height)

            for x_coord in range(-1, 2, 1):
                for y_coord in range(-1, 2, 1):
                    if x_coord == 0 and y_coord == 0:
                        continue

                    x_left = x_trans if x_coord == -1 else 0
                    y_top = y_trans if y_coord == -1 else 0
                    x_right = (width - x_trans) if x_coord == 1 else width
                    y_bot = (height - y_trans) if y_coord == 1 else height

                    transl_image = \
                        image.crop((x_left, y_top, x_right, y_bot))
                    new_filename = ('transl_%d_%d_' % (x_coord, y_coord)) + filename
                    transl_image.save(path + new_filename)


def add_gaussian_noise_all_images(path='data/', extension='.jpg'):
    for filename in os.listdir(path):
        if filename.endswith(extension):
            image = Image.open(path + filename)
            width, height = image.size
            image_array = np.asarray(image)

            noise = np.random.normal(0, 15, (height, width, 3))
            noisy_image_arr = image_array + noise
            noisy_image_arr = noisy_image_arr.astype(np.int16)

            clipped_img_arr = np.clip(noisy_image_arr, 0, 255)
            noisy_image = Image.fromarray(clipped_img_arr.astype(np.uint8))

            new_filename = 'noisy_' + filename
            noisy_image.save(path + new_filename)


def add_number_label(path='outputs/', save_path='outputs/', extension='.png'):
    count = 0
    sorted_filenames = sorted(os.listdir(path), key=lambda x: os.path.getctime(path + x))
    for filename in sorted_filenames:
        if filename.endswith(extension):
            count += 1
            image = Image.open(path + filename)
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), str(count))
            image.save(save_path + filename)
            