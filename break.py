from PIL import Image
from pathlib import Path
import os
import argparse


def split_images(img_dir, classes, res, prefix):
    """
    splits grid images into folders
    """
    os.mkdir(prefix)
    for i in range(classes):
        os.mkdir(prefix + '/' + 'class' + str(i))
    dp = Path(img_dir)
    input_dir = os.listdir(dp)
    frame_num = 1
    for i in input_dir:
        im = Image.open(Path(img_dir + '/' + i))
        im_width, im_height = im.size
        rows = im_height / res
        w, h = (int(im_width / classes), int(im_height / rows))  # number of rows
        k = 0
        count = 0
        for col_i in range(0, im_width, w):
            for row_i in range(0, im_height, h):
                crop = im.crop((col_i, row_i, col_i + w, row_i + h))
                save_to = os.path.join(Path(prefix + '/class' + str(k % classes)),
                                       "counter_{:06}.jpg")
                crop.save(save_to.format(frame_num))
                count += 1
                print("{}th no of image saved ".format(count))
                frame_num += 1
            k += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--img_dir', type=str, default='data', help='directory that contains grid images')
    parser.add_argument('--classes', type=int, default=2, help='number of classes')
    parser.add_argument('--res', type=int, default=256, help='resolution of images')
    parser.add_argument('--prefix', type=str, default='256_resolution',
                        help='parent directory that contains all the splitted classes')

    args = parser.parse_args()
    split_images(args.img_dir, args.classes + 1, args.res, args.prefix)

