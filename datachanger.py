#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""
Template doc
"""
import argparse
import cv2
import os

import numpy as np

from random import shuffle

def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img_path',
                        help='Path to images.')
    parser.add_argument('--mask_path',
                        help='Path to masks.')
    parser.add_argument('--save_path',
                        help='Path to save dir.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()
    img_data_path = args.img_path
    mask_data_path = args.mask_path
    list_mask = sorted(os.listdir(mask_data_path))
    list_img = sorted(os.listdir(img_data_path))

    for i in range(len(list_mask)):
        img = cv2.imread(img_data_path + list_img[i])
        mask = cv2.imread(mask_data_path + list_mask[i])
        black = np.all(mask == [0, 0, 0], axis=-1)
        mask[black] = [0, 0, 255]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 90, 255, cv2.THRESH_BINARY)
        cv2.imwrite(args.save_path+'masks/'+str(i)+'.png', mask)
        cv2.imwrite(args.save_path + 'imgs/' + str(i) + '.png', img)
    # for i in range(1,len(list1)):
    #     im = cv2.imread(mask_data_path+list1[i])
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #     cv2.imwrite(mask_data_path+list1[i], im)


if __name__ == '__main__':
    main()
