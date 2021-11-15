#!/usr/bin/env python3
import argparse
import numpy as np
import os
import plistlib

from collections import Counter
from PIL import Image, ImageFont, ImageDraw
from scipy.cluster.vq import kmeans, vq


def extractEmojisFromPlist(filepath, parser):
    if not(
        os.path.isfile(filepath) and os.path.splitext(filepath)[1] == ".plist"
    ):
        parser.error(f"No plist file found at: {filepath}")
    with open(filepath, "rb") as f:
        emoji_dict = plistlib.load(f)
        if type(emoji_dict) is not dict:
            parser.error(
                f"No emoji dictionary found in the input plist file at: {filepath}"
            )
        return emoji_dict


def drawEmojiCharToImage(emoji_char, size=(64, 64)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 64)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), emoji_char, embedded_color=True, font=font)
    # img.show()
    return img


def getDominantColorFromImage(image, numClusters):
    # arr.size is (width, height, 4 values for RGBA)
    arr = np.asarray(image)
    # arr_reshaped.size is (width * height, 4 values for RGBA)
    arr_reshaped = arr.reshape(
        arr.shape[0] * arr.shape[1], arr.shape[2]
    ).astype(float)
    # Filter out pixels that have an alpha value under 128 (less than 50% opacity)
    arr_filtered = arr_reshaped[arr_reshaped[:, 3] >= 128]

    # Construct color centroids and find the most common centroid,
    # which represents the image's dominant color.
    color_centroids, distortion = kmeans(arr_filtered, numClusters)
    centroid_indexes, distortion_values = vq(arr_filtered, color_centroids)
    most_common_centroid_index = Counter(centroid_indexes).most_common(1)[0][0]
    dominant_color_vector = color_centroids[most_common_centroid_index]
    dominant_color_rgba = [int(c) for c in dominant_color_vector]
    return dominant_color_rgba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emojiPlist", required=True,
        help="Path to DumpEmoji plist file that contains source emojis grouped into categories. These are the `Emoji_iOS<IOS_VERSION>_Simulator_EmojisInCate_<NUM_EMOJIS>.plist` files found at https://github.com/liuyuning/DumpEmoji/tree/master/Emojis"
    )
    parser.add_argument(
        "--numClusters", required=False, default=5, type=int,
        help="Number of clusters to use for finding the dominant color of an emoji using k-means clustering."
    )
    parser.add_argument(
        "--emojiCategory", "--category", required=False,
        help="Category of emojis to be used for generating the output image. If not provided, the default behavior is to use emojis from all categories."
    )
    args = parser.parse_args()

    emojis = extractEmojisFromPlist(args.emojiPlist, parser)
    if args.emojiCategory:
        if args.emojiCategory not in emojis.keys():
            available_categories = "\n\t" + "\n\t".join(emojis.keys())
            parser.error(
                f"Invalid emojiCategory supplied. Please supply one of the following categories: {available_categories}"
            )
        emojis = emojis[args.emojiCategory]
    else:
        # Flatten all emojis into a single list
        emojis = [
            emoji for categoryEmojis in emojis.values()
            for emoji in categoryEmojis
        ]
    
    for e in emojis:
        img = drawEmojiCharToImage(e)
        # img = drawEmojiCharToImage(emojis[10])
        dominant_color_rgba = getDominantColorFromImage(img, args.numClusters)
        # print(dominant_color_rgba)
        # break


if __name__ == "__main__":
    main()
