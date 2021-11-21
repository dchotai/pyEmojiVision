#!/usr/bin/env python3
import argparse
import numpy as np
import os
import plistlib

from collections import Counter
from PIL import Image, ImageFont, ImageDraw
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


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


def drawEmojiCharToImage(emoji_char, size=(32, 32)):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 32)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), emoji_char, embedded_color=True, font=font)
    # img.show()
    return img


def getDominantColorFromImage(image):
    # arr.size is (width, height, 4 values for RGBA)
    arr = np.asarray(image)
    # arr_reshaped.size is (width * height, 4 values for RGBA)
    arr_reshaped = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])
    # Filter out pixels that have an alpha value under 128 (less than 50% opacity)
    arr_filtered = arr_reshaped[arr_reshaped[:, 3] >= 128]

    # Construct color centroids and find the most common centroid,
    # which represents the image's dominant color.
    kmeans = KMeans(n_clusters=1).fit(arr_filtered)
    most_common_centroid_label = Counter(kmeans.labels_).most_common(1)[0][0]
    dominant_color_vector = kmeans.cluster_centers_[most_common_centroid_label]
    dominant_color_rgba = [int(c) for c in dominant_color_vector]
    return dominant_color_rgba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emojiPlist", required=True,
        help="Path to DumpEmoji plist file that contains source emojis grouped into categories. These are the `Emoji_iOS<IOS_VERSION>_Simulator_EmojisInCate_<NUM_EMOJIS>.plist` files found at https://github.com/liuyuning/DumpEmoji/tree/master/Emojis"
    )
    parser.add_argument(
        "--emojiSize", required=False, default=32, const=32, # TODO
        nargs="?", choices=(20, 32, 64), help="Emoji font size. Default is 32."
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

    emoji_dominant_colors = []
    for e in tqdm(emojis, desc="Building emoji color palette"):
        emoji_img = drawEmojiCharToImage(e)
        dominant_color_rgba = getDominantColorFromImage(emoji_img)
        emoji_dominant_colors.append(dominant_color_rgba)

    emoji_dominant_colors = np.array(emoji_dominant_colors)
    knn_classifier = KNeighborsClassifier(n_neighbors=9)
    knn_classifier.fit(emoji_dominant_colors, emojis)


if __name__ == "__main__":
    main()
