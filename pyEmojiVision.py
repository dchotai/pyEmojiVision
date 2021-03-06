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

EMOJI_FONT_SIZE = 20


def validateFilepath(filepath, parser):
    if not os.path.isfile(filepath):
        parser.error(f"No file found at {filepath}")
    return filepath


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


def downsizeImage(img, scaleFactor):
    image_size = img.size
    new_size = (img.size[0] // scaleFactor, img.size[1] // scaleFactor)
    downsized_image = img.resize(new_size, resample=Image.BILINEAR)
    return downsized_image


def drawEmojiStringToImage(emoji_string, width, height):
    img = Image.new("RGB", (width * EMOJI_FONT_SIZE, height * EMOJI_FONT_SIZE))
    font = ImageFont.truetype(
        "/System/Library/Fonts/Apple Color Emoji.ttc", EMOJI_FONT_SIZE
    )
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), emoji_string, embedded_color=True, font=font, spacing=0)
    return img


def constructOutputPath(input_path):
    full_input_path = os.path.abspath(os.path.expanduser(input_path))
    full_input_path_split = os.path.split(full_input_path)
    output_path = os.path.join(
        full_input_path_split[0],
        f"{os.path.splitext(full_input_path_split[1])[0]}_emoji.jpg"
    )
    # Append a number to the output path to avoid overwriting existing files
    if os.path.exists(output_path):
        count = 1
        output_path_attempt = f" ({count})".join(os.path.splitext(output_path))
        while os.path.exists(output_path_attempt):
            count += 1
            output_path_attempt = f" ({count})".join(os.path.splitext(output_path))
        output_path = output_path_attempt
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", nargs="+", type=lambda i: validateFilepath(i, parser),
        help="File path(s) of the input image(s) to convert into emojis. "
        "Output image(s) will be saved to the same location as the input image(s). "
        "You can pass in multiple, space-separated file paths to convert "
        "multiple images at once."
    )
    parser.add_argument(
        "--emojiPlist", required=True,
        help="Path to DumpEmoji plist file that contains source emojis grouped into categories. These are the `Emoji_iOS<IOS_VERSION>_Simulator_EmojisInCate_<NUM_EMOJIS>.plist` files found at https://github.com/liuyuning/DumpEmoji/tree/master/Emojis"
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

    dominant_emoji_colors = []
    for e in tqdm(emojis, desc="Building emoji color palette"):
        emoji_img = drawEmojiCharToImage(e)
        dominant_color_rgba = getDominantColorFromImage(emoji_img)
        dominant_emoji_colors.append(dominant_color_rgba)

    dominant_emoji_colors = np.array(dominant_emoji_colors)
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(dominant_emoji_colors, emojis)

    for input_file in args.input:
        print(f"Processing {os.path.basename(input_file)}")
        new_image = Image.open(input_file).convert("RGBA")
        downsized_image = downsizeImage(new_image, EMOJI_FONT_SIZE)
        downsized_image_arr = np.asarray(downsized_image).reshape((-1, 4))
        emoji_predictions = knn_classifier.predict(downsized_image_arr)

        width, height = downsized_image.size
        emoji_string = "".join([
            "".join(emoji_predictions[(i * width):((i+1) * width)]) + "\n"
            for i in range(height)
        ])

        output_image = drawEmojiStringToImage(emoji_string, width, height)
        output_path = constructOutputPath(input_file)
        print(f"Saving output to: {output_path}")
        output_image.save(output_path)


if __name__ == "__main__":
    main()
