# pyEmojiVision

Heavily inspired by [emojivision](https://github.com/gabrieloc/emojivision) and [EmojiVision](https://github.com/ihollander/emoji-vision). Emojis sourced from [DumpEmoji](https://github.com/liuyuning/DumpEmoji/).

## Example usage

```sh
~ python3 pyEmojiVision.py --emojiPlist emojis/Emoji_iOS10.3.1_Simulator_EmojisInCate_1432.plist
```

Install the required Python modules before running the script.
```sh
~ pip3 install -r requirements.txt
```

```sh
~ python3 pyEmojiVision.py
usage: pyEmojiVision.py [-h] --emojiPlist EMOJIPLIST [--numClusters NUMCLUSTERS] [--emojiCategory EMOJICATEGORY]

optional arguments:
  -h, --help            show this help message and exit
  --emojiPlist EMOJIPLIST
                        Path to DumpEmoji plist file that contains source emojis grouped into categories. These are the
                        `Emoji_iOS<IOS_VERSION>_Simulator_EmojisInCate_<NUM_EMOJIS>.plist` files found at
                        https://github.com/liuyuning/DumpEmoji/tree/master/Emojis
  --numClusters NUMCLUSTERS
                        Number of clusters to use for finding the dominant color of an emoji using k-means clustering.
  --emojiCategory EMOJICATEGORY, --category EMOJICATEGORY
                        Category of emojis to be used for generating the output image. If not provided, the default behavior is to use emojis from all
                        categories.
```