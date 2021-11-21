# pyEmojiVision

Heavily inspired by [emojivision](https://github.com/gabrieloc/emojivision) and [EmojiVision](https://github.com/ihollander/emoji-vision). Emojis sourced from [DumpEmoji](https://github.com/liuyuning/DumpEmoji/).

## Example usage

```sh
~ python3 pyEmojiVision.py img/oski.jpeg --emojiPlist emojis/Emoji_iOS10.3.1_Simulator_EmojisInCate_1432.plist
```

Install the required Python modules before running the script.
```sh
~ pip3 install -r requirements.txt
```

```
~ python3 pyEmojiVision.py -h
usage: pyEmojiVision.py [-h] --emojiPlist EMOJIPLIST [--emojiSize [{20,32,64}]] [--emojiCategory EMOJICATEGORY] input [input ...]

positional arguments:
  input                 File path(s) of the input image(s) to convert into emojis. Output image(s) will be saved to the same location as the input
                        image(s). You can pass in multiple, space-separated file paths to convert multiple images at once.

optional arguments:
  -h, --help            show this help message and exit
  --emojiPlist EMOJIPLIST
                        Path to DumpEmoji plist file that contains source emojis grouped into categories. These are the
                        `Emoji_iOS<IOS_VERSION>_Simulator_EmojisInCate_<NUM_EMOJIS>.plist` files found at
                        https://github.com/liuyuning/DumpEmoji/tree/master/Emojis
  --emojiSize [{20,32,64}]
                        Emoji font size. Default is 32.
  --emojiCategory EMOJICATEGORY, --category EMOJICATEGORY
                        Category of emojis to be used for generating the output image. If not provided, the default behavior is to use emojis from all
                        categories.
```