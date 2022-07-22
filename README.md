# CSGO-AI-AIMBOT

    Python based AI aimbot for Counter-Strike: Global Offensive.

## Warning

This is only PoC and should never be used to gain an advantage in the intented game!

## Installation

### Python

Download Python 3.9.6 [here](https://www.python.org/downloads/release/python-396/)

Select "Windows installer (64-bit)".

When the installer is open don't forget to mark the box with "Add Python 3.9 to PATH".

### CUDA

For the program to work you will also need CUDA Toolkit 11.4.0.
The toolkit can be downloaded [here](https://developer.nvidia.com/cuda-11-4-0-download-archive)

### cuDNN

When installed you will need cuDNN.
The files can be downloaded [here](https://mega.nz/file/4gVnXaTC#SPQdtGCe9lRq0Im6oKjPznD0TJErD4CC25UP7TE34Ug)

Simply drag the folder into the "NVIDIA GPU Computing Toolkit"-folder.
If you installed it without changing install directory it will be located in `C:\Program Files\NVIDIA GPU Computing Toolkit`.

### Requirements

Next you will need to install the python libraries used.
Open your shell, navigate to the folder and type:

```shell
pip install -r requirements.txt
```

## Usage

Raw input has to be off in order for the aimbot to work.
Remember also to turn off "Enhance pointer precision" for no mouse acceleration in your mouse settings

Furthermore, CSGO also have to be in window mode. Either "Windowed Fullscreen" or "In Window".

Edit the `config.ini` file to customize the AI to your liking.

## Troubleshooting

### Error:

```python
File "\CSGO-AI-AIMBOT\lib\site-packages\object_detection\utils\label_map_util.py", line 132,
in load_labelmap
    with tf.gfile.GFile(path, 'r') as fid:
AttributeError: module 'tensorflow' has no attribute 'gfile'
```

### Fix:

Open the python-file given in an IDE to edit it.

Replace

```python
with tf.gfile.GFile(path, 'r') as fid:
```

with

```python
with tf.io.gfile.GFile(path, 'r') as fid:
```

Save the file and run `main.py` again.

## License

MIT License

Copyright (c) [2022] [Patrick Sørensen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
