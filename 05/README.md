# Rock Paper Scissor Classifier

## Overview

This homework demonstrates how to classify images of rock, paper, scissors(scissors1, scissors2) using a neural network model trained with Teachable Machine.

There are two python scripts:

- rock-paper-scissors.py: This takes an image as an input and classifies it as a rock, paper or scissors.

- rock-paper-scissors-live.py: This code uses a webcam to classify the rock, paper or scissors in real time.

### Project Structure

- 29.jpg -> scissors1 image.
- keras_model.h5 -> Teachable Machine's trained model.
- labels.txt
- rock-paper-scissors.py
- rock-paper-scissors-live.py
- Teachable.ipynb -> For Practice

### Prerequisites

- Python 3.11.7
- pip
- Tensorflow
- Numpy
- Pillow
- Matplotlib
- OpenCV
- New conda environment (Optional)

### Usage

Example Input: To test a single image you can specify the location after the `python rock-paper-scissors.py "HERE"`.

```bash
python rock-paper-scissors.py 29.jpg
```

Example Output:

```bash
1/1 [==============================] - 0s 436ms/step
Class: scissors1
Confidence Score: 0.9999156
```

Real Time Input:

```bash
python rock-paper-scissors-live.py
```

You can press `q` to stop the real time feed.

#### Youtube Link : ([Link](https://youtu.be/FYHRKWBeXyA))
