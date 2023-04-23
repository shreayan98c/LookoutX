# LookoutX
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](<https://opensource.org/licenses/MIT>)

A smart assistant for the visually impaired, to make their lives easier.

Goal is to guide the visually impaired user via audio by answering their questions and queries and providing them with the response.

## Technologies used:
OpenCV for capturing and processing real-time video
SpeechRecognition for audio to text
CLIP for model
pyttsx for converting text to speech

## Data Pipeline
Data Collection for video stream and audio stream -> SpeechRecognition speech to text -> Model Training -> Model Evaluation -> Py text to speech

## Installation

These instructions assume a working installation of [Anaconda](https://www.anaconda.com/).

```bash
git clone git@github.com:shreayan98c/LookoutX.git
cd LookoutX
conda env create -f environment.yml
```

Depending on your desired configuration, you may need to install the
[PyTorch](https://pytorch.org/get-started/locally/) package separately. This can be done following
the instructions on the PyTorch website, in an empty conda environment. Then you can install the
remaining packages with:

```bash
conda activate lookoutx
pip install -r requirements.txt
pip install git+https://github.com/zphang/transformers.git@68d640f7c368bcaaaecfc678f11908ebbd3d6176
pip install -e .
```

This is only necessary if the installation from `environment.yml` fails.

## Usage

```bash
python main.py train
```

## License

This project is licensed under the terms of the MIT license.