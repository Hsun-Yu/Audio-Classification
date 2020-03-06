# Audio Classification
## Sound dataset
For this project I will use a dataset called [ESC-50](https://github.com/karolpiczak/ESC-50). The dataset contains 5000 sound and 50 classes.

First you have to download it:
```shell
$ git clone https://github.com/karolpiczak/ESC-50.git
```

## Project sturcture
I have 3 notebooks.
### [get_audio_feature](https://github.com/Hsun-Yu/Audio-Classification/blob/master/get_audio_feature.ipynb)
This notebook tells about reading audio and get feature.
### [audio_classification](https://github.com/Hsun-Yu/Audio-Classification/blob/master/audio_classification.ipynb)
This is the original version, it only uses frequency to classify(1D).
model: DNN
### [audio_classification_cnn](https://github.com/Hsun-Yu/Audio-Classification/blob/master/audio_classification_cnn.ipynb)
This is the CNN version, it uses all MFCC information to classify(2D).
model: CNN
