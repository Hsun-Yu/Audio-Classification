# Audio Classification
## Sound dataset
For this project I will use a dataset called [ESC-50](https://github.com/karolpiczak/ESC-50). The dataset contains 5000 sound and 50 classes.

First you have to download it:
```shell
$ git clone https://github.com/karolpiczak/ESC-50.git
```

## Project sturcture
I have 3 notebooks.
### get_audio_feature
This notebook tells about reading audio and get feature.
### audio_classification
This is the original version, it only uses frequency to classify(1D).
model: DNN
### audio_classification_cnn
This is the CNN version, it uses all MFCC information to classify(2D).
model: CNN
