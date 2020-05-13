# Thorax-Disease-Classification

#### Arguments:
Some of the important arguments are:

`--prep_data`: when used, the actual data will be downloaded and extracted to the `data` folder

`--model`: specifies the pre-trained model to be used in the unified architecture, e.g., `resnet`.

`--eval`: should be used when evaluating the models for drawing ROC curves and computing AUC values.

`--epoch`: specifies which checkpoint should be used for model evaluation

`--lr`: the learning rate for training the model (should also be specified for model evaluation to find the correct checkpoint folder).

`--freezed`: if specified, the parameters of the pre-trained model will be freezed.

#### Example scripts:
Data preparation: `python3 main.py --prep_data`

Training: `python3 main.py --model resnet --lr 2e-6 --freezed`

Evluation: `python3 main.py --eval --model resnet --freezed --lr 2e-6 --epoch 20`

#### Special packages:
Some of the packages needed to be installed include:

- `torchsummary`

#### Notes:
Default parameters like the learning rate are in the `params.json` file, but could be changed if specified in the program arguments (to be explained mode later).
