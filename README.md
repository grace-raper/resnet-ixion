##  Image Orientation Detection Using ResNet


Try it in [Replicate](https://replicate.com/grace-raper/resnet-rot)!

Or download the model from [GitHub Releases](https://github.com/parsapoorsh/resnet-ixion/releases):

## Intro
this model is a fine-tuned variant of resnet152.

it's been converted to a classfication model with 4 classes. `0°, 90°, 180°, 270°`.

![image](https://raw.githubusercontent.com/parsapoorsh/resnet-ixion/refs/heads/master/README.jpg)


### Training Data
for fine-tunning this model, i used MS COCO (Microsoft Common Objects in Context) dataset.
`train2017` and `test2017` for training, and `val2017` for validation.

### Training Hardware
GPU: `NVIDIA GTX 1660 SUPER 6 GIB VRAM`

time per epoch: ~ 3h:45m

## Licence
same licence as resnet. `apache-2.0`
