# Cifar10

Implement some CNN networks and modify ResNeXt to make the as few as possible subject to the top-5 accuracy must exceed 90%. 

The modified models are called **SimpleResNeXt_v1** and **SimpleResNeXt_v2**.

|  model name |# params|top-1 acc|top-5 acc|
| :-------------: | :-------------: | :-------------: | :-------------: |
| SimpleResNeXt_v1|9,834|74.69%|98.67%|
| SimpleResNeXt_v2|850|51.59%|93.78%|

## Training

To train SimpleResNeXt_v2 for 100 epochs, and start with learning rate 0.1

You can execute the following command

```
python main.py -model=SimpleResNeXt_v2 -n_epochs=100 -lr=0.1 
```

Note: learning rate will decay 0.05 for every epoch.

## Prediction 

Put the images you want to predict in the test_imgs folder, and choose which model you want to use:

```
python prediction.py -model=SimpleResNeXt_v2
```