# Cifar10

Implement of some CNN network and modify some of them to make the parameters fewer.

SimpleResNeXt_v1 need only 9,834 parameters but can get about 74.69% top-1 accuracy, and 98.67% top-5 accuracy.

SimpleResNeXt_v2 need only  850  parameters but can get about 51.59 % top-1 accuracy, and 93.78%  top-5 accuracy.


## How to Train

For example, if you want to train SimpleResNeXt_v2 for 100 epochs, and start with learning rate 0.1

You can type as following:

```
python main.py -model=SimpleResNeXt_v2 -n_epochs=100 -lr=0.1 
```

ote: learning rate will decay 0.05 for every epochs.