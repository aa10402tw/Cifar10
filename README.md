# Cifar10
Implement of some CNN network and modify some of them to make the parameters fewer.

## How to Train
For example, if you want to train SimpleResNeXt_v2 for 100 epochs, and start with learning rate 0.1
You can type as following:
```
python main.py -model=SimpleResNeXt_v2 -n_epochs=100 =lr=0.1 
```