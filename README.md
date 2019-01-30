## Kaggle Plant Seeding with Pytorch

<div align=center>
<img src='https://ws2.sinaimg.cn/large/006tNbRwly1fwdpjbncupj31i40ew0vk.jpg' width='500'>
</div>

### Introduction
In this project, we use the `Resnet50` with Pytorch to do the plant seed classification.For this task, the dataset is large and the image resolution is higher, the network is deeper.

### Git Project

open the terminal and run the command

```
git clone https://github.com/JimmyHHua/plantseed_pytorch.git
```

or you can download it via the website, and then run the below command

```bash
pip3 install -r requirements.txt
```

### Data Download
Download the data via the website (https://www.kaggle.com/c/plant-seedlings-classification/data)

<div align=center>
<img src='https://ws4.sinaimg.cn/large/006tNbRwly1fwdo7019xfj31kw13owgy.jpg' width='500'>
</div>

makedirs `data`，and copy these data into the `data`，then running the below command to get the data what we want.

```bash
cd data;
unzip -q train.zip; cp -r train train_valid;
unzip -q test.zip
cd ..; python preprocess.py
```
It may cost several minutes ...

### Training
Running the code:

```
python train.py --bs=128  # gpu Training
```

`bs` represent batch size, `use_gpu` means using gpu，and for other parameters, please refer to `train.py`. During the training process, the weights will be saved in `checkpoints` automatically.

### Submission

Afther the training, we could load the best weights we got before, and run the below command to archive the submission.csv file. Then we just need to submit this file to the Kaggle, and we will get the score in seconds.

```
python submit.py --model_path='checkpoints/model_best.pt'
```

<div align=center>
<img src='https://i.loli.net/2019/01/30/5c5155f7b9294.png' width='800'>
</div>
