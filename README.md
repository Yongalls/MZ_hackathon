# MZ_hackathon

## Dependencies
+ python>=3.6
+ torch==1.7.0
+ transformers==4.1.1

## Data preparation

Your directory tree should be look like this:
```
$ROOT
|--model
|--experiments
|   | (Your model will be saved here)
|   | (for training)
|--path_store
|   | (You should move trained model to this directory.)
|   | (for testing)
|   |--bert_base_3.pth
|   |--bert_base_4.pth
|   |--bert_base_12.pth
|   |--bert_base_15.pth
|   |--bert_base_17.pth
|   |--bert_base_18.pth
|   |--bert_base_19.pth
|--train.py
|--predict.py
|--dataloader.py
|--train.txt
|--dev.txt
|--devtest.txt
|--test.txt

```

## Training

```
$ python3 train.py --task baseline --num_labels 784 --save_model_name bert_base_3
$ python3 train.py --task augmented --num_labels 784 --save_model_name bert_base_4 --error_p 0_0.2 --do_augment_per_epoch False
$ python3 train.py --task augmented --num_labels 784 --save_model_name bert_base_12 --error_p 0_0.3 --do_augment_per_epoch False
$ python3 train.py --task augmented --num_labels 784 --save_model_name bert_base_15 --error_p 0_0.2_0.2_0.3_0.3 --do_augment_per_epoch False
$ python3 train.py --task baseline --num_labels 784 --save_model_name bert_base_18 -do_augment_per_epoch True --error_p_scheduling aug1
$ python3 train.py --task baseline --num_labels 784 --save_model_name bert_base_19 -do_augment_per_epoch True --error_p_scheduling aug2

$ python3 train.py --task baseline --num_labels 785 --save_model_name bert_base_17
```

## Prediction

Test with validation dataset (`dev.txt`)
```
$ python3 predict.py --dataset devtest
```
Test with test dataset (`test.txt`)
```
$ python3 predict.py --dataset test
```


## Results

### Fixed Dataset
The number of labels of given dataset is 785. But there are two labels (`2.의료-주차정산-주차비정산`, `5.의료-주차정산-주차비정산`) which are actually same. It might cause trouble in the training process, therefore overlapped label was removed before feeding data into the network. (The number of labels should be 784.) 6 models are trained with the fixed dataset in different augmentation way. 

|Model name|Description|Validation acc (%)|
|---|---|---|
|bert_base_3|baseline|69.408|
|bert_base_4|augmented x2 (20% error)|<strong>70.297</strong>|
|bert_base_12|augmented x2 (30% error)|69.213|
|bert_base_15|augmented x5 (20%, 30% error)|68.975|
|bert_base_18|augmented per epoch (20% error)|69.495|
|bert_base_19|augmented per epoch (linearly increase)|69.029|

### Original Dataset
Although the original dataset has a significant problem, the test dataset includes overlapped label. The network trained by fixed dataset must select randomly between two same labels, which causes negative effect on validation and test accuracy. So the original dataset is also used in training for better selection between the two labels.

|Model name|Description|Validation acc (%)|
|---|---|---|
|bert_base_17|baseline|69.657|

### Ensemble

6 networks trained by the fixed dataset are used for Soft Voting Ensemble. A model trained by the original dataset is exploited when predicting overlapped labels. Also we found 10 additional pairs of labels that have same meaning. To prevent meaningless loss of accuracy, we merged these 10 pairs of labels.

|  |  |
|--|--|
|8.IoT-ON/OFF-공기청정기끄기| 7.IoT-ON/OFF-공기청정기끔|
|7.IoT-ON/OFF-공기청정기작동| 8.IoT-ON/OFF-공기청정기켜기|
|7.IoT-ON/OFF-TV켜기| 8.IoT-ON/OFF-티브이켜기|
|7.IoT-Modechange-에너지절약모드실행| 7.IoT-Modechange-에너지절약모드전환|
|6.반복일상-기상-블라인드내리기| 6.반복일상-기상-블라인드닫기|
|6.반복일상-기상-블라인드열기| 6.반복일상-기상-블라인드올리기|
|5.차량제어-공조제어-에어컨끄기| 2.공조제어-차량에어컨-에어컨끄기|
|5.차량제어-공조제어-에어컨켜기| 2.공조제어-차량에어컨-에어컨켜기|
|5.차량제어-공조제어-히터끄기| 2.공조제어-차량히터-히터끄기|
|5.차량제어-공조제어-히터켜기| 2.공조제어-차량히터-히터켜기|
|2.의료-주차정산-주차비정산| 5.의료-주차정산-주차비정산|

We gained 3.403% additional accuracy compared to single baseline model with original dataset.

| |original|excluding 1 label|excluding 11 labels|
|---|---|---|---|
|Validation acc (%)|72.703|72.757|<strong>73.060</strong>|
