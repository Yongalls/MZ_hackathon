import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler

from dataloader import load_dataset, Processor

from transformers import (
    AdamW,
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from model import AlbertForClassification, BertForClassification

MODEL_CLASSES = {
    "bert": (BertConfig, BertForClassification, BertTokenizer),
    "albert": (AlbertConfig, AlbertForClassification, AlbertTokenizer),
}

# paths
root = "/content/drive/MyDrive/Colab Notebooks/MZ_hackathon"
model_dir = root + '/path_store'

model_type = "bert"
config_name = "bert-base-multilingual-cased"
tokenizer_name = "bert-base-multilingual-cased"
model_name = "bert-base-multilingual-cased"

device = torch.device('cuda')
max_seq_len = 50
ignore_index = 0
batchsize = 24
dropout_rate = 0 # apply for classification layer


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    out = exp_x / sum_exp_x
    return out


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(model_name, model):
    state = torch.load(os.path.join(model_dir, model_name))
    model.load_state_dict(state['model'])
    print('model {} loaded. checkpoint state: epoch {}, step {}, best acc {}'.format(model_name, state['epoch'], state['step'], state['best_acc']))
    return model



def predict(val_dataset, models, model_785, dict_labels_inv_784, dict_labels_inv_785):
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batchsize)

    accs = AverageMeter()

    models[0] = load_model('bert_base_3.pth', models[0])
    models[1] = load_model('bert_base_4.pth', models[1])
    models[2] = load_model('bert_base_12.pth', models[2])
    models[3] = load_model('bert_base_15.pth', models[3])
    models[4] = load_model('bert_base_18.pth', models[4])
    models[5] = load_model('bert_base_19.pth', models[5])

    model_785 = load_model('bert_base_17.pth', model_785)



    for model in models:
        model.eval()
    model_785.eval()

    count = 0
    with open(os.path.join(root, 'result.txt'), 'w', encoding='utf-8') as f:
        for step, batch in enumerate(val_dataloader):

            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }

                outputs = []
                for model in models:
                    output = model(**inputs)
                    outputs.append(output.data.cpu().numpy())

                output_785 = model_785(**inputs)

                bs = outputs[0].shape[0]

                pred_numpy = np.zeros(outputs[0].shape)
                for output in outputs:
                    pred_numpy += output

                pred_numpy = np.argmax(pred_numpy, axis=1)
                pred_numpy_785 = output_785.data.cpu().numpy()
                pred_numpy_785 = np.argmax(pred_numpy_785, axis=1)

                for i in range(bs):
                    if dict_labels_inv_784[pred_numpy[i]].split('.')[1] == "의료-주차정산-주차비정산":
                        if dict_labels_inv_785[pred_numpy_785[i]].split('.')[1] == "의료-주차정산-주차비정산":
                            f.write(dict_labels_inv_785[pred_numpy_785[i]] + '\n')
                        else:
                            f.write(dict_labels_inv_784[pred_numpy[i]] + '\n')
                    else:
                        f.write(dict_labels_inv_784[pred_numpy[i]] + '\n')
                    count += 1
    print("Num of data: ", count)


def calculate_accuracy(pred_file_name, answer_file_name):
    dict_labels = {
        "8.IoT-ON/OFF-공기청정기끄기": 0, "7.IoT-ON/OFF-공기청정기끔": 0,
        "7.IoT-ON/OFF-공기청정기작동": 1, "8.IoT-ON/OFF-공기청정기켜기": 1,
        "7.IoT-ON/OFF-TV켜기": 2, "8.IoT-ON/OFF-티브이켜기": 2,
        "7.IoT-Modechange-에너지절약모드실행": 3, "7.IoT-Modechange-에너지절약모드전환": 3,
        "6.반복일상-기상-블라인드내리기": 4, "6.반복일상-기상-블라인드닫기": 4,
        "6.반복일상-기상-블라인드열기": 5, "6.반복일상-기상-블라인드올리기": 5,
        "5.차량제어-공조제어-에어컨끄기": 6, "2.공조제어-차량에어컨-에어컨끄기": 6,
        "5.차량제어-공조제어-에어컨켜기": 7, "2.공조제어-차량에어컨-에어컨켜기": 7,
        "5.차량제어-공조제어-히터끄기": 8, "2.공조제어-차량히터-히터끄기": 8,
        "5.차량제어-공조제어-히터켜기": 9, "2.공조제어-차량히터-히터켜기": 9,
        "2.의료-주차정산-주차비정산": 10, "5.의료-주차정산-주차비정산": 10
    }

    answers = []
    preds = []

    with open(os.path.join(root, answer_file_name), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label = line.strip().split('\t')[0]
            answers.append(label)

    with open(os.path.join(root, pred_file_name), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label = line.strip().split('\t')[0]
            preds.append(label)

    assert len(answers) == len(preds)

    num_pass_785 = 0
    num_pass_774 = 0

    for i in range(len(answers)):
        if answers[i] == preds[i]:
            num_pass_785 += 1
        if (preds[i] in dict_labels and answers[i] in dict_labels and dict_labels[preds[i]] == dict_labels[answers[i]]) or preds[i] == answers[i]:
            num_pass_774 += 1

    acc_785 = num_pass_785 / len(answers) * 100
    acc_774 = num_pass_774 / len(answers) * 100

    return acc_785, acc_774



def main(args):

    # set seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    config = config_class.from_pretrained(config_name)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

    models = []
    for i in range(6):
        model = model_class.from_pretrained(model_name, config=config, num_labels=784, dropout_rate=dropout_rate)
        model.to(device)
        models.append(model)

    model_785 = model_class.from_pretrained(model_name, config=config, num_labels=785, dropout_rate=dropout_rate)
    model_785.to(device)

    processor = Processor()
    dict_labels_inv_784 = processor.dict_labels_inv_784
    dict_labels_inv_785 = processor.dict_labels_inv_785


    print("predict start")
    val_dataset = load_dataset('test', args.dataset + '.txt', 784, processor, max_seq_len, tokenizer, ignore_index)
    predict(val_dataset, models, model_785, dict_labels_inv_784, dict_labels_inv_785)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default=None, required=True, type=str, help="The dataset to predict")

    args = parser.parse_args()

    main(args)
