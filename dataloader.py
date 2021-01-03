import random
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import os
import math

from transformers import BertTokenizer

root = "/content/drive/MyDrive/Colab Notebooks/MZ_hackathon"

# set seed
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class InputExample(object):
    def __init__(self, text, label):
        self.words = text.split() # fix later
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids


class Processor(object):

    def __init__(self):
        self.dict_labels, self.dict_labels_inv = self.create_dict()
        assert len(self.dict_labels) == 784
        self.dif_words, self.num_data = self.create_vocab_change()
        print(len(self.dif_words))

    def create_dict(self):
        dict_labels = {}
        dict_labels_inv = {}
        class_id = 0

        with open(os.path.join(root, 'train.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                label = line.strip().split('\t')[0].split('.')[1]
                if label not in dict_labels:
                    dict_labels[label] = class_id
                    dict_labels_inv[class_id] = label
                    class_id += 1

        return dict_labels, dict_labels_inv

    def create_vocab_change(self):
        lines = []

        with open(os.path.join(root, 'train.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                lines.append(line.strip())

        num_data = int(len(lines) / 3)
        assert num_data == 27904

        dif_words = {}

        for i in range(num_data):
            if len(lines[i].split('\t')) != 2 or len(lines[num_data + i].split('\t')) != 2 or len(lines[2 * num_data + i].split('\t')) != 2:
                print(lines[i] + '\n')
                continue
            text1 = lines[i].split('\t')[1].split()
            text2 = lines[num_data + i].split('\t')[1].split()
            text3 = lines[2 * num_data + i].split('\t')[1].split()
            if len(text1) != len(text2) or len(text2) != len(text3) or len(text3) != len(text1):
                continue
            for w in range(len(text1)):
                if text1[w] != text2[w]:
                    if text1[w] not in dif_words:
                        dif_words[text1[w]] = [text2[w]]
                    elif text2[w] not in dif_words[text1[w]]:
                        dif_words[text1[w]].append(text2[w])
                    if text2[w] not in dif_words:
                        dif_words[text2[w]] = [text1[w]]
                    elif text1[w] not in dif_words[text2[w]]:
                        dif_words[text2[w]].append(text1[w])
                if text2[w] != text3[w]:
                    if text2[w] not in dif_words:
                        dif_words[text2[w]] = [text3[w]]
                    elif text3[w] not in dif_words[text2[w]]:
                        dif_words[text2[w]].append(text3[w])
                    if text3[w] not in dif_words:
                        dif_words[text3[w]] = [text2[w]]
                    elif text2[w] not in dif_words[text3[w]]:
                        dif_words[text3[w]].append(text2[w])
                if text1[w] != text3[w]:
                    if text1[w] not in dif_words:
                        dif_words[text1[w]] = [text3[w]]
                    elif text3[w] not in dif_words[text1[w]]:
                        dif_words[text1[w]].append(text3[w])
                    if text3[w] not in dif_words:
                        dif_words[text3[w]] = [text1[w]]
                    elif text1[w] not in dif_words[text3[w]]:
                        dif_words[text3[w]].append(text1[w])

        return dif_words, num_data

    def get_examples(self, mode):
        examples = []
        file_name = mode + '.txt'

        with open(os.path.join(root, file_name), 'r', encoding='utf-8') as f:
            i = 0
            for line in f.readlines():
                # ex_id = "%s-%s" % (mode, i)
                # if i > 5000:
                #     break
                if len(line.strip().split('\t')) != 2:
                    print(i, line.strip(), len(line.strip().split('\t')))
                    continue
                i += 1
                text = line.strip().split('\t')[1]
                label = line.strip().split('\t')[0].split('.')[1]
                label_id = self.dict_labels[label]
                examples.append(InputExample(text=text, label=label_id))

        print(len(examples))

        return examples

    def get_augmented_examples(self, p):
        examples = []
        lines = []

        with open(os.path.join(root, 'train.txt'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                lines.append(line.strip())

        for i in range(self.num_data):
            label = lines[i].split('\t')[0].split('.')[1]
            label_id = self.dict_labels[label]
            for j in range(3):
                if len(lines[j * self.num_data + i].split('\t')) != 2:
                    continue
                text = lines[j * self.num_data + i].split('\t')[1].split()
                changed = []
                for word in text:
                    if word in self.dif_words.keys():
                        r = random.uniform(0, 1)
                        if r < p:
                            changed.append(random.choice(self.dif_words[word]))
                        else:
                            changed.append(word)
                    else:
                        changed.append(word)
                changed = ' '.join(changed)
                examples.append(InputExample(text=changed, label=label_id))

        print(len(examples))

        return examples


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenization -> fix later
        tokens = []
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)


        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_ids = example.label

        if ex_index < 5:
            print("*** Example ***")
            # print("guid: %s" % example.guid)
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("intent_label: %d" % (label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids,
                          ))

    return features


def load_dataset(mode, processor, max_seq_len, tokenizer, ignore_index):
    examples = processor.get_examples(mode)

    features = convert_examples_to_features(examples, max_seq_len, tokenizer,
                                            pad_token_label_id=ignore_index)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

    return dataset

def load_augmented_dataset(processor, p, max_seq_len, tokenizer, ignore_index):
    examples = processor.get_augmented_examples(p)

    features = convert_examples_to_features(examples, max_seq_len, tokenizer,
                                            pad_token_label_id=ignore_index)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

    return dataset

if __name__ == '__main__':
    plt.figure(figsize=(20,5))
    train_len_tokens = []
    val_len_tokens = []
    label_len_tokens = []

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    unk_token = tokenizer.unk_token
    processor = Processor()


    line = '모든 좌석 시트 통풍 오프.'
    words = line.split()
    tokens = []
    for word in words:
        word_tokens = tokenizer.tokenize(word)
        if not word_tokens:
            word_tokens = [unk_token]  # For handling the bad-encoded word
        tokens.extend(word_tokens)
    print(tokens)


    # print("load train dataset")
    # examples = processor.get_examples('train')
    # for (ex_index, example) in enumerate(examples):
    #     tokens = []
    #     for word in example.words:
    #         word_tokens = tokenizer.tokenize(word)
    #         if not word_tokens:
    #             word_tokens = [unk_token]  # For handling the bad-encoded word
    #         tokens.extend(word_tokens)

    #     # Account for [CLS] and [SEP]
    #     special_tokens_count = 2
    #     train_len_tokens.append(len(tokens) + special_tokens_count)

    # print("Maximum length of tokens: {}, Minimum length of tokens: {}". format(max(train_len_tokens), min(train_len_tokens)))
    # plt.hist(train_len_tokens, bins=range(0,40,2))

    # print("load val dataset")
    # examples = processor.get_examples('dev')
    # for (ex_index, example) in enumerate(examples):
    #     tokens = []
    #     for word in example.words:
    #         word_tokens = tokenizer.tokenize(word)
    #         if not word_tokens:
    #             word_tokens = [unk_token]  # For handling the bad-encoded word
    #         tokens.extend(word_tokens)

    #     # Account for [CLS] and [SEP]
    #     special_tokens_count = 2
    #     val_len_tokens.append(len(tokens) + special_tokens_count)

    # print("Maximum length of tokens: {}, Minimum length of tokens: {}". format(max(val_len_tokens), min(val_len_tokens)))
    # plt.hist(val_len_tokens, bins=range(0,40,2))

    # plt.savefig(root + '/tokens_length_distribution.png')

    # plt.clf()
    # print("load label")
    # dict_labels = processor.dict_labels

    # with open(os.path.join(root, 'metainfo.txt'), 'w', encoding='utf-8') as f:
    #     for (label, i) in dict_labels.items():
    #         tokens = []
    #         labels = label.split('-')
    #         for word in labels:
    #             assert len(word.split()) == 1
    #             word_tokens = tokenizer.tokenize(word)
    #             if not word_tokens:
    #                 word_tokens = [unk_token]  # For handling the bad-encoded word
    #             tokens.extend(word_tokens)
    #         f.write(str(i) + ': ' + label + '\t' + ' '.join(tokens) + '\n')

    #         # Account for [CLS] and [SEP]
    #         special_tokens_count = 2
    #         val_len_tokens.append(len(tokens) + special_tokens_count)


    # print("Maximum length of tokens: {}, Minimum length of tokens: {}". format(max(label_len_tokens), min(label_len_tokens)))
    # plt.hist(label_len_tokens, bins=range(0,40,2))

    # plt.savefig(root + '/label_tokens_length_distribution.png')
