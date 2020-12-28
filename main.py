import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from dataloader import load_dataset

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

# constatns
num_labels = 784 #785

# global config (args)
device = torch.device('cuda')
gradient_accumulation_steps = 1
max_grad_norm = 1.0

num_train_epochs = 2
batchsize = 24
logging_steps = 100
evaluate_during_training = False #True
weight_decay = 0
learning_rate = 1e-5
adam_epsilon = 1e-8
warmup_steps = 0


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


def train(train_dataset, model, tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batchsize)

    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    tr_loss, logging_loss = 0.0, 0.0
    global_step = 0
    model.zero_grad()
    best_f1, best_exact = -1, -1

    for epoch in range(num_train_epochs):
        losses = AverageMeter()
        accs = AverageMeter()

        for step, batch in enumerate(train_dataloader):

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
            }
            labels = batch[3]

            # if args.model_type in ["xlm", "roberta", "distilbert"]:
            #     del inputs["token_type_ids"]
            #
            # if args.model_type in ["xlnet", "xlm"]:
            #     inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
            #     if args.version_2_with_negative:
            #         inputs.update({"is_impossible": batch[7]})

            output = model(**inputs)
            bs = output.size(0)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(output.view(-1, num_labels), labels.view(-1))

            pred_numpy = output.data.cpu().numpy()
            pred_numpy = np.argmax(pred_numpy, axis=1)
            label_numpy = labels.data.cpu().numpy()
            acc = np.sum(pred_numpy == label_numpy) / bs

            # AverageMeter update
            losses.update(loss.data.cpu(), bs)
            accs.update(acc, bs)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()


            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if logging_steps > 0 and global_step % logging_steps == 0:
                    print("Traininng epoch {}, step {}, loss ({}/{}), accuracy ({}/{})". format(epoch, global_step, losses.val, losses.avg, accs.val, accs.avg))
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if evaluate_during_training:
                        logger.info("Validation start for epoch {}".format(epoch))
                        result = evaluate(args, model, tokenizer, prefix=epoch)
                        _f1, _exact = result["f1"], result["exact"]
                        is_best = _f1 > best_f1
                        best_f1 = max(_f1, best_f1)

                        current_loss = (tr_loss - logging_loss) / logging_steps
                        logging_loss = tr_loss

                        logger.info(
                            "best_f1_val = {}, f1_val = {}, exact_val = {}, loss = {}, global_step = {}, " \
                            .format(best_f1, _f1, _exact, current_loss, global_step))


    return global_step, losses.avg


def main():

    # set seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # config
    mode = "train"
    model_type = "bert"

    config_name = "bert-base-multilingual-cased"
    tokenizer_name = "bert-base-multilingual-cased"
    model_name = "bert-base-multilingual-cased"

    # hyper-parameter
    max_seq_len = 50
    ignore_index = 0
    dropout_rate = 0.1 # apply for classification layer

    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

    config = config_class.from_pretrained(config_name)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    model = model_class.from_pretrained(model_name,
        config=config,
        num_labels=num_labels,
        dropout_rate=dropout_rate
    )

    train_dataset = load_dataset('train', max_seq_len, tokenizer, ignore_index)

    model.to(device)

    if mode == "train":
        print("train start")
        train(train_dataset, model, tokenizer)
    elif mode == "val":
        print("validation start")
    else:
        print("test start")

if __name__ == '__main__' :
    main()
