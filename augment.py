import os
import random

root = "/content/drive/MyDrive/Colab Notebooks/MZ_hackathon"

# set seed
seed = 123
random.seed(seed)

p = 0.2
num_data = 27904

lines = []

with open(os.path.join(root, 'train.txt'), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        lines.append(line.strip())

assert len(lines) == 3 * num_data

dif_words = {}

for i in range(num_data):
    if len(lines[i].split('\t')) != 2 or len(lines[num_data + i].split('\t')) != 2 or len(lines[2 * num_data + i].split('\t')) != 2:
        print(lines[i] + '\n')
        continue
    text1 = lines[i].split('\t')[1].split()
    text2 = lines[num_data + i].split('\t')[1].split()
    text3 = lines[2 * num_data + i].split('\t')[1].split()
    if len(text1) != len(text2) or len(text2) != len(text3) or len(text3) != len(text1):
        # print(text1)
        # print(text2)
        # print(text3)
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

count = 0
with open(os.path.join(root, 'aug_train.txt'), 'w', encoding='utf-8') as f:
    for i in range(num_data):
        label = lines[i].split('\t')[0]
        for j in range(3):
            if len(lines[j * num_data + i].split('\t')) != 2:
                continue
            text = lines[j * num_data + i].split('\t')[1].split()
            changed = []
            for word in text:
                if word in dif_words.keys():
                    r = random.uniform(0, 1)
                    if r < p:
                        changed.append(random.choice(dif_words[word]))
                    else:
                        changed.append(word)
                else:
                    changed.append(word)
            changed = ' '.join(changed)
            f.write(lines[j * num_data + i] + '\n')
            f.write(label + '\t' + changed + '\n')
            count += 2

print(count)
