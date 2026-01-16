import copy
import datetime
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

from transformers import BertModel, BertTokenizer

from visual import main_attn

device = "cuda:0"
task_ = 2
seed = 110

sent2id = {"负面": 0, "正面": 1}
id2sent = {0: "负面", 1: "正面"}

if task_ == 1:
    # https://gitee.com/dianzck/nlp-option/
    data_path = "../nlp_option_sentiment.jsonl"
    label2id = {"O": 0, "B-Aspect": 1, "I-Aspect": 2, "B-Opinion": 3, "I-Opinion": 4, "[CLS]": 5, "[SEP]": 6}
    id2label = {0: "O", 1: "B-Aspect", 2: "I-Aspect", 3: "B-Opinion", 4: "I-Opinion", 5: "[CLS]", 6: "[SEP]"}
elif task_ == 2:
    data_path = "../opi_mini_sentiment.jsonl"
    label2id = {"O": 0, "B-ASP": 1, "I-ASP": 2, "B-OPI": 3, "I-OPI": 4, "[CLS]": 5, "[SEP]": 6}
    id2label = {0: "O", 1: "B-ASP", 2: "I-ASP", 3: "B-OPI", 4: "I-OPI", 5: "[CLS]", 6: "[SEP]"}

num_label = len(label2id)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OM_dataset(Dataset):
    def __init__(self, data, label, seq, sent_label):
        self.data = data
        self.label = label
        self.seq = seq
        self.sent_label = sent_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item], self.seq[item], self.sent_label[item]


class BatchTextCall(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("../bert-base-chinese")

    def __call__(self, batch):
        data = [{"text": te[0]} for te in batch]
        label = [{"label": [5] + te[1] + [6]} for te in batch]
        seq = [{"seq": ["[CLS]"] + te[2] + ["SEP"]} for te in batch]
        sent_label = [te[3] for te in batch]
        for (e, e_label) in zip(data, label):
            tokenize = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + list(e['text']) + ["[SEP]"])
            e['bert_id'] = np.array(tokenize)
            e['attn_mask'] = np.ones(len(tokenize))
            zero_indices = np.where(np.array(e_label["label"]) == 0)[0].tolist()
            e['aug_id'] = self.Augmented_data(tokenize, zero_indices)

        text_len = np.array([len(e['bert_id']) for e in data])
        max_text_len = max(text_len)

        text = np.zeros([len(data), max_text_len], dtype=np.int64)
        text_mask = np.zeros([len(data), max_text_len], dtype=np.int64)
        aug_text = np.zeros([len(data), max_text_len], dtype=np.int64)

        label_new = np.zeros([len(label), max_text_len], dtype=np.int64)
        seq_new = np.zeros([len(seq), max_text_len], dtype=object)

        for i in range(len(data)):
            text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']
            text_mask[i, :len(data[i]['attn_mask'])] = data[i]['attn_mask']
            aug_text[i, :len(data[i]['aug_id'])] = data[i]['aug_id']

        for i in range(len(label)):
            label_new[i, :len(label[i]["label"])] = label[i]["label"]

        for i in range(len(label)):
            seq_new[i, :len(seq[i]["seq"])] = seq[i]["seq"]
            seq_new[i, (len(seq[i]["seq"]) + 1):] = "O"

        new_data = {
            "input_ids": torch.tensor(text).to(device),
            "attention_mask": torch.tensor(text_mask, dtype=torch.float32).to(device),
            "aug_input_ids": torch.tensor(aug_text).to(device)
        }

        label = torch.tensor(label_new).to(device)
        sent_label = torch.tensor(sent_label).to(device)
        seq = seq_new.tolist()

        return new_data, label, seq, sent_label

    def Augmented_data(self, tokenize, indices):

        for idx in indices:
            if random.uniform(0, 1) < 0.35:
                random_words = np.random.randint(0, len(self.tokenizer))
                tokenize[idx] = random_words

        return tokenize


def dot_similarity(x1, x2):
    return torch.matmul(x1, x2.t())


class bert_cont_mix(nn.Module):
    def __init__(self, num_label):
        super().__init__()

        self.bert = BertModel.from_pretrained("../bert-base-chinese")

        self.prejector = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
        )

        self.opi_cls = nn.Sequential(
            nn.Linear(768, num_label)
        )
        self.sen_cls = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 2),
        )

        self.tua = 0.7

    def Contrast_loss_func(self, ebd1, ebd2):
        ebd1 = self.prejector(ebd1)
        ebd2 = self.prejector(ebd2)

        b, h = ebd1.shape

        l_pos = torch.bmm(ebd1.view(b, 1, h), ebd2.view(b, h, 1)).squeeze(-1)
        l_neg = torch.mm(ebd1, ebd2.T)

        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(b, dtype=torch.long, device=device)

        loss = F.cross_entropy(logits / self.tua, labels, reduction="none")

        return loss

    def ContrastMix(self, ebd1, ebd2, Mix_ebd, lam, mix_ids):
        ebd1 = self.Aug_ebd(ebd1)
        ebd2 = self.Aug_ebd(ebd2)
        Mix_ebd = self.Aug_ebd(Mix_ebd)

        sample_loss = self.Contrast_loss_func(ebd1, ebd2).mean()

        Mix_loss1 = self.Contrast_loss_func(ebd1, Mix_ebd)
        Mix_loss2 = self.Contrast_loss_func(ebd2[mix_ids], Mix_ebd)

        Mix_loss = (lam * Mix_loss1 + (1 - lam) * Mix_loss2).mean()

        return 0.7 * sample_loss + 0.5 * Mix_loss

    def Aug_ebd(self, ebd):
        is_zero = (torch.sum(torch.abs(ebd), dim=2) > 1e-8).float()
        soft_len = torch.sum(is_zero, dim=1, keepdim=True)
        soft_len[soft_len < 1] = 1
        ebd = torch.sum(ebd, dim=1)
        ebd = ebd / soft_len

        return ebd

    def forward(self, X):
        Embed1 = self.bert.embeddings(input_ids=X["input_ids"])
        Embed2 = self.bert.embeddings(input_ids=X["aug_input_ids"])

        batch_size, seq_len = Embed1.size(0), Embed1.size(1)
        mix_ids = torch.randint(batch_size, (batch_size,))

        Mix_Embed = Embed2[mix_ids]

        lam = torch.distributions.beta.Beta(0.5, 0.5).sample((batch_size, 1, 1)).to(device)
        Mix_Embed = lam * Embed1 + (1 - lam) * Mix_Embed

        ebd1 = self.bert.encoder(
            hidden_states=Embed1,
            attention_mask=X["attention_mask"].unsqueeze(1).unsqueeze(1)
        )[0]
        ebd2 = self.bert.encoder(
            hidden_states=Embed2,
            attention_mask=X["attention_mask"].unsqueeze(1).unsqueeze(1)
        )[0]
        Mix_ebd = self.bert.encoder(
            hidden_states=Mix_Embed,
            attention_mask=X["attention_mask"].unsqueeze(1).unsqueeze(1)
        )[0]

        out_opi = self.opi_cls(ebd1)
        out_sen = self.sen_cls(ebd1[:, 0, :])

        loss_ccont = self.ContrastMix(ebd1, ebd2, Mix_ebd, lam.squeeze(-1), mix_ids)

        return out_opi, out_sen, loss_ccont

    def prediction(self, X):
        output = self.bert(input_ids=X["input_ids"], attention_mask=X["attention_mask"], output_attentions=True)
        ebd = output[0]
        logits_opi = self.opi_cls(ebd)
        logits_sen = self.sen_cls(ebd[:, 0, :])
        return logits_opi, logits_sen, output


if __name__ == "__main__":

    all_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            all_data.append(raw)
    random.shuffle(all_data)
    data, label, seq, sent_label = [], [], [], []
    for raw in all_data:
        data.append(raw["text"])
        label.append(raw["label"])
        seq.append(raw["seq"])
        sent_label.append(sent2id[raw["sentiment"]])

    length = 100
    train_data = data[:length]
    train_label = label[:length]
    train_seq = seq[:length]

    test_data = data[length:]
    test_label = label[length:]
    test_seq = seq[length:]

    train_dataset = OM_dataset(train_data, train_label, train_seq, sent_label)
    test_dataset = OM_dataset(test_data, test_label, test_seq, sent_label)

    text_dataset_call = BatchTextCall()
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=text_dataset_call)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=text_dataset_call)

    net = bert_cont_mix(num_label)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    loss_func = nn.CrossEntropyLoss()

    # main_attn(text_dataset_call.tokenizer, net, test_dataloader)

    best_f11 = 0.
    best_f1_asp = 0.
    best_f1_opi = 0.
    patience = 0
    for ep in range(10000):
        net.train()
        train_dataloader = tqdm(train_dataloader, total=len(train_dataloader), ncols=80, leave=False,
                                desc=colored('Training on train', 'yellow'))
        for batch in train_dataloader:
            data, label, seq, sent_label = batch
            output_opi, output_sen, cont_loss = net(data)

            output_opi = output_opi.reshape(-1, num_label)
            label = label.reshape(-1)
            loss = loss_func(output_opi, label) + F.cross_entropy(output_sen, sent_label) + cont_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            net.eval()
            accuracy = []
            accuracy_sen = []
            all_predictions = []
            all_labels = []
            all_sent_predictions = []
            all_sent_labels = []

            all_aspect_pred = []
            all_aspect_true = []
            all_opinion_pred = []
            all_opinion_true = []
            test_dataloader = tqdm(test_dataloader, total=len(test_dataloader), ncols=80, leave=False,
                                   desc=colored('Testing', 'yellow'))
            for test_batch in test_dataloader:
                test_data, test_label, test_seq, test_sent_label = test_batch
                logits_opi, logits_sen = net.prediction(test_data)
                predictions = torch.argmax(logits_opi, dim=-1).detach().cpu()
                predictions_sen = torch.argmax(logits_sen, dim=-1).detach().cpu()  # (b, )
                test_label = test_label.detach().cpu()

                attn_mask = test_data["attention_mask"]
                for i in range(len(test_label)):
                    seq_len = int(attn_mask[i].sum())
                    seq_preds = predictions[i][:seq_len]
                    seq_labels = test_label[i][:seq_len]
                    accuracy.extend((seq_preds == seq_labels).float().tolist())
                    all_predictions.extend(seq_preds.tolist())
                    all_labels.extend(seq_labels.tolist())

                    tem_aspect_pred = []
                    tem_aspect_true = []
                    tem_opinion_pred = []
                    tem_opinion_true = []
                    iter_seq_labels = iter(copy.deepcopy(seq_labels))
                    for i in range(seq_len):
                        true = next(iter_seq_labels)
                        if true == 1 or true == 2:
                            tem_aspect_true.append(seq_labels[i].item())
                            tem_aspect_pred.append(seq_preds[i].item())
                        if true == 3 or true == 4:
                            tem_opinion_true.append(seq_labels[i].item())
                            tem_opinion_pred.append(seq_preds[i].item())

                    all_aspect_pred.extend(tem_aspect_pred)
                    all_aspect_true.extend(tem_aspect_true)
                    all_opinion_pred.extend(tem_opinion_pred)
                    all_opinion_true.extend(tem_opinion_true)

                accuracy_sen.extend((predictions_sen == test_sent_label.detach().cpu()).float().tolist())
                all_sent_predictions.extend(predictions_sen.tolist())
                all_sent_labels.extend(test_sent_label.detach().cpu().tolist())

            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)
            all_sent_predictions = np.array(all_sent_predictions)
            all_sent_labels = np.array(all_sent_labels)

            accuracy = np.mean(np.array(accuracy))
            precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

            accuracy_sen = np.mean(np.array(accuracy_sen))
            precision_sen = precision_score(all_sent_labels, all_sent_predictions)
            recall_sen = recall_score(all_sent_labels, all_sent_predictions)
            f1_sen = f1_score(all_sent_labels, all_sent_predictions)

            accuracy_asp = np.mean(np.array(tem_aspect_pred) == np.array(tem_aspect_true))
            precision_asp = precision_score(tem_aspect_true, tem_aspect_pred, average='weighted', zero_division=0)
            recall_asp = recall_score(tem_aspect_true, tem_aspect_pred, average='weighted', zero_division=0)
            f1_asp = f1_score(tem_aspect_true, tem_aspect_pred, average='weighted', zero_division=0)

            accuracy_opi = np.mean(np.array(tem_opinion_pred) == np.array(tem_opinion_true))
            precision_opi = precision_score(tem_opinion_true, tem_opinion_pred, average='weighted', zero_division=0)
            recall_opi = recall_score(tem_opinion_true, tem_opinion_pred, average='weighted', zero_division=0)
            f1_opi = f1_score(tem_opinion_true, tem_opinion_pred, average='weighted', zero_division=0)
            # print(len(all_aspect_true), len(all_aspect_pred), len(all_opinion_true), len(all_opinion_pred))
            print(
                "{}, {} {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                    datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), colored(ep, "green"),
                    colored("All Acc:", "blue"), accuracy,
                    colored("All F1:", "blue"), f1,
                    colored("Sen Acc:", "blue"), accuracy_sen,
                    colored("Sen F1:", "blue"), f1_sen,
                    colored("Asp Acc:", "blue"), accuracy_asp,
                    colored("Asp F1:", "blue"), f1_asp,
                    colored("OPI Acc:", "blue"), accuracy_opi,
                    colored("OPI F1:", "blue"), f1_opi,
                ), flush=True)

            if best_f11 < f1 or best_f1_asp < f1_asp or best_f1_opi < f1_opi:
                if best_f1_asp < f1_asp or best_f1_opi < f1_opi:
                    best_f1 = f1
                    best_acc = accuracy
                    best_pre = precision
                    best_rec = recall
                    best_f1_sen = f1_sen
                    best_acc_sen = accuracy_sen
                    best_pre_sen = precision_sen
                    best_rec_sen = recall_sen
                    best_f1_asp = f1_asp
                    best_acc_asp = accuracy_asp
                    best_pre_asp = precision_asp
                    best_rec_asp = recall_asp
                    best_f1_opi = f1_opi
                    best_acc_opi = accuracy_opi
                    best_pre_opi = precision_opi
                    best_rec_opi = recall_opi
                    print("{}, {} {}".format(
                        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), colored(ep, "green"),
                        colored("saving the best model state", "red")), flush=True)
                    torch.save(net.state_dict(), f'best_model_{task_}.pt')
                    patience = 0
                if best_f11 < f1:
                    best_f11 = f1
                    print("{}, {} {}, Current patience: {}".format(
                        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), colored(ep, "green"),
                        colored("Best F1 in the All", "red"), patience), flush=True)
            else:
                patience += 1

            if patience > 10:
                result = {
                    "test_acc": best_acc,
                    "test_pre": best_pre,
                    "test_rec": best_rec,
                    "test_f1": best_f1,
                    "test_acc_asp": best_acc_asp,
                    "test_pre_asp": best_pre_asp,
                    "test_rec_asp": best_rec_asp,
                    "test_f1_asp": best_f1_asp,
                    "test_acc_opi": best_acc_opi,
                    "test_pre_opi": best_pre_opi,
                    "test_rec_opi": best_rec_opi,
                    "test_f1_opi": best_f1_opi,
                    "test_acc_sen": best_acc_sen,
                    "test_pre_sen": best_pre_sen,
                    "test_rec_sen": best_rec_sen,
                    "test_f1_sen": best_f1_sen,
                    "dataset": "nlp_option" if task_ == 1 else "text_mining"
                }
                with open("result.jsonl", "a", encoding='UTF-8') as f:
                    result = json.dumps(result, ensure_ascii=False)
                    f.writelines(result + '\n')
                break