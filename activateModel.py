import os
import torch
import re
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torch import nn, optim
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from transformers import BasicTokenizer
import logging
logging.basicConfig(level=logging.INFO)

device = torch.device('cuda')

vocab_dict = np.load('dictionary/vocab_dict1.npy',allow_pickle=True)
vocab_dict = vocab_dict.item()
max_length = 300
def tokenize(sentence):
    sentence = BasicTokenizer().tokenize(text=sentence)
    return sentence

class Vocab():
    def __init__(self):
        self.unk = 0
        self.pad = 1
        self.unk_word = 'unk'
        self.pad_word = 'pad'
        self.dict = vocab_dict
        self.count = {}


    def pad_sentence(self,sentence,max=max_length):
        if len(sentence) >= max:
            sentence = sentence[:max]
        else:
            sentence = sentence + [self.pad_word] * (max-len(sentence))
        return sentence

    def word_to_num(self,sentence):
        for i in range(len(sentence)):
            if sentence[i] in self.dict:
                sentence[i] = self.dict[sentence[i]]
            else:
                sentence[i] = self.dict['unk']
        return np.array(sentence)

vocab = Vocab()

class ImdbDataset(Dataset):
    def __init__(self):
        super(ImdbDataset,self).__init__()
        self.train_path = r'D:\collected data\IMDB_review\test'
        train_path = self.train_path
        temp_path = [os.path.join(train_path,'pos'),os.path.join(train_path,'neg')]
        self.total_path = []
        self.total_path_temp = []
        for path in temp_path:
            pos_text = [os.path.join(path,i) for i in os.listdir(path)]
            self.total_path.extend(pos_text)
            self.total_path_temp.extend(pos_text)

        for i in range(len(self.total_path)):
            with open(self.total_path[i],'r',encoding='utf-8') as file:
                line = file.readline()
                line = tokenize(line)
                line = vocab.pad_sentence(line)
                self.total_path[i] = line
            if (i + 1) % 1000 == 0:
                logging.info('already processed file %d', i + 1)
        logging.info('file processing complete')



    def __getitem__(self,index):
        text = self.total_path[index][:]  ##???????????????text
        file_path = self.total_path_temp[index]
        label_str = file_path.split('\\')[-2]
        label = 1 if label_str=='pos' else 0    ##?????????????????????
        label = torch.tensor(label, dtype=torch.int32, device=device)
        content = vocab.word_to_num(text)
        content = torch.from_numpy(content)
        content = content.long().to(device)
        return {'label':label,'text':content}


    def __len__(self):
        return len(self.total_path)

##dataLoader???????????????label???text?????????????????????????????????enumerate()??????dataLoader?????????????????????????????????
data_loader = DataLoader(ImdbDataset(),batch_size=40,shuffle=True)

class LSTM(nn.Module):
    def __init__(self,vocab_dim,embedding_dim,hidden_dim):
        super(LSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_dim,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            bidirectional=True,
                            dropout=0.5,
                            batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        embedding = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedding)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = self.dropout(hidden)
        out = self.linear(hidden)

        return out


lstm = LSTM(26181,300,256)



vocab_vector = np.load('dictionary/vocab_vector.npy',allow_pickle=True)
vector_tensor = torch.from_numpy(vocab_vector)
lstm.embedding.from_pretrained(vector_tensor)
lossFun = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(lstm.parameters(),lr=1e-3)
lstm.to(device)


lstm.load_state_dict(torch.load("model/BiLSTM_2.pth"))


epochs = 3
acc = 0
for epoch in range(epochs):
    avg_acc = []
    lstm.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch + 1} accuracy {acc}"):
            pred = lstm(batch['text']).squeeze(1)
            preds = torch.round(torch.sigmoid(pred))
            correct = torch.eq(preds, batch['label']).float()
            acc = correct.sum() / len(correct)
            acc = acc.item()
            avg_acc.append(acc)
    acc = np.array(avg_acc).mean()



def predict_sentiment(sentence):  # ?????????????????????I love This film bad
    sentence = BasicTokenizer().tokenize(sentence)
    sentence = vocab.word_to_num(sentence)
    sentence_tensor = torch.from_numpy(sentence)
    tensor = sentence_tensor.long().to(device)
    pred = torch.sigmoid(lstm(tensor.unsqueeze(0)).squeeze(1))
    # return pred
    print(pred)
    if pred.item() <= 0.5:
        print('??????')
    elif pred.item() > 0.5:
        print('??????')



for i in range(999):
    input_sentence = input()
    predict_sentiment(input_sentence)