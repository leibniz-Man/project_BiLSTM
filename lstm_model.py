import os
import torch
import re
import random
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torch import nn, optim
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from transformers import BasicTokenizer
import logging

# set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

logging.basicConfig(level=logging.INFO)
device = torch.device('cuda')
glove_dict = np.load('dictionary/text_dict.npy',allow_pickle=True)
glove_dict = glove_dict.tolist()
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
        self.dict = {
            self.unk_word:self.unk,
            self.pad_word:self.pad
        }
        self.count = {}

    def count_word(self,sentence):
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1

    def build_vocab(self,min_length=5):
        self.count = {word:value for word,value in self.count.items() if value > min_length}
        for word,value in self.count.items():
            if word in glove_dict:
                self.dict[word] = len(self.dict)
        # self.dict = dict(zip(self.dict.values(),self.dict.keys()))
        return self.dict

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
        self.train_path = r'D:\collected data\IMDB_review\train'
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
                vocab.count_word(line)
                line = vocab.pad_sentence(line)
                self.total_path[i] = line
            if (i+1) % 1000 == 0:
                logging.info('already processed file %d',i+1)
        logging.info('file processing complete')

        self.vocab_dict = vocab.build_vocab()
        # np.save('vocab_dict1', vocab.dict)
        logging.info('vocab dim : %s',len(self.vocab_dict))


    def __getitem__(self,index):
        text = self.total_path[index][:]  ##每个文件的text
        file_path = self.total_path_temp[index]
        label_str = file_path.split('\\')[-2]
        label = 1 if label_str=='pos' else 0    ##每个文件的标签
        label = torch.tensor(label, dtype=torch.int32, device=device)
        content = vocab.word_to_num(text)
        content = torch.from_numpy(content)
        content = content.long().to(device)
        return {'label':label,'text':content}


    def __len__(self):
        return len(self.total_path)

##dataLoader可以自动将label和text组成元组，然后只要通过enumerate()枚举dataLoader的每个数据并处理就行了
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
            nn.Linear(hidden_dim*2,20),
            nn.ReLU(),
            nn.Linear(20,1)
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
# lstm.load_state_dict(torch.load('model/BiLSTM_2.pth'))

epochs = 10
acc = 0
los = 0
lossList = []
accList = []
for epoch in range(epochs):
    avg_acc = []
    loss_acc = []
    lstm.train()
    for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1} accuracy {acc} loss {los}"):
        pred = lstm(batch['text']).squeeze(1)
        loss = lossFun(pred, batch['label'].float())
        loss_acc.append(loss.item())
        preds = torch.round(torch.sigmoid(pred))
        correct = torch.eq(preds, batch['label']).float()
        acc = correct.sum() / len(correct)
        acc = acc.item()
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    los = np.array(loss_acc).mean()
    acc = np.array(avg_acc).mean()
    lossList.append(los)
    accList.append(acc)


y = torch.arange(1,epochs+1,dtype=torch.float32)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,14))
ax1.set_title('accuracy')
ax2.set_title('loss')
ax1.plot(y,accList,color='turquoise')
ax2.plot(y,lossList)

plt.show()




torch.save(lstm.state_dict(), 'model/BiLSTM_2.pth')


