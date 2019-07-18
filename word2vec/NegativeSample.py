
from __future__ import print_function, unicode_literals, absolute_import, division

import torch, torch as t, torch.nn as nn, torch.optim as optim, numpy as np
from torch.utils.data import Dataset, DataLoader


class Data_set(Dataset): # 演示如何使用 Dataset
    def __init__(self):
        X = ['x_' + str(i) for i in range(12)]
        y = [str(1)] * 6 + [str(0)] * 6
        self.data = list(zip(X, y))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y

# dataset = Data_set()
# dataloader = DataLoader(dataset, batch_size=3, shuffle=True)        
# for _, (x, y) in enumerate(dataloader): print(x,y)
# #######输出:
# # ('id10', 'id0', 'id6') ('0', '1', '0')
# # ('id8', 'id1', 'id11') ('0', '1', '0')
# # ('id3', 'id5', 'id7') ('1', '1', '0')
# # ('id4', 'id9', 'id2') ('1', '0', '1')


class PermutedSubsampleedCorpus(Dataset): # 看上面的例子理解
    '''语料采样,防止频率大的单词过分多,得到采样分布 p(x) https://github.com/theeluwin/pytorch-sgns/blob/master/train.py
    '''
    def __init__(self, data, word_sample=None):
        self.data = []

        for iword, owords in data:
            if np.random.rand() > word_sample[iword]:
                self.data.append((iword, owords))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, list(owords)

########################### 模型部分
class SkipGramNegativeSample(nn.Module):
    '''SkipGram词向量模型的简单实现, 仅供学习参考'''

    def __init__(self, vocab_size=20000, emb_dim=300, padding_idx=0, 
                 n_negs=5, weights=None):
        super(SkipGramNegativeSample, self).__init__()                   
        self.vocab_size = vocab_size # 词典
        self.emb_dim = emb_dim # 嵌入维数
        self.n_negs = n_negs   # 负采样个数, 或者理解成正样本的倍数, 
        self.weights = weights # 可选的负采样策略
        self.ivectors = self.create_emb(vocab_size, emb_dim, padding_idx)
        self.ovectors = self.create_emb(vocab_size, emb_dim, padding_idx)
        self.loss_fn  = None # 负采样的模型比较特殊, 前向传播直接最大化负的对数似然函数, 所以不需要额外的损失函数
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, iword, owords):
        '''前向传播'''
      
        if type(iword) is not torch.Tensor: # 检查输入的数据类型, 如果不是整型的tensor, 强制转换
            iword  = torch.LongTensor(iword)
            owords = torch.LongTensor(owords)

        batch_size = iword.size(0)      # iwords: (batch_size)
        context_size = owords.size(-1)  # owords: (batch_size, c * 2) c可以查看论文 https://arxiv.org/pdf/1310.4546.pdf

        if self.weights is not None: # 提供采样策略就采样
            nwords = torch.multinomial(self.weights, batch_size * context_size * self.n_negs, replacement=True).view(batch_size, -1)
                
        else: # 默认随机采样
            nwords = torch.randint(0, self.vocab_size, (batch_size, context_size * self.n_negs)) # 负采样出来的样本

        ivectors = self.ivectors(iword).unsqueeze(2)  # (batch_size, embeding_dim, 1)
        ovectors = self.ovectors(owords)              # (batch_size, context_size, embedding_dim)
        nvectors = self.ovectors(nwords).neg()        # (batch_size, context_size * n_negs, embedding_dim)
            # 从cos相似度的角度看, 加负号是让目标词远离负样本

        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1) # (batch_size)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1) # (batch_size)
        
        return -(oloss + nloss).mean() # 直接输出目标函数的损失了

    def create_emb(self, vocab_size, emb_dim=300, padding_idx=0):

        E = nn.Embedding(vocab_size, emb_dim, padding_idx)
        # E.weight.data.uniform_(- 0.5 / emb_dim, 0.5 / emb_dim) 另一种初始化方法, 只是会导致 0 初始化非0向量,并且不会更新
        E.weight = nn.Parameter(
                            torch.cat(
                                [torch.zeros(1, emb_dim),
                                 torch.FloatTensor(vocab_size - 1, emb_dim).uniform_(-0.5 / emb_dim, 0.5 / emb_dim)]
                            ), 
                            requires_grad=True # 默认是True
                   )
        return E

    def train_batch(self, iword, owords):

        self.zero_grad()
        out = self(iword, owords)
        loss = out # 这个比较特殊, 模型输出就是loss        
        loss.backward()
        self.optimizer.step()
        return loss.item()

###################数据处理部分 参考了 https://github.com/gutouyu/ML_CIA/tree/master/Embedding/code
T = 1e-5
CONTEXT_SIZE=2
with open('text9', 'rb+') as f: 
    raw_data = f.readlines()
    raw_data = raw_data[0].split()

vocab = set(raw_data)
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

word_count = dict()
for word in raw_data:
    try: word_count[word] += 1
    except KeyError: word_count[word] = 1

word_frequency = np.array(list(word_count.values()))
word_frequency = word_frequency / word_frequency.sum()
word_sample = 1 - np.sqrt(T / word_frequency)
word_sample = np.clip(word_sample, 0, 1)
word_sample = {wc[0]: s for wc, s in zip(word_count.items(), word_sample)}      

data = []
for target_pos in range(CONTEXT_SIZE, len(raw_data) - CONTEXT_SIZE):
    context = []
    for w in range(-CONTEXT_SIZE, CONTEXT_SIZE + 1):
        if w == 0:
            continue
        context.append(raw_data[target_pos + w])
    data.append((raw_data[target_pos], context))

dataset = PermutedSubsampleedCorpus(data, word_sample)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False) # 批数据生成器
############################# 数据生成结束 一个样本 是这样的  iword, owords :  'w3', ['w1', 'w2', 'w4', 'w5']


# 训练部分
losses = []
model = SkipGramNegativeSample(vocab_size, 8, n_negs=5)
for epoch in range(10):
    total_loss = 0
    for _, (iword, owords) in enumerate(dataloader):
        iword = list(map(lambda x: word_to_ix[x], iword))
        iword = torch.LongTensor(iword)

        owords = list(map(list, owords))
        owords = np.array(owords).T

        myfunc = np.vectorize(lambda x: word_to_ix[x])
        owords = list(map(myfunc, owords))
        owords = torch.LongTensor(owords)

        total_loss += model.train_batch(iword, owords) 

    losses.append(total_loss)
print(losses)
