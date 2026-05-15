import os
import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
# length of the context / 上下文大小
block_size = 8
# length of training batch / 并行训练的样本数
batch_size = 32
# learning rate / 学习率 -> 参数更新的步长
learning_rate = 1e-3
# evaluation intevals / 每多少 step 进行一次 loss 验证，监控训练过程
eval_interval = 500
# all interations / 总共迭代多少步 iteration
max_iters = 5000
# evaluation iterations / validation 的时候采样多少 batch
eval_iters = 200
# 训练设备
device = 'mps' if torch.mps.is_available() else 'cpu'

# number of the embed dimension
# in version 2, we used 32 feature channels to tokenized the vocabulary instead of bi-encoding in version 1
n_embd = 32
# 
# head_size = 16



# load plain text of all Harry Potter books
with open('./source/Complete_Harry_Potter_txt_file/Harry_Potter_complete_dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# construct a vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# mapping the characters to integers (this is a typical progress called tokenizer)
# define a simple encoder and a decoder
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)


# turn vocabulary list to tensor for parallel computing
data = torch.tensor(encode(text), dtype=torch.long)
# split the data into train sets and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


torch.manual_seed(507)

def get_batch(split):
    # randomly select a batch from the data
    data = train_data if split == 'train' else val_data
    # {batch_size} random numbers to locate the begin of the block
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # stack: from vectors to a metrix
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# tell torch the code inside doesm't called backward()
# so it save the memory by not recording the runnning params
@torch.no_grad
def estimate_loss():
    out = {}
    # turn the model into evaluaiton phase
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # turn the model back to training phase
    model.train()
    return out


# one-head self attention
class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        # creat linears to map from token_embed to self-attention vectors
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 一个用于 mask 的下三角矩阵
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape   # (B,T,C)
        k = self.key(x)     # (B,T,head_size)
        q = self.query(x)     # (B,T,head_size)
        # compute the weight
        wei = q @ k.transpose(-2, -1) * C**-0.5     # (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # perform weighted aggregation
        v = self.value(x)   # (B,T,head_size)
        out = wei @ v       # (B,T,head_size)
        return out


# create a multi-head self-attention
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)



# 二元语言模型：
# Version 1
# 1.预测的时候仅根据上一个 token 的内容进行预测；
# 2.特征向量维数等于词汇表长度，词嵌入向量直接当做 logits 来用；
# Version 2
# 1.加了一个单头的自注意力模块；
# Version 3
# 1.改成了多头注意力
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # version 1: 词嵌入，特征维数等于词汇表长度
        # 其实相当于我们做了一个二维编码，最终的 logits 直接按照查找表的形式（哪个是 1 结果就是哪个字符）给出
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        # version 2: 实现一个 self-language model head
        # # 重新做词嵌入，用 32 维的特征向量
        # self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # # 除了编码 token 的 identification，还要编码字符出现的位置信息
        # self.positon_embedding_table = nn.Embedding(vocab_size, n_embd)
        # # a single-head self-attention
        # self.sa_head = Head(n_embd)
        # # 一个线性层将特征映射成 logits
        # self.lm_head = nn.Linear(n_embd, vocab_size)

        # version 3: multi-head self-language model head
        # 重新做词嵌入，用 32 维的特征向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 除了编码 token 的 identification，还要编码字符出现的位置信息
        self.positon_embedding_table = nn.Embedding(vocab_size, n_embd)
        # 4 head of 8-dimentional self-attention
        self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # 一个线性层将特征映射成 logits
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        # version 1: directly look up the vocabulary to get logits based on the last token
        # logits = self.token_embedding_table(idx)
        # version 2: 
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # (batch_size, context_len, n_embd)
        position_emb = self.positon_embedding_table(torch.arange(T, device=device)) # (context_len, n_embd)
        # 将 token 的 identification 和 position 信息都编码进来
        x = token_emb + position_emb
        # 引入一个注意力头让 context 中的 token 能够“交流“起来
        # x = self.sa_head(x)
        # apply multi-head self attention
        x = self.sa_heads(x)
        logits = self.lm_head(x) # (batch_size, context_len, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # reshape to the expected input of cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # 用交叉熵（ -ln(dist) ）作为最经典的 loss 函数
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # predict the next token and cat at the end of the input
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # call itself to predict
            logits, loss = self(idx_cond)
            # only last token is the prediction we need
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # pick up the specific token accroding to the probability
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# iterations
for iter in range(max_iters):
# if trained based all train data, it's called "epoch". 1 epoch = many many iteration

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    # feed forward.
    # i prefer this edition: logits, loss = m.forward(xb, yb)
    logits, loss = model.forward(xb, yb)
    # erase the previous grediant
    optimizer.zero_grad(set_to_none=True)
    # backward to calculate the gradients of parameters
    loss.backward()
    # optimize the parameters
    optimizer.step()

print(loss.item())

context = torch.zeros((1,1), dtype=torch.long, device=device)
# 从一个换行符开始预测
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))