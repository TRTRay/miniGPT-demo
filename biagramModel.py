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
learning_rate = 1e-2
# evaluation intevals / 每多少 step 进行一次 loss 验证，监控训练过程
eval_interval = 300
# all interations / 总共迭代多少步 iteration
max_iters = 3000
# evaluation iterations / validation 的时候采样多少 batch
eval_iters = 200
device = 'mps' if torch.mps.is_available() else 'cpu'



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


# 二元语言模型：
# 1.预测的时候仅根据上一个 token 的内容进行预测；
# 2.特征向量维数等于词汇表长度，词嵌入向量直接当做 logits 来用；
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # 词嵌入，特征维数等于词汇表长度
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
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
            # call itself to predict
            logits, loss = self(idx)
            # only last token is the prediction we need
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # pick up the specific token accroding to the probability
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BigramLanguageModel(vocab_size)
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