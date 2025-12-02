import argparse

import torch
from torch import optim

from transformer import Transformer

torch.manual_seed(1337)
torch.set_float32_matmul_precision('high')


parser = argparse.ArgumentParser(
    prog='single_ffwd',
    description='Train either a vanilla or single ffwd transformer on tinyshakespeare dataset.',
)
parser.add_argument('mode', choices=['vanilla', 'single_ffwd'])
args = parser.parse_args()

SINGLE_FFWD = args.mode == 'single_ffwd'

# CONTEXT_LENGTH = 256
# D_MODEL = 384
# N_HEADS = 4
# N_LAYERS = 6
# DROPOUT = 0.2


CONTEXT_LENGTH = 256
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 6
DROPOUT = 0.2

BATCH_SIZE = 1024
LR = 3e-4

with open('shakespeare.txt', 'r') as file:
    train_tokens = file.read()
    vocab = sorted(list(set(train_tokens)))

VOCAB_SIZE = len(vocab)

print(''.join(vocab))

stoi = { token: i for i, token in enumerate(vocab) }
itos = { i: token for i, token in enumerate(vocab) }

tokens = torch.tensor([stoi[token] for token in train_tokens])  # 200k words
train_tokens = tokens[:int(len(tokens) * 0.9)]
test_tokens = tokens[int(len(tokens) * 0.9):]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, N_LAYERS, N_HEADS, DROPOUT, single_ffwd=SINGLE_FFWD)

attention_params = 0
ffwd_params = 0
layernorm_params = 0
tok_embedding_params = 0
pos_embedding_params = 0
other_params = 0

for name, param in model.named_parameters():
    if '.attention.' in name:
        attention_params += param.numel()
    elif '.ffwd.' in name or 'single_ffwd' in name:
        ffwd_params += param.numel()
    elif 'layernorm' in name:
        layernorm_params += param.numel()
    elif 'tok_embedding' in name:
        tok_embedding_params += param.numel()
    elif 'pos_embedding' in name:
        pos_embedding_params += param.numel()
    else:
        print(name)
        other_params += param.numel()

total_ffwd_att = attention_params + ffwd_params
total = attention_params + ffwd_params + layernorm_params + tok_embedding_params + pos_embedding_params + other_params

print(f'{ffwd_params=}')
print(f'{attention_params=}')
print(f'{layernorm_params=}')
print(f'{tok_embedding_params=}')
print(f'{pos_embedding_params=}')
print(f'{other_params=}')
print(f'{total=}')
print(f'{total_ffwd_att=}')

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

@torch.no_grad
def get_test_loss():
    model.eval()

    n_loss = 3
    loss_values = torch.zeros((n_loss,))

    for i in range(n_loss):
        indices = torch.randint(0, len(test_tokens) - CONTEXT_LENGTH, (BATCH_SIZE, 1)).repeat(1, CONTEXT_LENGTH) + torch.arange(CONTEXT_LENGTH)

        test_x = test_tokens[indices].to(device)
        test_y = test_tokens[indices+1].to(device)

        _, loss = model(test_x, test_y)

        loss_values[i] = loss

    model.train()
    return loss_values.mean()

train_loss_values = []
test_loss_values = []

if __name__ == '__main__':
    for i in range(10_000):

        indices = torch.randint(0, len(train_tokens) - CONTEXT_LENGTH, (BATCH_SIZE, 1)).repeat(1, CONTEXT_LENGTH) + torch.arange(CONTEXT_LENGTH)

        train_x = train_tokens[indices].to(device)
        train_y = train_tokens[indices+1].to(device)

        optimizer.zero_grad()

        logits, loss = model(train_x, train_y)
        # print(logits)

        loss.backward()
        optimizer.step()

        train_loss_values.append(loss.item())

        # print(loss.item())

        if i % 10 == 0:
            test_loss = get_test_loss()
            test_loss_values.append(test_loss)

            print(i)
            print(f'test loss: {test_loss}')
            print(f'train loss: {loss.item()}')

