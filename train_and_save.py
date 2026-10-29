import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337 )
batch_size = 64 # how many independent sequences will we process in parallel?
max_iters = 5000
loss_step = 300
block_size = 256 # maximum context length for predictions
training_split = 0.9
eval_interval = 500
eval_iters = 200 # for estimating loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4 # used because self-attention can't tolerate very large learning rates so model is much larger
n_embed = 384
dropout=0.2 # probability of dropping out a neuron
n_layers = 6
n_head = 6
torch.set_float32_matmul_precision('high') # allows TF32 matmuls, a one-line speed-up on this card.




with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Tokenize the text
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
encoder = lambda x: [string_to_int[c] for c in x]  # x is a word
decoder = lambda x: ''.join([int_to_string[i] for i in x])

data = torch.tensor(encoder(text), dtype=torch.long)

n = int(training_split * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    ''' generate a small batch of data of inputs x and targets y'''
    data = train_data if split == 'train' else val_data


    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # move the data to the device, hopefully a GPU
    return x.to(device), y.to(device)

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# print('----')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        # key: what do i contain / broadcast?
        # query: what information do i want to extract from the other nodes?
        # value: what will i send to the other nodes if they ask for it

        # think of the key as a broadcast, the query as a request, and the value as a response.
        # another way is to think of a directed graph of nodes, where the key is the node's content,
        # the query is the node's request, and the value is the node's response.
        # the line comes out of the node, and the direction is determined by the query direction,
        # and the edge strength is determined by the attention scores, which are the dot products of queries and keys!
        
        # in this example, we are dot producting the query and the key to get the attention scores, which is a measure of
        # how relevant the other nodes are to the current node.
        # we then use the attention scores to weight the values. this is how we get the final output from the head, as the
        # sum of the values, weighted by how much each
        # value is relevant to the current node.
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        # we are dividing by the sqrt of the dimension / channel / head size to prevent the attention scores from exploding, otherwise
        # they would converge to a one-hot vector
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # project the output back to the embedding dimension, so we can add to the residual stream
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # project up to 4 * n_embed 
            nn.Linear(n_embed, 4 * n_embed),
            # then apply a non-linearity
            nn.ReLU(),
            # then project back to the embedding dimension, so we can add to the residual stream
            nn.Linear(4 * n_embed, n_embed),
            # then apply a dropout
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head # 32//4 = 8, self-attention heads of 8 dimension each
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # previously layernorm was applied after, but nowadays it's applied before, to prevent the gradient from exploding
        # this is because the gradient of the layernorm is 1, so if we applied it after, the gradient would be 1 * the gradient
        # of the self-attention or feedforward,
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# token encoding: "I like cats" → [15496,  180, 12345]
# token embeddeding: 15496 -> [0.43, ... , 9,2] of size n_embed
class TheTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) 
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # note that we are defining something with weights of dim n_embed * vocab_size, it's not that we are acting on these numbers and producing a scalar

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # in this case, the lm_head has dimension = n_embed < vocab_size. what does that mean? it means we are compressing the space that the tokens live in, which
        # will help the model generalise onto a knowledge manifold in latent space, if n_embed > vocab_size then there would be massive redundancy and no generalisation (and hence efficiency)
        # in all LLMs, n_embed < vocab size. for gpt5 it's hundreds of thousands, while the embedding space is tens of thousands
        tok_embed = self.token_embedding_table(idx) # (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_embed + pos_embed # (B,T,C)   this encodes the identity of the token, plus it's position in the context window (but context is not learned yet)
        #  So what is positional embedding actually useful for?
        # It gives the model a way to distinguish:
        # “dog bites man”
        # “man bites dog”
        # These become different
        # Without position:
        # the model would see a "bag of tokens", not a sequence. it would mean the meaning is permutation invariant in the time dimension
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # (B, T, vocab_size). note that this isn't a matrix, it's a tensor plus a bias, so it's technically affine: f(x) = Wx + b
        
        if targets is None:
            return logits, None
        
        B, T, C = logits.shape

        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) where each row is a sequence of tokens. we want to generate new tokens for each row.
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # idx_cond is (B, block_size)
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution to get the next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # add the new token to the sequence and move to the next time step
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = TheTransformerModel()
# initialise the model on the device, hopefully a GPU
m = model.to(device)
print(m)

logits, loss = m(xb, yb)
print(loss) 

print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))

optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

for iter in range(max_iters):

    if iter % loss_step == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)

    optimiser.zero_grad(set_to_none=True)

    loss.backward()

    optimiser.step()

# save the model
torch.save(model.state_dict(), 'model.pt')

# back to evaluation / inference mode, not training mode
model.eval()
# tell pytorch not to track gradients because we're not doing any backprop
with torch.no_grad():
    print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))
