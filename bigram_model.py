import torch
torch.manual_seed(1337 )

# read in text (currently set to works of shakespeare). shakespeare works is about 1.1mb 
with open('input.txt', 'r') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Tokenize the text
string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}
encoder = lambda x: [string_to_int[c] for c in x]  # x is a word
decoder = lambda x: ''.join([int_to_string[i] for i in x])

# Pull the text into a tensor
data = torch.tensor(encoder(text), dtype=torch.long)

# Split into training and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Block size aka the context length is 8 according to mr karpathy. this is like the maximum time dimension of the data that we will consider (1 up until block_size)
block_size = 8
# how many independent sequences will we process in parallel? this is like the batch dimension of the data
batch_size = 4 

def get_batch(split):
    ''' generate a small batch of data of inputs x and targets y'''
    # select the data, either training data or validation data
    data = train_data if split == 'train' else val_data


    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

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


# Import the neural network object that we will use to construct our model and that pytorch can read
#  And also the functional library that we will use to apply cross entropy loss to the model to train
# the weights, and also the softmax function, to generate new text
import torch.nn as nn
from torch.nn import functional as F
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # an embedding when it is created is just a matrix of x rows and y columns. 
        # x is the number of tokens in the vocabulary and y is the dimension of the embedding.
        # in the bigram language model, the embedding is the same as the output. the end result is just mapping the transition matrix taken
        # from the training data to a vector space where the transition matrix is parametrised by the softmax function.
        # this is a simple model, and really this is not a real embedding. it is just a lookup table. by later embedding models we will see that the embedding is a parameter of the model and is learned.
        # the softmax function is a function that takes a vector and returns a probability distribution over the vocabulary.
        # 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # print(self.token_embedding_table.weight)

    def forward(self, idx, targets=None):
        # idx is a tensor of shape (batch_size, block_size)
        # self.token_embedding_table is a tensor of shape (vocab_size, vocab_size)
        # the output is a tensor of shape (batch_size, block_size, vocab_size)
        # note that we are not applying a function to idx, we are composing the embedding table with idx to get a new tensor. each point in batch x time is give a logit vector over the vocabulary.
        # this is how in pytorch we apply a function to a tensor.
        # a question that karpathy doesn't go into is, if we are just doing next token prediction with no attention, why even have the time dimension greater than 1?
        # the answer is yes, you can just roll the time dimension into the batch dimension, and do parallel processing along the new batch dimension. but this is not how we are presenting here.
        # we will later add attention to the model, and attention will allow us to look at the context of the sequence.

        logits = self.token_embedding_table(idx) # (batch_size, block_size, vocab_size) = (B, T, C) = (4, 8, 65) = (batch_size, context_window, vocab_size)
        
        # do a forward pass without targets to generate from the learned weights
        if targets is None:
            return logits, None
        
        # B, T, C are integers
        B, T, C = logits.shape
        # we are rolling the time dimension into the batch dimension to get a single tensor of shape (batch_size * context_window, vocab_size)
        # karpthay says this is because the cross entropy loss function expects this shape.
        # but it is interesting to note that this also aligns with what i said above about the time dimension being greater than 1.
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        # untrained, the cross entropy should be -ln(1/C) = ln(C). this is because the lowest entropy state, hence highest probability
        # is the uniform distribution. and since we randomly set the probabilities, and uniform is the lowest entropy, it should be close to this, 
        # i.e. -ln(1/65) = ln(65) ~ 4.17
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices (over the vocab_size) in the current context, T goes from 0 to max_new_tokens
        for _ in range(max_new_tokens):
            # get the predictions according to however the embbedding table looks right now, without training it's random
            logits, _ = self(idx)
            # focus only on the last time step, because we're ignoring time/context window
            # in this simple model, so getting the final token in the time dimension
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities. softmax is a multi-dimensional sigmoid, which projects K-dim euclidean space to the
            # K-1 dim simplex, without the constraint of sum=1 lowering the dimension, therefore summing over C after
            # applying softmax gives (B, 1) a tensor of ones of dimension B
            probs = F.softmax(logits, dim=-1) # (B, C)
            # randomly sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m = model.to(device)
print(m)

logits, loss = m(xb, yb)  # currently garbage because we haven't trained the model yet  
print(loss) # value is 5.034, which means there is extra entropy, that's just random noise though

# generate some text. this is pretty stupid because we set up the forward pass to have a context window, but then 
# we throw it away since only the last token has a new value that is generated after. anyways, here we are printing 
# the logits over the vocab size for the first batch
# for batch size one and context one, generate 100 tokens after the character indexed by a 0, which is the new line character
# then decode the numbers into the text
# the [0] is because we have a batch dimension of 1, so just take the first and only batch, and this becomes a tensor
# of T = max_new_tokens
# this will just be random text
print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# now we can do some training, by creating an optimiser object
learning_rate = 1e-3
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# but first, define the estimate loss function
# this decorator tells pytorch not to retain the grad graph
@torch.no_grad()
def estimate_loss():
    eval_iters = 200
    out = {}
    # model.train() sets model.training = True
    # model.eval() sets model.training = False
    # This flag is checked by certain layers during forward()
    # Dropout behaves differently depending on what is set:
    #   train: randomly zeroes activations
    #   eval: passes activations unchanged
    #    train: y=x⋅Bernoulli(p)
    #    eval: y=x
    # BatchNorm behaves differently:
    #   train: uses current batch mean/variance and updates running stats
    #   eval: uses stored running mean/variance only
    # Custom layers can branch on self.training
    # Bigram model has:
    #   no dropout
    #   no batchnorm
    #   no self.training conditionals
    # Therefore train() and eval() produce identical outputs
    # only the flag changes, not computation
    # They are used for compatibility with future architectures

    # Set evaluation mode -> think deterministic vs stochastic inference behavior. you want some noise when training
    model.eval()

    # evaluate mean loss over 200 iterations for training and valuation data
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # set back to training mode
    model.train()
    return out

# now we do some inference with the model trained
batch_size = 32
for iter in range(10000):

    # every now and again, test the model against training and val data
    if iter % 300 == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    # get some sample data from shakespears
    xb, yb = get_batch('train')

    # get the cross entropy loss and logits, by doing a forward pass!!! every operation in the system is recorded 
    logits, loss = m(xb, yb)

    # clear the gradients
    optimiser.zero_grad(set_to_none=True)

    # get the derivatives values through a backwards pass !!!! this populates the gradients all of the parameters
    loss.backward()

    # update the weights depending on the gradients
    optimiser.step()

    # this will never get above 2.3 cross entropy, because that's the hard limit of a one context window with this data
    # this is basically the true conditional entropy for this model, with probabilities just being the empirical 
    # conditional distribution i.e. p(xt+1 | xt) = count(xt to xt+1) / count(xt)
    # print(loss.item())

# print the inference
print(decoder(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
