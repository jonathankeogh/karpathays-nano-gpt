import torch
torch.manual_seed(1337)

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
encoder = lambda x: [string_to_int[c] for c in x]
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
import torch.nn as nn

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # an embedding when it is created is just a matrix of x rows and y columns. 
        # x is the number of tokens in the vocabulary and y is the dimension of the embedding.
        # in the bigram language model, the embedding is the same as the output. the end result is just mapping the transition matrix taken
        # from the training data to a vector space where the transition matrix is parametrised by the softmax function.
        # the softmax function is a function that takes a vector and returns a probability distribution over the vocabulary.
        # 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        print(self.token_embedding_table.weight)
        exit()

    def forward(self, idx):
        return self.token_embedding_table(idx)

m = BigramLanguageModel(vocab_size)
print(m)
exit()