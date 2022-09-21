# the dotted lines denote the beginning and end of a code block

# ------------------------------------------------------------------

# making a character-level 'language model' (multilayer perceptron), trained on a list of names.
# its layers are: (one_hot encoded input vectors -->) embedding --> hidden --> output (logits) --> softmax (probabilities)

import torch, random, time, json
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# import dataset (list of names), mapping from letter to index and vice versa
# '.' will be the start and end character

words = open('names.txt', 'r').read().split('\n')
letters = list(sorted(set(''.join(words))))
stoi = {k:n+1 for n,k in enumerate(letters)}
stoi['.'] = 0
itos = {n:k for k,n in stoi.items()}

# ------------------------------------------------------------------

# ------------------------------------------------------------------

# functions for setting model weights, building the training/dev/test datasets, storing a json file


def get_parameters(context_size, embedding_dimension, hidden_dimension):
  g = torch.Generator().manual_seed(2147483647)
  input_dimension = 27
  output_dimension = 27
  
  # weights connecting the one_hot encoded inputs to the embedding layer
  # though the one-hots are used to index into C, not multiplied by C.
  C = torch.randn((input_dimension, embedding_dimension), generator=g)

  # the weights connecting the embedding layer to the hidden layer
  W1 = torch.randn((context_size*embedding_dimension, hidden_dimension), generator=g)
  b1 = torch.randn(hidden_dimension, generator=g)

  # the weights connecting the hidden layer to the output layer (logits)
  W2 = torch.randn((hidden_dimension, output_dimension), generator=g)
  b2 = torch.randn(output_dimension, generator=g)

  # i forgot why i do this.
  parameters = [C, W1, b1, W2, b2]
  for p in parameters:
    p.requires_grad = True
  
  return C, W1, b1, W2, b2

def build_dataset(words, block_size):
  X, Y = [], []
  for w in words:
    context = [0] * block_size
    for ch in w+'.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)

      context = context[1:] + [ix]
    
  X = torch.tensor(X)
  Y = torch.tensor(Y)

  return X, Y

def toJson(obj, filename, overwrite = 0):
    if overwrite:
        mode = "w"
    else:
        mode = "x"
    with open(filename, mode) as f:
        json.dump(obj, f, indent=1)
# ------------------------------------------------------------------

# ------------------------------------------------------------------

# setting hyperparameters, building datasets, training the model

# model parameters
context_size = 5
embedding_dimension = 5
hidden_dimension = 300
model_params = {'context_size':context_size,
                'embedding_dimension':embedding_dimension,
                'hidden_dimension':hidden_dimension}

# initialize weights
C, W1, b1, W2, b2 = get_parameters(**model_params)
parameters = [C, W1, b1, W2, b2]

# training parameters
total_iterations = 50000
batch_size = 50
lrs = [0.1, 0.01, 0.001]
thresholds = list(map(lambda fl:int(round(fl, 0)), [((n+1)/len(lrs))*total_iterations for n in range(len(lrs))]))
training_params = {'total_iterations':total_iterations,
                   'batch_size':batch_size,
                   'learning rates':f'lrs: {lrs}, thresholds: {thresholds}'}

# building the training, dev and test datasets. 80/10/10 split.
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
# create datasets
Xtr, Ytr = build_dataset(words[:n1], context_size)
Xdev, Ydev = build_dataset(words[n1:n2], context_size)
Xte, Yte = build_dataset(words[n2:], context_size)

# janky way of adjusting the learning rate
def get_lr(lrs, thresholds, i):
  for n, t in enumerate(thresholds):
    if i < t:
      return lrs[n]

# for tracking stats
track_frequency = 1000
track_every = total_iterations//track_frequency
stats = {
         'model_params': model_params,
         'training_params':training_params,
         'training loss':0,
         'dev loss':0,
         'training time':0,
         'start at':str(time.time()).partition('.')[0],
         'end at':'',
         'track_frequency':track_frequency,
         'losses':{}
         }

track_counter = 0
# start training
for i in range(total_iterations):

  # construct minibatch
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))

  # forward pass
  emb = C[Xtr[ix]]
  h = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ytr[ix])

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = get_lr(lrs, thresholds, i)
  for p in parameters:
    p.data += -lr * p.grad
  
  # track stats
  if track_counter > track_every:
    stats['losses'][i] = {'lr':lr, 'loss':loss.item()}
    track_counter = 0
  else:
    track_counter += 1

stats['end at'] = str(time.time()).partition('.')[0]
stats['training time'] = int(stats['end at']) - int(stats['start at'])

# eval training loss
emb = C[Xtr]
h = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
stats['training loss'] = loss.item()

# eval dev loss
emb = C[Xdev]
h = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
stats['dev loss'] = loss.item()

print(f'dev loss: {stats["dev loss"]}')
toJson(stats, str(time.time()).partition('.')[0], overwrite=0)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# plotting stuff
steps = list(stats['losses'].keys())
losses = [d['loss'] for d in stats['losses'].values()]
lrs_used = [d['lr'] for d in stats['losses'].values()]
plt.plot(steps, losses)
# ------------------------------------------------------------------
