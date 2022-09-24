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
context_size = 4
embedding_dimension = 4
hidden_dimension = 200

# training parameters
total_iterations = 30000
batch_size = 200

# put stuff in handy dictionaries
training_params = {'total_iterations':total_iterations,
                   'batch_size':batch_size}
model_params = {'context_size':context_size,
                'embedding_dimension':embedding_dimension,
                'hidden_dimension':hidden_dimension}

# initialize weights
C, W1, b1, W2, b2 = get_parameters(**model_params)
parameters = [C, W1, b1, W2, b2]

# building the training, dev and test datasets. 80/10/10 split.
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
# create datasets
Xtr, Ytr = build_dataset(words[:n1], context_size)
Xdev, Ydev = build_dataset(words[n1:n2], context_size)
Xte, Yte = build_dataset(words[n2:], context_size)


# for tracking stats
stats = {
         'model_params': model_params,
         'training_params':training_params,
         'training loss':0,
         'dev loss':0,
         'training time':0,
         'start at':str(time.time()).partition('.')[0],
         'end at':'',
         'losses':{}
         }

lr = 1
lrs = []
losses = []
# start training
for i in range(total_iterations):

  # construct minibatch
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))

  # forward pass
  emb = C[Xtr[ix]]
  h = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits, Ytr[ix])

  if 1:
    losses.append(loss.log10().item())
  else:
    losses.append(loss.item())
  lrs.append(lr)

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  if i > 2000:
    lr = 0.01
  elif i > 1000:
    lr = 0.1
  else:
    lr = 1

  # change parameters
  for p in parameters:
    p.data += -lr * p.grad

stats['end at'] = str(time.time()).partition('.')[0]
stats['losses'] = losses
stats['training time'] = int(stats['end at']) - int(stats['start at'])

print(f'cont:{context_size}, emb:{embedding_dimension}, hidden:{hidden_dimension}, batch:{batch_size}, iterations:{total_iterations}, time:{stats["training time"]}')

# eval training loss
emb = C[Xtr]
h = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
stats['training loss'] = loss.item()

print(f'training loss: {stats["training loss"]}')

# eval dev loss
emb = C[Xdev]
h = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
stats['dev loss'] = loss.item()

print(f'dev loss: {stats["dev loss"]}')


if 0:
  print(f'dev loss: {stats["dev loss"]}')
  print(f'stats: {stats}')
  print(len(losses))
toJson(stats, str(time.time()).partition('.')[0], overwrite=0)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# plotting stuff

if 0:
  plt.plot(range(len(losses)), lrs)
else:
  plt.plot(range(len(losses)), losses)
# ------------------------------------------------------------------

# ------------------------------------------------------------------

# a rewrite of the previous code but with 2 extra hidden layers.

def get_bigboi_parameters(context_size, embedding_dimension, hidden_dimension):
  g = torch.Generator().manual_seed(2147483647)
  input_dimension = 27
  output_dimension = 27
  
  # weights connecting the one_hot encoded inputs to the embedding layer
  # though the one-hots are used to index into C, not multiplied by C.
  C = torch.randn((input_dimension, embedding_dimension), generator=g)

  # the weights connecting the embedding layer to the hidden layer
  W1 = torch.randn((context_size*embedding_dimension, hidden_dimension), generator=g)
  b1 = torch.randn(hidden_dimension, generator=g)

  # the weights connecting the hidden layers to the output layers (logits)
  W2 = torch.randn((int(hidden_dimension), int(hidden_dimension)), generator=g)
  b2 = torch.randn(W2.shape[1], generator=g)
  
  W3 = torch.randn((int(hidden_dimension), int(hidden_dimension)), generator=g)
  b3 = torch.randn(W3.shape[1], generator=g)
  
  W4 = torch.randn((int(hidden_dimension), output_dimension), generator=g)
  b4 = torch.randn(W4.shape[1], generator=g)

  # i forgot why i do this.
  parameters = [C, W1, b1, W2, b2, W3, b3, W4, b4]
  for p in parameters:
    p.requires_grad = True

  ridiculous = {s:'-'.join(map(str,tup)) for s, tup in list(zip('C,W1,b1,W2,b2,W3,b3,W4,b4'.split(','), [ [p.shape[i] for i in range(len(p.shape))] for p in parameters]))}
  print(ridiculous)
  return C, W1, b1, W2, b2, W3, b3, W4, b4
# ------------------------------------------------------------------

# ------------------------------------------------------------------

# setting hyperparameters, building datasets, training the model

# model parameters
context_size = 3
embedding_dimension = 3
hidden_dimension = 100

# training parameters
total_iterations = 20000
batch_size = 50
gravity = 0

# put stuff in handy dictionaries
training_params = {'total_iterations':total_iterations,
                   'batch_size':batch_size}
model_params = {'context_size':context_size,
                'embedding_dimension':embedding_dimension,
                'hidden_dimension':hidden_dimension}

# initialize weights
C, W1, b1, W2, b2, W3, b3, W4, b4 = get_bigboi_parameters(**model_params)
parameters = [C, W1, b1, W2, b2, W3, b3, W4, b4]


# building the training, dev and test datasets. 80/10/10 split.
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
# create datasets
Xtr, Ytr = build_dataset(words[:n1], context_size)
Xdev, Ydev = build_dataset(words[n1:n2], context_size)
Xte, Yte = build_dataset(words[n2:], context_size)


# for tracking stats
stats = {
         'model_params': model_params,
         'training_params':training_params,
         'training loss':0,
         'dev loss':0,
         'training time':0,
         'start at':str(time.time()).partition('.')[0],
         'end at':'',
         'losses':{}
         }

lr = 1
lrs = []
losses = []
# start training
for i in range(total_iterations):

  # construct minibatch
  ix = torch.randint(0, Xtr.shape[0], (batch_size,))

  # forward pass
  emb = C[Xtr[ix]]
  h1 = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
  h2 = torch.tanh(h1 @ W2 + b2)
  h3 = torch.tanh(h2 @ W3 + b3)
  logits = h3 @ W4 + b4
  loss = F.cross_entropy(logits, Ytr[ix])# + gravity*( (W1**2).mean() + (W2**2).mean() + (W3**2).mean() + (W4**2).mean() )

  if 1:
    losses.append(loss.log10().item())
  else:
    losses.append(loss.item())
  lrs.append(lr)

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  if i > 1000:
    lr = 0.01
  else:
    lr = 0.1

  # change parameters
  for p in parameters:
    p.data += -lr * p.grad

stats['end at'] = str(time.time()).partition('.')[0]
stats['losses'] = losses
stats['training time'] = int(stats['end at']) - int(stats['start at'])

print(f'cont:{context_size}, emb:{embedding_dimension}, hidden:{hidden_dimension}, batch:{batch_size}, iterations:{total_iterations}, time:{stats["training time"]}')

# eval training loss
emb = C[Xtr]
h1 = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
h2 = torch.tanh(h1 @ W2 + b2)
h3 = torch.tanh(h2 @ W3 + b3)
logits = h3 @ W4 + b4
loss = F.cross_entropy(logits, Ytr)# + 0.01*(W1**2).mean() + gravity*( (W1**2).mean() + (W2**2).mean() + (W3**2).mean() + (W4**2).mean() )
stats['training loss'] = loss.item()

print(f'training loss: {stats["training loss"]}')

# eval training loss
emb = C[Xdev]
h1 = torch.tanh(emb.view(-1, model_params['context_size']*model_params['embedding_dimension']) @ W1 + b1)
h2 = torch.tanh(h1 @ W2 + b2)
h3 = torch.tanh(h2 @ W3 + b3)
logits = h3 @ W4 + b4
loss = F.cross_entropy(logits, Ydev)# + 0.01*(W1**2).mean() + gravity*( (W1**2).mean() + (W2**2).mean() + (W3**2).mean() + (W4**2).mean() )
stats['dev loss'] = loss.item()

print(f'dev loss: {stats["dev loss"]}')


if 0:
  print(f'dev loss: {stats["dev loss"]}')
  print(f'stats: {stats}')
  print(len(losses))
toJson(stats, str(time.time()).partition('.')[0], overwrite=0)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# plotting stuff

if 0:
  plt.plot(range(len(losses)), lrs)
else:
  plt.plot(range(len(losses)), losses)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# sampling from the model
# first encode inputs, then embed them, then send them to the hidden layer, then to the final layer, then normalize logits and sample from them
def predict(context):
  ix = list(map(lambda s:stoi[s], list(context)))

  emb = C[ix]
  h = torch.tanh(emb.view(-1, context_size*embedding_dimension) @ W1 + b1)
  logits = h @ W2 + b2
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdims=True)

  choice = torch.multinomial(probs, num_samples=1, replacement=True).item()
  return itos[choice]

def makeName():
  context = '.'*context_size
  while True:
    prediction = predict(context)
    
    if prediction == '.':
      return context
      break
    else:
      context = context[1:] + prediction

for i in range(20):
  print(makeName())
# ------------------------------------------------------------------
