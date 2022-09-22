# ------------------------------------------------------------------

# making a character-level 'language model' (multilayer perceptron), trained on a list of names.
# its layers are: (one_hot encoded input vectors -->) embedding --> hidden --> output (logits) --> softmax (probabilities)

import torch, random, time, json, copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

# import dataset (list of names), mapping from letter to index and vice versa
# '.' will be the start and end character

words = open('/kaggle/input/nameslist/names.txt', 'r').read().split('\n')
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


def train(params):
    # model parameters
    context_size = params['context_size']
    embedding_dimension = params['embedding_dimension']
    hidden_dimension = params['hidden_dimension']
    
    # training parameters
    batch_size = params['batch_size']
    total_iterations = params['total_iterations']
          
    # initialize weights
    model_params = {k:v for k,v in params.items() if k in ['embedding_dimension', 'context_size', 'hidden_dimension']}
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
             'params':params,
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
      h = torch.tanh(emb.view(-1, context_size*embedding_dimension) @ W1 + b1)
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

      if i > 4000:
        lr = 0.01
      elif i > 2000:
        lr = 0.1
      else:
        lr = 1

      # change parameters
      for p in parameters:
        p.data += -lr * p.grad

    stats['end at'] = str(time.time()).partition('.')[0]
    stats['losses'] = losses
    stats['training time'] = int(stats['end at']) - int(stats['start at'])

    # eval training loss
    emb = C[Xtr]
    h = torch.tanh(emb.view(-1, context_size*embedding_dimension) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)
    stats['training loss'] = loss.item()
    
    trLoss = round(loss.item(), 3)

    # eval dev loss
    emb = C[Xdev]
    h = torch.tanh(emb.view(-1, context_size*embedding_dimension) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ydev)
    stats['dev loss'] = loss.item()

    devLoss = round(loss.item(), 3)
    
    return {'dev loss':devLoss, 'training loss':trLoss, 'training time':stats['training time'], 'params':params}

def explore_hyperparameters(params, string, frm, to, step_size):
    # for better dictionary printing
    def dString(d):
        return ', '.join([f'{str(k)}:{str(v)}' for k,v in d.items()])
    
    params[string] = frm
    results = []
    
    steps = (to-frm)//(step_size-1)+1
    for i in range(steps):
        
        results.append(copy.deepcopy(train(params)))
        
        params[string] += step_size
    
    print('++++++++++++++++++++++++++++++++')
    frozen =  {k:v for k,v in params.items() if k != string}
    print(f'frozen parameters:\n{dString(frozen)}')
    print(f'\niterated on {string} between {frm}-{to}')
    
    ranked = sorted(results, key=lambda d:d['dev loss'])
    print(f'best one was {ranked[0]["params"][string]}')
    print(f'\nfuller results:')
    for d in ranked:
        print(dString(d))
    print('++++++++++++++++++++++++++++++++')

params = {
    'context_size':3,
    'embedding_dimension':6,
    'hidden_dimension':300,
    'total_iterations':1000,
    'batch_size':0
}
explore_hyperparameters(params, 'batch_size', 50, 200, 50)
# ------------------------------------------------------------------
