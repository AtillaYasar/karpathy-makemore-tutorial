# note: incomplete. this is just a convenient place to store things. requires a specific type of json files, and a gpt2 vocabulary

import os, json, copy, time
import numpy as np

def openListFile(filename):
    with open(filename, "r") as f:
        lst = json.load(f)
    return lst

def decode(ids):
    if type(ids) == int:
        return gpt2_vocab[ids]
    return list(map(lambda ID:gpt2_vocab[int(ID)], ids))

def my_decode(ids):
    if type(ids) == int:
        return my_vocab[ids]
    return list(map(lambda ID:my_vocab[int(ID)], ids))

def my_encoder(sequence):
    if type(sequence) == str:
        return my_vocab_reverse[sequence]
    return list(map(lambda string:my_vocab_reverse[string], sequence))

def make_vocab_subset(dictionaries, prompts, logit_amount=3):
    # these 3 are <start>, <end>, and padding.
    my_vocab = {-(n+1):string for n, string in enumerate(['<start>', '<end>', '<padding>'] + prompts)}
    for logprobs in dictionaries:
        for generation in logprobs:
            alts = []
            for alt_token in generation['before']:
                token_id = alt_token[0][0]
                my_vocab[token_id] = decode(alt_token[0][0])
    return my_vocab

def examples_from_sequence(sequence, context_length):
    padded_Xs = []
    context = [my_vocab_reverse['<padding>']]*context_length
    for token in sequence:
        nxt = token
        context = context[1:] + [nxt]
        padded_Xs.append(copy.deepcopy(context))
    
    return padded_Xs
# prompts have negative indices.
def make_examples(logprobs, prompt, logit_amount=3):
    xs = []
    ys = []

    sequence = [my_vocab_reverse[prompt]]
    for generation in logprobs:
        logits = np.zeros((len(my_vocab),1), dtype=np.float32)
        for alt_token in generation['before'][:logit_amount]:
            alt_id = alt_token[0][0]
            
            if alt_id not in my_vocab.keys():
                print(f'{alt_id} not in vocab.\nlogprobs:{logprobs}')
                
            logit = alt_token[1][0]
            logits[to_position[alt_id]] = copy.deepcopy(logit)
        ys.append(copy.deepcopy(logits))
        sequence.append(generation['chosen'][0][0][0])
    xs += examples_from_sequence(sequence, context_length=5)

    if len(xs) != len(ys):
        exit('unequal length')
    return xs, ys

# load things
onKaggle = 1
if onKaggle:
    gpt2_vocab = openListFile('/kaggle/input/gpt2-tokens/gpt2 tokens')
    data_folder = '/kaggle/input/jsons'
else:
    gpt2_vocab = openListFile(r'C:\Users\Gebruiker\Desktop\ATTG exploration/gpt2 tokens')
    data_folder = os.getcwd()
    

json_amount = 5
names = [f'{data_folder}/{f}' for f in os.listdir(data_folder) if 'raw' in f][:json_amount]

jsons = []
prompts = []
for name in names:
    d = openListFile(name)['raw response']
    jsons.append(d['logprobs'])
    prompts.append(d['payload']['input'])

# n+1 because it's going to start counting from -1
prompt_encodings = {p:n+1 for n,p in enumerate(prompts)}

# this has just a list of indices
my_vocab = make_vocab_subset(jsons, prompts) # for decoding
my_vocab_reverse = {v:k for k,v in my_vocab.items()} # for encoding
to_position = {index:pos for pos, index in enumerate(list(my_vocab.keys()))} # for finding position in numpy array, given index
from_position = {v:k for k,v in to_position.items()} # for finding the index, given numpy array position

Xs = []
Ys = []
t0 = time.time()
for i in range(json_amount):
    xs, ys = make_examples(jsons[i], prompts[i])
    Xs += copy.deepcopy(xs)
    Ys += copy.deepcopy(ys)
for seq,pred in zip(Xs, Ys):
    print(seq)
    print(my_decode(seq))
    print(pred[:10])
    print()
