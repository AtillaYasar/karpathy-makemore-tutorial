# note: incomplete and doesn't work correctly. this is just a convenient place to store things. requires a specific type of json files, and a gpt2 vocabulary

debug = False
logit_amount = 3
context_length = 10

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

def decode_array(arr):
    converted = []
    for pos, logit in enumerate(np.nditer(arr)):
        if logit != 0:
            converted.append((from_position[pos], my_vocab[from_position[pos]], float(logit)))
    return converted

def my_decode(ids):
    if type(ids) == int:
        return my_vocab[ids]
    return list(map(lambda ID:my_vocab[int(ID)], ids))

def my_encoder(sequence):
    if type(sequence) == str:
        return my_vocab_reverse[sequence]
    return list(map(lambda string:my_vocab_reverse[string], sequence))

def make_vocab_subset(dictionaries, prompts):
    # these 3 are <start>, <end>, and padding.
    my_vocab = {-(n+1):string for n, string in enumerate(['<start>', '<end>', '<padding>'] + prompts)}
    for logprobs in dictionaries:
        for generation in logprobs:
            alts = []
            for alt_token in generation['before'][:logit_amount]:
                token_id = alt_token[0][0]
                if token_id not in my_vocab:
                    my_vocab[token_id] = decode(token_id)

            chosen_id = generation['chosen'][0][0][0]
            if chosen_id not in my_vocab:
                my_vocab[chosen_id] = decode(chosen_id)
    return my_vocab

def examples_from_sequence(sequence):
    available = list(my_vocab.keys())
    padded_Xs = []
    
    start_token = my_vocab_reverse['<start>']
    end_token = my_vocab_reverse['<end>']
    
    context = [my_vocab_reverse['<padding>']]*context_length
    for token_id in [start_token, *sequence, end_token]:
        nxt = token_id
        context = context[1:] + [nxt]
        padded_Xs.append(copy.deepcopy(context))
    
    return padded_Xs
# prompts have negative indices.
def make_examples(logprobs, prompt):
    xs = []
    ys = []
    
    available = my_vocab.keys()
    
    sequence = [my_vocab_reverse[prompt]]
    for generation in logprobs:
        logits = np.zeros((len(my_vocab),1), dtype=np.float32)
        for alt_token in generation['before'][:logit_amount]:
            alt_id = alt_token[0][0]
            logit = alt_token[1][0]
            logits[to_position[alt_id]] = copy.deepcopy(logit)
        ys.append(copy.deepcopy(logits))
        
        chosen_id = generation['chosen'][0][0][0]
        if chosen_id not in available:
            print(f'{chosen_id} was added but it\'s not in vocab')
            raise SystemExit("Exit from script")
            
        sequence.append(copy.deepcopy(chosen_id))
        if debug:
            print(f'last chosen was: {generation["chosen"][0][0][0]}')
    xs += examples_from_sequence(sequence)

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
    
# gathering the logprobs
json_amount = 5
names = [f'{data_folder}/{f}' for f in os.listdir(data_folder) if 'raw' in f][:json_amount]
jsons = []
prompts = []
outputs = []
for name in names:
    d = openListFile(name)['raw response']
    jsons.append(d['logprobs'])
    prompts.append(d['payload']['input'])
    outputs.append(d['output'])

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
    for a,b in zip(pred, decode_array(pred)):
        print(a,b)
    print('------------------------------')
