import os, json, copy, time
import numpy as np

onKaggle = 1
debug = False
logit_amount = 3
context_length = 10

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
        if float(logit) != 0.0:
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

def rolling_window(sequence, length):
    subsets = []
    
    subset = sequence[:length]
    subsets.append(subset)
    for item in sequence[length:]:
        subset = subset[1:] + [item]
        subsets.append(subset)
    
    return subsets

def examples_from_sequence(sequence):    
    start_token = my_vocab_reverse['<start>']
    end_token = my_vocab_reverse['<end>']
    padding_token = my_vocab_reverse['<padding>']
    
    full = [padding_token]*(context_length-2) + [start_token] + [*sequence]
    res = rolling_window(full, context_length)
    #print(f'full: {full}')
    #print(f'res: {res}')
    return res

# prompts have negative indices.
def make_examples(logprobs, prompt):
    xs = []
    ys = []
    
    available = my_vocab.keys()
    
    sequence = [my_vocab_reverse[prompt]]
    for generation in logprobs:
        # make logits array
        logits = np.zeros((len(my_vocab),1), dtype=np.float32)
        for alt_token in generation['before'][:logit_amount]:
            alt_id = alt_token[0][0]
            logit = alt_token[1][0]
            logits[to_position[alt_id]] = copy.deepcopy(logit)
        ys.append(copy.deepcopy(logits))
        
        chosen_id = generation['chosen'][0][0][0]
        if chosen_id not in available:
            error_message = f'{chosen_id} was added but it\'s not in vocab'
            if onKaggle:
                raise SystemExit(error_message)
            else:
                exit(error_message)
        sequence.append(copy.deepcopy(chosen_id))
        
        if debug:
            print(f'last chosen was: {generation["chosen"][0][0][0]}')
            
    xs += examples_from_sequence(sequence[:-1])

    if len(xs) != len(ys):
        error_message = f'cause: unequal length ({len(xs)},{len(ys)})\nxs:{xs}\nys:{ys}'
        if onKaggle:
            raise SystemExit(error_message)
        else:
            exit(error_message)
        
    return xs, ys

# load things
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
    for a,b in zip([p for p in pred if float(p) != 0.0], decode_array(pred)):
        print(b)
    print('------------------------------')
