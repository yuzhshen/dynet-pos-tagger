import os
import pickle
import numpy as np
import dynet as dy

NUM_LAYERS=2
BATCH_SIZE = 16
INPUT_DIM=50
HIDDEN_DIM=10
TRAINING_USE_FIRST_N=100
GLOVE_PATH = '/Users/zhao/dev/#resources/glove.6B/glove.6B.50d.txt'

TREEBANK_POSFILE_PATH = '/Users/zhao/dev/dynet-pos-tagger/penn-treebank-pos'

def load_pretrained_glove(vocab_dict):
    original_glove_name = GLOVE_PATH.split('/')[-1]
    pruned_glove_name = original_glove_name[:-4]+'_pruned.npz'
    if not os.path.isfile(pruned_glove_name):
        print('Loading GloVe from original file...')
        glove = np.loadtxt(GLOVE_PATH, dtype='str', comments=None)
        words = glove[:, 0]
        vectors = glove[:, 1:].astype('float')
        print('Saving pruned GloVe for future loading...')
        where_in_vocab = [e for (e,w) in enumerate(words) if w in vocab_dict]
        pruned_words = np.array(words[where_in_vocab])
        pruned_vectors = np.array(vectors[where_in_vocab])
        np.savez(pruned_glove_name, words=pruned_words, vectors=pruned_vectors)
        print('Done.')
    else:
        print('Loading pruned GloVe...')
        npzfile = np.load(pruned_glove_name)
        words = npzfile['words']
        vectors = npzfile['vectors']
        print('Done.')
    return words, vectors

def format_as_folder_num(num):
    l = len(str(num))
    return (2-l)*'0'+str(num)

def load_dataset(folder_range):
    print('Creating dataset...')
    word_counter, tag_counter = {}, {}
    word_sentences = []
    tag_sentences = []
    for range_idx in folder_range:
        folder_path = TREEBANK_POSFILE_PATH+'/'+format_as_folder_num(range_idx)
        for pos_filename in os.listdir(folder_path):
            pos_filepath = folder_path+'/'+pos_filename
            with open(pos_filepath) as f:
                file_contents = f.readlines()
                single_word_seq = []
                single_tag_seq = []
                for line in [x for x in file_contents if '/' in x]:
                    line_list = line.replace('[','').replace(']','').strip(' \n').split(' ')
                    for item in line_list:
                        if item == '':
                            continue
                        item = item.replace('\/','-')
                        word, tag = item.split('/')
                        single_word_seq.append(word.lower())
                        single_tag_seq.append(tag)
                        tag_counter[tag] = tag_counter.get(tag,0)+1
                        word_counter[word] = word_counter.get(word,0)+1
                        if tag=='.':
                            word_sentences.append(single_word_seq)
                            tag_sentences.append(single_tag_seq)
                            single_word_seq = []
                            single_tag_seq = []
    print('Done.')
    return word_sentences, tag_sentences, word_counter, tag_counter

def shuffle_datasets(words, tags):
    indices = np.arange(len(words))
    np.random.shuffle(indices)
    shuffled_words = np.array(words)[indices]
    shuffled_tags = np.array(tags)[indices]
    return shuffled_words, shuffled_tags

class TaggerLstm(object):
    def __init__(self, layers, in_dim, hidden_dim, out_dim, model):
        self.builder = dy.BiRNNBuilder(layers, in_dim, out_dim, model, dy.VanillaLSTMBuilder)
        self.W = model.add_parameters((out_dim, out_dim))
        self.b = model.add_parameters((out_dim))
        self.training = True

    def __call__(self, sequence):
        if self.training:
            self.builder.set_dropout(0.3)
        else:
            self.builder.set_dropout(0.0)
        outputs = self.builder.transduce(sequence)
        results = [self.W*elt+self.b for elt in outputs]
        return results

word_dataset, tag_dataset, word_counter, tag_counter = load_dataset(range(0,5))
tag_set = list(tag_counter.keys())
tag_to_num = dict(zip(tag_set, range(len(tag_set))))
num_to_tag = dict(zip(range(len(tag_set)), tag_set))

glove_words, glove_vectors = load_pretrained_glove(word_counter)
word_to_num = dict(zip(glove_words, range(len(glove_words))))

m = dy.Model()
trainer = dy.AdamTrainer(m)
embeds = m.add_lookup_parameters((len(glove_vectors)+1, INPUT_DIM), init=np.vstack([glove_vectors, np.zeros(INPUT_DIM)]))
tagger = TaggerLstm(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, len(tag_counter), m)

# training
sum_of_losses = 0.0
sum_of_tokens = 0
for epoch in range(20):
    tagger.training = True
    shuffled_words, shuffled_tags = shuffle_datasets(word_dataset, tag_dataset)
    for enum_idx, (sequence,labels) in enumerate(zip(shuffled_words, shuffled_tags)):
        num_correct, num_total = 0, 0
        if enum_idx%BATCH_SIZE == 0:
            dy.renew_cg() # new computation graph
            total_loss_list = []    
        numeric_sequence = [word_to_num.get(x,len(glove_words)) for x in sequence]
        numeric_labels = [tag_to_num[x] for x in labels]
        vecs = [embeds[elt] for elt in numeric_sequence]
        preds = tagger(vecs)
        step_loss_list = []
        for i in range(len(preds)):
            step_loss = dy.pickneglogsoftmax(preds[i], numeric_labels[i])
            step_loss_list.append(step_loss)
            sum_of_losses += step_loss.npvalue()
            sum_of_tokens += 1
            num_total += 1
            if np.argmax(preds[i].npvalue()) == numeric_labels[i]:
                num_correct+=1
        total_loss = dy.esum(step_loss_list)
        total_loss_list.append(total_loss)
        if (enum_idx+1)%BATCH_SIZE == 0:
            batch_total_loss = dy.esum(total_loss_list)/BATCH_SIZE
            batch_total_loss.backward()
            trainer.update()
    print('Per token average loss: {:.4f}'.format(int(sum_of_losses)/sum_of_tokens))
    sum_of_losses = 0.0
    sum_of_tokens = 0
    print('Acc: {:.4f}'.format(num_correct/num_total))
    # TODO: compute validation accuracy


# # prediction
# tagger.training = False
# for sequence in [(1,4,12,1), (42,2), (56,2,17)]:
#     dy.renew_cg() # new computation graph
#     vecs = [embeds[i] for i in sequence]
#     preds = dy.softmax(tagger(vecs))
#     vals  = preds.npvalue()
#     print(np.argmax(vals), vals)