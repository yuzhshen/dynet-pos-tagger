import numpy as np
import dynet as dy
# project file imports
import data

NUM_LAYERS=2
BATCH_SIZE = 16
INPUT_DIM=50
HIDDEN_DIM=30
GLOVE_PATH = '/Users/zhao/dev/#resources/glove.6B/glove.6B.50d.txt'
POS_TREEBANK_PATH = '/Users/zhao/dev/dynet-pos-tagger/penn-treebank-pos'

class TaggerLstm(object):
    def __init__(self, layers, in_dim, hidden_dim, out_dim, model):
        self.builder = dy.BiRNNBuilder(layers, in_dim, hidden_dim, model, dy.VanillaLSTMBuilder)
        self.W = model.add_parameters((out_dim, hidden_dim))
        self.b = model.add_parameters((out_dim))
        self.training = True

    def __call__(self, sequence):
        if self.training:
            self.builder.set_dropout(0.2)
        else:
            self.builder.set_dropout(0.0)
        outputs = self.builder.transduce(sequence)
        results = [self.W*elt+self.b for elt in outputs]
        return results

word_dataset, tag_dataset, word_counter, tag_counter = data.load_dataset(range(0,5), POS_TREEBANK_PATH)
tag_set = list(tag_counter.keys())
tag_to_num = dict(zip(tag_set, range(len(tag_set))))

glove_words, glove_vectors = data.load_pretrained_glove(word_counter, GLOVE_PATH)
word_to_num = dict(zip(glove_words, range(len(glove_words))))

m = dy.Model()
trainer = dy.AdamTrainer(m)
embeds = m.add_lookup_parameters((len(glove_vectors)+1, INPUT_DIM), init=np.vstack([glove_vectors, np.zeros(INPUT_DIM)]))
tagger = TaggerLstm(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, len(tag_counter), m)

# training
for epoch in range(20):
    print('Training: Epoch {}'.format(epoch+1))
    sum_of_losses = 0.0
    sum_of_tokens = 0
    tagger.training = True
    shuffled_words, shuffled_tags = data.shuffle_datasets(word_dataset, tag_dataset)
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