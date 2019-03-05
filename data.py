import os
import numpy as np

def words_and_vectors_from_glove_file(filepath):
    print('Loading GloVe from original file...')
    glove = np.loadtxt(filepath, dtype='str', comments=None)
    words = glove[:, 0]
    vectors = glove[:, 1:].astype('float')
    return words, vectors

def save_pruned_glove(words, vectors, vocab, save_filepath):
    print('Saving pruned GloVe for future loading...')
    where_in_vocab = [e for (e,w) in enumerate(words) if w in vocab]
    pruned_words = np.array(words[where_in_vocab])
    pruned_vectors = np.array(vectors[where_in_vocab])
    np.savez(save_filepath, words=pruned_words, vectors=pruned_vectors)

def load_pruned_glove(filepath):
    print('Loading pruned GloVe...')
    npzfile = np.load(filepath)
    return npzfile['words'], npzfile['vectors']

def load_pretrained_glove(vocab_dict, glove_filepath):
    original_glove_name = glove_filepath.split('/')[-1]
    pruned_glove_name = original_glove_name[:-4]+'_pruned.npz'
    if not os.path.isfile(pruned_glove_name):
        words, vectors = words_and_vectors_from_glove_file(glove_filepath)
        save_pruned_glove(words, vectors, vocab_dict, pruned_glove_name)
    else:
        words, vectors = load_pruned_glove(pruned_glove_name)
    return words, vectors

def format_as_folder_num(num):
    l = len(str(num))
    return (2-l)*'0'+str(num)

def parse_single_file(filepath, word_counter, tag_counter):
    word_sentences_in_file, tag_sentences_in_file = [], []
    words_in_sentence, tags_in_sentence = [], []
    with open(filepath) as f:
        file_contents = f.readlines()
    parseable_lines = [x for x in file_contents if '/' in x]
    for line in parseable_lines:
        split_line = line.replace('[','').replace(']','').strip(' \n').split(' ')
        for item in split_line:
            if item == '':
                continue
            item = item.replace('\/','-')
            word, tag = item.split('/')
            words_in_sentence.append(word.lower())
            tags_in_sentence.append(tag)
            tag_counter[tag] = tag_counter.get(tag,0)+1
            word_counter[word] = word_counter.get(word,0)+1
            if tag=='.':
                word_sentences_in_file.append(words_in_sentence)
                tag_sentences_in_file.append(tags_in_sentence)
                words_in_sentence = []
                tags_in_sentence = []
    return word_sentences_in_file, tag_sentences_in_file

def load_dataset(folder_range, treebank_path):
    print('Creating dataset...')
    word_counter, tag_counter = {}, {}
    all_word_sentences = []
    all_tag_sentences = []
    for range_idx in folder_range:
        folder_path = treebank_path+'/'+format_as_folder_num(range_idx)
        for pos_filename in os.listdir(folder_path):
            pos_filepath = folder_path+'/'+pos_filename
            word_sentences_in_file, tag_sentences_in_file = parse_single_file(pos_filepath, word_counter, tag_counter)
            all_word_sentences+=word_sentences_in_file
            all_tag_sentences+=tag_sentences_in_file
    return all_word_sentences, all_tag_sentences, word_counter, tag_counter

def shuffle_datasets(words, tags):
    indices = np.arange(len(words))
    np.random.shuffle(indices)
    shuffled_words = np.array(words)[indices]
    shuffled_tags = np.array(tags)[indices]
    return shuffled_words, shuffled_tags