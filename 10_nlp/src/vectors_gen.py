import codecs
from nltk import word_tokenize, sent_tokenize

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, pos_tag_sents
import itertools
from nltk.corpus import stopwords
import string
import gensim

corpuses = ['JohnGalsworthy', 'WilliamShakespeare', 'WinstonChurchill', 'BenjaminFranklin', 'MarkTwain']
for corpus in corpuses:

    print(corpus)
    path = 'Corpuses/'+corpus+'.txt'
    one_long_string = ""
    with codecs.open(path, 'r', 'utf-8-sig') as text_file:
        one_long_string = text_file.read()

    print('Tokenizing...')
    sentences = sent_tokenize(one_long_string)
    del(one_long_string)
    tokenized_sentences = map(word_tokenize, sentences)
    del(sentences)

    print('Lemmatizing...')
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_sentences = map(lambda sentence: map(wordnet_lemmatizer.lemmatize, sentence), tokenized_sentences)
    del(tokenized_sentences)

    print('POS tagging ...')
    pos_sentences = pos_tag_sents(lemmatized_sentences)
    del(lemmatized_sentences)

    tags_to_not_lowercase = set(['NNP', 'NNPS'])
    tags_to_preserve = set(['JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RBR', 'RBS','UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])

    def carefully_lowercase(words):
        return [(word.lower(), pos) if pos not in tags_to_not_lowercase else (word, pos)
                for (word, pos) in words]

    def filter_meaningful(words):
        return [word for (word, pos) in words if pos in tags_to_preserve]

    print('Lowercase ...')
    lowercased_pos_sentences = map(carefully_lowercase,  pos_sentences)
    del(pos_sentences)
    sentences_to_train_on = map(lambda words: [word for (word, pos) in words], lowercased_pos_sentences)
    
    print('Vocabulary ...')

    filtered = map(filter_meaningful, lowercased_pos_sentences)
    flatten = list(itertools.chain(*filtered))
    words_to_keep = set(flatten)

    del(filtered, flatten, lowercased_pos_sentences)

    print('Stop words')

    stop_words = set(stopwords.words('english') + list(string.punctuation) + ['wa'])
    def trim_rule(word, count, min_count):
        if word not in words_to_keep or word in stop_words:
            return gensim.utils.RULE_DISCARD
        else:
            return gensim.utils.RULE_DEFAULT


    print('Word2Vec training...')
    model = gensim.models.Word2Vec(sentences_to_train_on, min_count=15, trim_rule=trim_rule)

    print('Saving model...')
    model.wv.save_word2vec_format(fname=corpus+'.bin', binary=True)