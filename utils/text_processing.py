from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer


def create_analyser(data,col,type_ngrams = 'words') :
    if type_ngrams == 'words' :
        k1 = 1
        k2 = 1
    elif type_ngrams == 'N_grams' :
        k1 = 1
        k2 = 3
    elif type_ngrams == 'Only_N_grams' :
        k1 = 2
        k2 = 3
    vectorizer = TfidfVectorizer(ngram_range=(k1,k2),lowercase =False,stop_words=None)
    vectorizer.fit(list(data[col]))
    analyser = vectorizer.build_analyzer()
    
    return analyser

def create_docs2(data,col,analyser) :
    new = data.apply(lambda x:analyser(x[col]), axis=1)
    return new 