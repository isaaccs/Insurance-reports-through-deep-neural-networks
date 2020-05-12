from gensim.models.fasttext import FastText
import numpy as np


def instantiateEmbenddingMatrix(corpus, tokenizer,
                                vocabulary_size,sequence_length, 
                                feature_size , window_context = 3, min_word_count = 10,
                                sample = 1e-3, sg=0, overwrite=False, load=True):

    if load == True:
        try:
            embedding_matrix = None
            print("Loading embedding matrix...")
            embedding_matrix = np.genfromtxt('embedding.csv', delimiter=',')
            ft_model = FastText.load("ft_model.model")
            
        except:
            embedding_matrix = None
            pass
    else:
        embedding_matrix = None
    if embedding_matrix is None or overwrite or load == False:
        # Word vector dimensionality  
        # Context window size                                                                                    
        # Minimum word count                        
        # Downsample setting for frequent words
        # sg decides whether to use the skip-gram model (1) or CBOW (0)
        ft_model = FastText(corpus,min_n=0,max_n=3
                            , size=feature_size, 
                            window=window_context,min_count=min_word_count,sample=sample, sg=sg, iter=1000)    

        ft_model.save("ft_model.model")
        print('Preparing embedding matrix...')
        words_not_found = []
        nb_words = vocabulary_size
        word_index = tokenizer.word_index
        embedding_matrix = np.zeros((nb_words, feature_size))
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            embedding_vector = ft_model.wv.get_vector(word)
            if (embedding_vector is not None) and len(embedding_vector) > 0:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                words_not_found.append(word)
        print(embedding_matrix.shape)
        print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        np.savetxt('embedding.csv', embedding_matrix, delimiter=',')
            
    return embedding_matrix,ft_model
