import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from utils.text_processing import *
from utils.metrics import *
from utils.embedding import *
from utils.loss import *
from utils.models import *
from utils.activation import *

def randomly (label):
    label2=[]
    for i in range(len(label)):
        value=random.uniform(label[i]-0.1, label[i]+0.1)
        label2.append(value)
    return label2
 

def random_data_set(df,df_test,dev,methode,target,text,corpus
                    ,nbdataset = 2
                    ,threshold = 0.9
                    ,label_reg=['non grave', 'grave']
                    ,core_model=core_model_LSTM
                    ,type_ngrams='words'
                    ,min_count = 2
                    ,optimizer = tf.keras.optimizers.Adagrad.__name__
                    ,loss=ncce
                    ,change_proba = True
                    ,print_model = True
                    ,print_top_acc = True
                    ,print_cnf_matrix = True
                    ,print_kde = True
                    ,w_model='f1'
                    ,epochs = 5
                    ,batch_size = 32
                    ,seuil='F1'
                    , load_emb=True

                   ):

    # S'assurer que target est dans df !

    if type(target) is str:
        df = df[[text,target]]

       

    # Separation de df et de la target
    df = df.reset_index(drop=True)
    df_target = df[target].copy()
   

    label = df_target.unique()


    min_label = df[target].value_counts().idxmin()


   
    analyser=create_analyser(df,text,type_ngrams)
    df=create_docs2(df,text,analyser)
 
    # Tokenization du text
    tokenizer = Tokenizer(lower=False, filters='')
    num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])
    tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
    tokenizer.fit_on_texts(df)
    df = tokenizer.texts_to_sequences(df)
    maxlen = len(max(df,key=len))
    df = pad_sequences(sequences=df, maxlen=maxlen)
  
    # Reshaping
    nsample = np.shape(df)[0]
    x = np.array(df).reshape((nsample,-1))
    y_tmp = np.array(df_target).reshape((nsample,))

    y = np.array(df_target).reshape((nsample,))

 
   

    x_test=create_docs2(df_test,text,analyser)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(sequences=x_test, maxlen=maxlen)
   
    x_dev=create_docs2(dev,text,analyser)
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_dev = pad_sequences(sequences=x_dev, maxlen=maxlen)
    
    input_dim = int(np.max(df) + 1)
    print(input_dim)
    sequence_length = df.shape[1] # 56
    vocabulary_size = input_dim
    embedding_dim=100
    embedding_matrix,ft_model=instantiateEmbenddingMatrix(corpus=corpus, tokenizer=tokenizer,
                                vocabulary_size=vocabulary_size,sequence_length=sequence_length,feature_size = embedding_dim,load=load_emb)

    label = df_target.unique()


    a2c = {}
    for j in range(len(label)):
        a2c[label[j]]=j



    y_test = np.array([ a2c[a] for a in df_test[target]])
    y_test = to_categorical(y_test)

    y_dev = np.array([ a2c[a] for a in dev[target]])
    y_dev = to_categorical(y_dev)
    n_out=len(label)

    for i in label:
 
        msk = y == i
 
        globals()['sub_data{}'.format(i)] = x[msk,:]

        globals()['y_sub_data{}'.format(i)]= y[msk]

        globals()['nrow_sub_data{}'.format(i)] = np.shape(globals()['sub_data{}'.format(i)])[0]

       
####################################################################
#          Type Methode                                            #
####################################################################
 
    if type(methode) is str:
        if methode=="balanced" or methode=="randomly" or methode=="mixte":
            percent = []
            for i in label :
                percent.append(globals()['nrow_sub_data{}'.format(i)])
            print(percent)
            s_p=np.sum(percent)
            min_p=np.min(percent)
            percent=[threshold*min_p/x for x in percent]

            percent = np.round(percent,3)
            print(percent)
        if methode=="classical":
            percent=np.ones(len(label))
            nbdataset=1
    elif type(methode) is list:
        if (all(isinstance(item, float)for item in methode)) or (all(isinstance(item, int) for item in methode)):
            percent=methode
            print(percent)
        else : raise(print('enter a list of int'))
    else : raise(print("enter balanced or randmoly or a list of int or float"))
   
 
    m=methode
    type_methode=["randomly","balanced","classical"]
    for i in range(nbdataset):
        print(i)
       
####################################################################
#          Dataset creation                                        #
####################################################################
 
        if m=="mixte":
            methode=random.choices(type_methode,weights=[0.7,0.2,0.1])
            print(methode)
       
        if  methode=="randomly":
            percent2=randomly (percent)
            print(percent2)
            j=0
            for l in label:
                globals()['percent{}'.format(l)]=percent2[j]
                j+=1
                globals()['smp_size{}'.format(l)]=int(globals()['percent{}'.format(l)]*globals()['nrow_sub_data{}'.format(l)])
      
        elif methode=="balanced"  :
            j=0
            for l in label:
                globals()['percent{}'.format(l)]=percent[j]
                j+=1
                globals()['smp_size{}'.format(l)]=int(globals()['percent{}'.format(l)]*globals()['nrow_sub_data{}'.format(l)])
        elif type(methode) is list:
            j=0
            for l in label:
                globals()['percent{}'.format(l)]=percent[j]
                j+=1
                globals()['smp_size{}'.format(l)]=int(globals()['percent{}'.format(l)]*globals()['nrow_sub_data{}'.format(l)])
 
 
       

        y_1= np.zeros((len(dev), n_out))

       
        
        pi=0
       
        somme = 0
        for k in label :
            somme += globals()['smp_size{}'.format(k)]
          
        x_train= np.empty((0,maxlen)) #pd.DataFrame()

        y_train = np.empty((0,1))

 
           
            
        for j in label :
           
            mskk = np.random.random_sample((globals()['nrow_sub_data{}'.format(j)],)) < globals()['percent{}'.format(j)]
            globals()['sub_data{}_1'.format(j)] = globals()['sub_data{}'.format(j)][mskk,:]

            x_train = np.concatenate([x_train,globals()['sub_data{}_1'.format(j)]])

            y_train = np.concatenate([y_train,a2c[j]*np.ones((np.shape(globals()['sub_data{}_1'.format(j)])[0],1))])

 
 
 
 
 

        y_train = to_categorical(y_train)
 
        outputs=[]
 
####################################################################
#          Models Parameters                                       #
####################################################################
 
        input_dim = int(np.max(x) + 1)
        sequence_length = x_train.shape[1] # 56
        vocabulary_size = input_dim

 
               
 # this returns a tensor
        print("Creating Model...")
        inputs = tf.keras.Input(shape=(sequence_length,), dtype='int32')
        outputs = core_model(inputs,sequence_length = int(x_train.shape[1]),
                         vocabulary_size =input_dim,n_out=n_out,embedding_dim=embedding_dim,
                         embedding_matrix=embedding_matrix) 
####################################################################
#           Creates a model                                        #
####################################################################
 
       
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(model.summary())
        checkpoint = tf.keras.callbacks.ModelCheckpoint('weights.best.{}.hdf5'.format(i), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        if optimizer == 'adam' :
            optim = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
        elif optimizer == 'rms' :
            optim = tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, decay=1e-4)
        elif optimizer == 'sgd':
            optim=tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=1.9)
        elif optimizer == 'adadelta':
            optim=tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        else :
            optim=optimizer
        model.compile(optimizer=optim, loss=loss, metrics=[tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')])
        if print_model == True :
            tf.keras.utils.plot_model(model,to_file='demo.png',show_shapes=True)
              
        print("Traning Model...")
        learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
 
        seqModel=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,shuffle=True
                          
                  , callbacks=[checkpoint,tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')]
                  , validation_data=(x_test, y_test)) 

        
        model=tf.keras.models.load_model('weights.best.{}.hdf5'.format(i) ,custom_objects={'swish':swish,'ncce':ncce
                                                                                                  
                                                                                                  })
       
 
 
       
        y_2 = model.predict(x_test)

        y_test_bis=pd.DataFrame(y_test)
        y_1_bis=pd.DataFrame(y_2,)

        y_1_bis.columns= a2c.keys()
        y_test_bis.columns= a2c.keys()
        y_1_bis[target]=y_1_bis.max(axis=1)
        y_test_bis[target]=0


        for u in range(len(list(a2c.keys()))):
            y_1_bis[target][y_1_bis[target]==y_1_bis[list(a2c.keys())[u]]]=u
            y_test_bis[target][y_test_bis[list(a2c.keys())[u]]==1]=u
        y_1_bis = y_1_bis.drop(list(a2c.keys()), axis=1)
        y_test_bis = y_test_bis.drop(list(a2c.keys()), axis=1)
####################################################################
#           Weighted Model                                         #
####################################################################
        if w_model=='f1':
            p=f1_score(y_test_bis, y_1_bis, average='weighted')
        if w_model=='precision':
            p=precision_score(y_test_bis, y_1_bis, average='weighted')
        if w_model=='f1_min':
            s=metrics.classification_report(y_test_bis, y_1_bis,target_names=a2c.keys(),output_dict =True)

            f1_grave = float(s['grave']['f1-score'])
            p=f1_grave

        if w_model=='precision_min':
            s=metrics.classification_report(y_test_bis, y_1_bis,target_names=a2c.keys())
            a=pd.DataFrame(report2dict(s)).T
            b=a.sort_values(by=['support'])
            p=b['precision'][0]
            lst=[a,b]
            del lst
        if w_model=='jaccard':
            p=jaccard_similarity_score(y_test_bis, y_1_bis)
        if nbdataset==1:
            p=1


        if p==0:
            p=0.00000000001
####################################################################
#           Save Model                                             #
####################################################################
        fichier = open("test.txt", "a")
        fichier.write(('{0}\t weights.best.{1}.hdf5\n'.format(p, i)))
        fichier.close()
        pi+=p
        y_2 = model.predict(x_dev)
 
 
        y_1+=np.multiply(y_2,p)                                                       
 

    y_1=np.multiply(y_1,1/pi)
   

 

    y_test_bis=pd.DataFrame(y_dev)
    y_test_bis.columns= a2c.keys()
 
   

   

    y_test_bis=pd.DataFrame(y_dev)
    y_1_bis=pd.DataFrame(y_1)
    y_1_bis.columns= a2c.keys()
    y_test_bis.columns= a2c.keys()
    y_1_bis[target]=y_1_bis.max(axis=1)
    y_test_bis[target]=0

 
 
####################################################################
#           Plot density                                           #
####################################################################  

    if print_kde == True  :
        fig, ax = plt.subplots(figsize=(11,11))
        for lab in label:
    # Subset to the airline
            subset = y_1_bis[lab]

    # Draw the density plot
            sns.distplot(subset, hist = True, kde = True,norm_hist=True,
                 kde_kws = {'linewidth': 3},label = lab)
        plt.legend()
        plt.savefig("plot_kde", dpi = 300)


####################################################################
#           Plot Top N                                             #
####################################################################

    if print_top_acc == True and n_out>2:
        prediction_evaluation(np.array(y_test_bis),np.array( y_1_bis))

####################################################################
#           Analysis of proba                                      #
####################################################################


        y_test_bis=pd.DataFrame(y_dev)
        y_test_bis.columns= a2c.keys()
        y_test_bis[target]=0


    if seuil=='F1':
        seuil=F1(y_test_bis[list(a2c.keys())[1]], y_1_bis[[list(a2c.keys())[1]]])
        print(seuil)
        y_1_bis[target] = np.where( y_1_bis[[list(a2c.keys())[1]]]>seuil,1, 0)
        for u in range(len(list(a2c.keys()))):
            y_test_bis[target][y_test_bis[list(a2c.keys())[u]]==1]=u
        y_1_bis = y_1_bis.drop(list(a2c.keys()), axis=1)
        y_test_bis = y_test_bis.drop(list(a2c.keys()), axis=1)



    if seuil=='max':

        y_1_bis[target]=y_1_bis.max(axis=1)
        for u in range(len(list(a2c.keys()))):
            y_1_bis[target][y_1_bis[target]==y_1_bis[list(a2c.keys())[u]]]=u
            y_test_bis[target][y_test_bis[list(a2c.keys())[u]]==1]=u


        sub1=[]
        sub2=[]   
        for i in range(len(y_1_bis)):    
            if y_1_bis[target][i]==1:
                if y_1_bis[target][i]==y_1_bis[target][i]:
                    sub1.append(y_1_bis[min_label][i])
                else:
                    sub2.append(y_1_bis[min_label][i])
        if len(sub1) !=0 :
            max_value = max(sub1)
            min_value = min(sub1)
            avg_value = sum(sub1)/len(sub1)

            print("stat of TP :")
            print ('max_{}'.format(max_value))
            print ('min_{}'.format(min_value))
            print ('avg_{}'.format(avg_value))
        if len(sub2) !=0 :   
            max_value = max(sub2)
            min_value = min(sub2)
            avg_value = sum(sub2)/len(sub2)
            print("stat of FP :")
            print ('max_{}'.format(max_value))
            print ('min_{}'.format(min_value))
            print ('avg_{}'.format(avg_value))


        y_1_bis = y_1_bis.drop(list(a2c.keys()), axis=1)
        y_test_bis = y_test_bis.drop(list(a2c.keys()), axis=1)
       
       
 
 
    
####################################################################
#           Plot CNF Matrix                                        #
#################################################################### 
 
    if print_cnf_matrix == True :
# Compute confusion matrix
        class_names = list(a2c.keys())
        cnf_matrix = confusion_matrix(y_test_bis, y_1_bis)
        np.set_printoptions(precision=2)
       
# Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
        plt.savefig("plot_cnf_matrix_{}.png".format(target), dpi = 300)
       
# Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
        plt.savefig("plot_cnf_matrix_norm_{}.png".format(target), dpi = 300)
  
 
        plt.show() 
####################################################################
#           Plot & save Metrics classif report                     #
####################################################################  
 
    print(metrics.classification_report(y_test_bis, y_1_bis,target_names=label))

    scoring=metrics.classification_report(y_test_bis, y_1_bis,target_names=label)
    return scoring