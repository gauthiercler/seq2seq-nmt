## Neural Machine Translation with seq2seq architecture

![](https://cdn-images-1.medium.com/max/1000/1*PFPSLEjIe152uR9UR59LDA.png)  
*Image from Standford CS224N*

#### Usage

````shell script
usage: main.py [-h] [--learning-rate LEARNING_RATE]
               [--lang {afr,sqi,arq,ara,aze,eus,bel,ben,ber,bos,bul,mya,yue,cat,
                        ceb,dtp,cbk,cmn,chv,hrv,ces,dan,nld,est,fin,fra,glg,kat,
                        deu,ell,heb,hin,hun,isl,ilo,ind,ita,jpn,kab,kan,pam,kha,
                        khm,kor,kur,lvs,lit,nds,mkd,zsm,mal,mri,mar,nst,max,nob,
                        pes,pol,por,ron,rus,srp,slk,slv,spa,swe,tgl,tam,tat,tel,
                        tha,tur,tuk,ukr,urd,uig,vie,war,zza}]
               [--reversed] [--epochs EPOCHS] [--dropout DROPOUT]
               [--max-words MAX_WORDS] [--cv-size CV_SIZE] [--use-attention]
               [--verbose-rate VERBOSE_RATE]
               [--sets-size SETS_SIZE [SETS_SIZE ...]]
               [--teacher-forcing {beam-search,curriculum}]

optional arguments:
  -h, --help
     show this help message and exit
  --learning-rate LEARNING_RATE
     step size toward minimum of loss (default: 0.01)
  --lang {afr,sqi,arq,ara,aze,eus,bel,ben,ber,bos,bul,mya,yue,cat,
          ceb,dtp,cbk,cmn,chv,hrv,ces,dan,nld,est,fin,fra,glg,kat,
          deu,ell,heb,hin,hun,isl,ilo,ind,ita,jpn,kab,kan,pam,kha,
          khm,kor,kur,lvs,lit,nds,mkd,zsm,mal,mri,mar,nst,max,nob,
          pes,pol,por,ron,rus,srp,slk,slv,spa,swe,tgl,tam,tat,tel,
          tha,tur,tuk,ukr,urd,uig,vie,war,zza}
     lang pair to translate from/to (default: fra)
  --reversed
     if defined, translate from --lang to english, otherwise translate from
     english (default: False)
  --epochs EPOCHS
     number of epochs to train on dataset (default: 10)
  --dropout DROPOUT
     probability to apply dropout for regularization (default: 0.1)
  --max-words MAX_WORDS
     maximum number of words by sentence (default: 3)
  --cv-size CV_SIZE
     size of the context vector to represent source sequence (default: 256)
  --use-attention
     use attention mechanism in decoder (default: False)
  --verbose-rate VERBOSE_RATE
     print interval (default: 10)
  --sets-size SETS_SIZE [SETS_SIZE ...]
     percentage for train, dev and test sets (default: [0.8, 0.1, 0.1])
  --teacher-forcing {beam-search,curriculum}
     teacher forcing technique to use (default: curriculum)

````

Data from https://www.manythings.org/anki/, extracted from Tatoeba Project

#### TODO
- [ ] Beam Search
- [X] Curriculum Teacher Forcing
- [ ] Scheduled sampling
- [ ] use LSTM
- [X] Attention
- [X] Dropout
- [X] Save model state at defined intervals
- [ ] Use trained Embeddings (GloVe, Doc2Vec...)
- [ ] Streamlit playground with trained models (or upload model)
- [X] Multiple language pairs support
- [X] train, dev, test sets
- [ ] buckets
- [ ] backward feeding
- [X] Bleu score
