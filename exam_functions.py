def batch_preprocess_texts(
    texts,
    nlp=None,
    remove_stopwords=True,
    remove_punct=True,
    use_lemmas=False,
    disable=["ner"],
    batch_size=50,
    n_process=-1,
):
    """Efficiently preprocess a collection of texts using nlp.pipe()
    Args:
        texts (collection of strings): collection of texts to process (e.g. df['text'])
        nlp (spacy pipe), optional): Spacy nlp pipe. Defaults to None; if None, it creates a default 'en_core_web_sm' pipe.
        remove_stopwords (bool, optional): Controls stopword removal. Defaults to True.
        remove_punct (bool, optional): Controls punctuation removal. Defaults to True.
        use_lemmas (bool, optional): lemmatize tokens. Defaults to False.
        disable (list of strings, optional): named pipeline elements to disable. Defaults to ["ner"]: Used with nlp.pipe(disable=disable)
        batch_size (int, optional): Number of texts to process in a batch. Defaults to 50.
        n_process (int, optional): Number of CPU processors to use. Defaults to -1 (meaning all CPU cores).
    Returns:
        list of tokens
    """
    # from tqdm.notebook import tqdm
    from tqdm import tqdm
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    processed_texts = []
    for doc in tqdm(nlp.pipe(texts, disable=disable, batch_size=batch_size, n_process=n_process)):
        tokens = []
        for token in doc:
            # Check if should remove stopwords and if token is stopword
            if (remove_stopwords == True) and (token.is_stop == True):
                # Continue the loop with the next token
                continue
            # Check if should remove stopwords and if token is stopword
            if (remove_punct == True) and (token.is_punct == True):
                continue
            # Check if should remove stopwords and if token is stopword
            if (remove_punct == True) and (token.is_space == True):
                continue
            
            ## Determine final form of output list of tokens/lemmas
            if use_lemmas:
                tokens.append(token.lemma_.lower())
            else:
                tokens.append(token.text.lower())
        processed_texts.append(tokens)
    return processed_texts

def get_ngram_measures_finder(tokens, ngrams=2, measure='raw_freq', top_n=None, min_freq = 1,
                             words_colname='Words'):
    import nltk
    if ngrams == 4:
        MeasuresClass = nltk.collocations.QuadgramAssocMeasures
        FinderClass = nltk.collocations.QuadgramCollocationFinder
        
    elif ngrams == 3: 
        MeasuresClass = nltk.collocations.TrigramAssocMeasures
        FinderClass = nltk.collocations.TrigramCollocationFinder
    else:
        MeasuresClass = nltk.collocations.BigramAssocMeasures
        FinderClass = nltk.collocations.BigramCollocationFinder

    measures = MeasuresClass()
    
   
    finder = FinderClass.from_words(tokens)
    finder.apply_freq_filter(min_freq)
    if measure=='pmi':
        scored_ngrams = finder.score_ngrams(measures.pmi)
    else:
        measure='raw_freq'
        scored_ngrams = finder.score_ngrams(measures.raw_freq)

    df_ngrams = pd.DataFrame(scored_ngrams, columns=[words_colname, measure.replace("_",' ').title()])
    if top_n is not None:
        return df_ngrams.head(top_n)
    else:
        return df_ngrams

def make_custom_nlp(
    disable=["ner"],
    contractions=["don't", "can't", "couldn't", "you'd", "I'll"],
    stopwords_to_add=[],
    stopwords_to_remove=[],
    spacy_model = "en_core_web_sm"
):
    """Returns a custom spacy nlp pipeline.
    
    Args:
        disable (list, optional): Names of pipe components to disable. Defaults to ["ner"].
        contractions (list, optional): List of contractions to add as special cases. Defaults to ["don't", "can't", "couldn't", "you'd", "I'll"].
        stopwords_to_add(list, optional): List of words to set as stopwords (word.is_stop=True)
        stopwords_to_remove(list, optional): List of words to remove from stopwords (word.is_stop=False)
        spacy_model(string, optional): String to select a spacy language model. (Defaults to "en_core_web_sm".)
                            Additional Options:  "en_core_web_md", "en_core_web_lg"; 
                            (Must first download the model by name in the terminal:
                            e.g.  "python -m spacy download en_core_web_lg" )
            
    Returns:
        nlp pipeline: spacy pipeline with special cases and updated nlp.Default.stopwords
    """
    # Load the English NLP model
    nlp = spacy.load(spacy_model, disable=disable)
    
    # Adding Special Cases 
    # Loop through the contractions list and add special cases
    for contraction in contractions:
        special_case = [{"ORTH": contraction}]
        nlp.tokenizer.add_special_case(contraction, special_case)
    
    # Adding stopwords
    for word in stopwords_to_add:
        # Set the is_stop attribute for the word in the vocab dict to true.
        nlp.vocab[
            word
        ].is_stop = True  # this determines spacy's treatmean of the word as a stop word
        # Add the word to the list of stopwords (for easily tracking stopwords)
        nlp.Defaults.stop_words.add(word)
    
    # Removing Stopwords
    for word in stopwords_to_remove:
        
        # Ensure the words are not recognized as stopwords
        nlp.vocab[word].is_stop = False
        nlp.Defaults.stop_words.discard(word)
        
    return nlp

import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser','ner'])

def lemmatize(text):
    doc = nlp(text)
    lemmatized_text = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return lemmatized_text