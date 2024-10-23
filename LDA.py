from gensim import corpora, models
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


class LDATopicModel:
    def __init__(self, num_topics=10):
        self.num_topics = num_topics
        self.dictionary = None
        self.lda_model = None

    def fit(self, texts):
        # Tokenize the texts
        tokenized_texts = [word_tokenize(text) for text in texts]

        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]

        # Train LDA model
        self.lda_model = models.LdaModel(corpus, num_topics=self.num_topics, id2word=self.dictionary)

    def transform(self, texts):
        # Tokenize the texts
        tokenized_texts = [word_tokenize(text) for text in texts]

        # Assign topics to each text
        corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]
        topics = self.lda_model.get_document_topics(corpus)

        # Extract the dominant topic for each text
        dominant_topics = [max(t, key=lambda x: x[1])[0] for t in topics]
        return dominant_topics

    def fit_transform(self, texts):
        # Tokenize the texts
        tokenized_texts = [word_tokenize(text) for text in texts]

        # Create dictionary and corpus
        self.dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]

        # Train LDA model
        self.lda_model = models.LdaModel(corpus, num_topics=self.num_topics, id2word=self.dictionary)

        # Assign topics to each text
        topics = self.lda_model.get_document_topics(corpus)

        # Extract the dominant topic for each text
        dominant_topics = [max(t, key=lambda x: x[1])[0] for t in topics]
        return dominant_topics
