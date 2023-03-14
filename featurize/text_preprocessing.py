import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    A class for performing common text preprocessing tasks.

    Parameters:
        stop_words: list of strings
            A list of stop words to remove from text.

        lemmatizer: object
            An instance of the WordNetLemmatizer class from the NLTK library.
    """

    def __init__(self, stop_words=None, lemmatizer=None):
        if stop_words is None:
            self.stop_words = stopwords.words('english')
        else:
            self.stop_words = stop_words

        if lemmatizer is None:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = lemmatizer

    def remove_special_chars(self, text):
        """
        Remove special characters from text.

        Parameters:
            text: str
                The text to process.

        Returns:
            str
                The text with special characters removed.
        """
        processed_text = re.sub(r'[^\w\s]', '', text)
        return processed_text

    def remove_stop_words(self, text):
        """
        Remove stop words from text.

        Parameters:
            text: str
                The text to process.

        Returns:
            str
                The text with stop words removed.
        """
        words = word_tokenize(text)
        filtered_words = [w for w in words if w.lower() not in self.stop_words]
        processed_text = ' '.join(filtered_words)
        return processed_text

    def lemmatize_text(self, text):
        """
        Lemmatize text.

        Parameters:
            text: str
                The text to process.

        Returns:
            str
                The text with words lemmatized.
        """
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(w) for w in words]
        processed_text = ' '.join(lemmatized_words)
        return processed_text
