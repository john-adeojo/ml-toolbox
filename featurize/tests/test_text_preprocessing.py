import unittest
from text_preprocessing import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.tp = TextPreprocessor()

    def test_remove_special_chars(self):
        """
        Test that remove_special_chars removes special characters from text.
        """
        text = "This is some text!@#$%^&*()_+-={}[]|\\;:'\",.<>/?`~"
        processed_text = self.tp.remove_special_chars(text)
        expected_text = "This is some text"
        self.assertEqual(processed_text, expected_text)

    def test_remove_stop_words(self):
        """
        Test that remove_stop_words removes stop words from text.
        """
        text = "This is some text that includes stop words"
        processed_text = self.tp.remove_stop_words(text)
        expected_text = "text includes stop words"
        self.assertEqual(processed_text, expected_text)

    def test_lemmatize_text(self):
        """
        Test that lemmatize_text lemmatizes text.
        """
        text = "This is some text that needs to be lemmatized"
        processed_text = self.tp.lemmatize_text(text)
        expected_text = "This is some text that need to be lemmatized"
        self.assertEqual(processed_text, expected_text)

    def test_text_preprocessing(self):
        """
        Test that text_preprocessing performs all text preprocessing steps correctly.
        """
        text = "This is some text!@#$%^&*()_+-={}[]|\\;:'\",.<>/?`~ that includes stop words and needs to be lemmatized"
        processed_text = self.tp.remove_special_chars(text)
        processed_text = self.tp.remove_stop_words(processed_text)
        processed_text = self.tp.lemmatize_text(processed_text)
        expected_text = "text includes need lemmatized"
        self.assertEqual(processed_text, expected_text)

    def test_remove_special_chars_with_no_special_chars(self):
        """
        Test that remove_special_chars returns the original text when there are no special characters.
        """
        text = "This is some text"
        processed_text = self.tp.remove_special_chars(text)
        self.assertEqual(processed_text, text)

    def test_remove_stop_words_with_no_stop_words(self):
        """
        Test that remove_stop_words returns the original text when there are no stop words.
        """
        text = "This text does not contain any stop words"
        processed_text = self.tp.remove_stop_words(text)
        self.assertEqual(processed_text, text)

    def test_lemmatize_text_with_single_word(self):
        """
        Test that lemmatize_text returns the original text when given a single word.
        """
        text = "text"
        processed_text = self.tp.lemmatize_text(text)
        self.assertEqual(processed_text, text)
