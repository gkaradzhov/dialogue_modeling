from toolz import itertoolz, compose
from toolz.curried import map as cmap, sliding_window, pluck
from sklearn.feature_extraction.text import TfidfVectorizer


class SkipGramVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        return lambda doc: self._word_skip_grams(
            compose(tokenize, preprocess, self.decode)(doc),
            stop_words)

    def _word_skip_grams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of 1-skip-2-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        return compose(cmap(' '.join), pluck([0, 2]), sliding_window(3))(tokens)