import sys
import epitran
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from logic.text_analysis import TextAnalysis
from logic.utils import Utils
from root import DIR_INPUT, DIR_MODELS


class FeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, lang='es', text_analysis=None):
        try:
            if text_analysis is None:
                self.ta = TextAnalysis(lang=lang)
            else:
                self.ta = text_analysis
            file_word_embedding_en = DIR_MODELS + 'word_embedding_en.model'
            file_word_embedding_es = DIR_MODELS + 'word_embedding_es.model'
            file_phonestheme_embedding_en = DIR_MODELS + 'phonestheme_embedding_en.model'
            file_phonestheme_embedding_es = DIR_MODELS + 'phonestheme_embedding_es.model'
            file_phoneme_embedding_en = DIR_MODELS + 'phoneme_embedding_en.model'
            file_phoneme_embedding_es = DIR_MODELS + 'phoneme_embedding_es.model'
            if lang == 'es':
                epi = epitran.Epitran('spa-Latn')
                word_embedding = Word2Vec.load(file_word_embedding_es)
                phonestheme_embedding = Word2Vec.load(file_phonestheme_embedding_es)
                phoneme_embedding = Word2Vec.load(file_phoneme_embedding_es)
            else:
                epi = epitran.Epitran('eng-Latn')
                word_embedding = Word2Vec.load(file_word_embedding_en)
                phonestheme_embedding = Word2Vec.load(file_phonestheme_embedding_en)
                phoneme_embedding = Word2Vec.load(file_phoneme_embedding_en)

            self.epi = epi
            self.word= word_embedding
            self.phonestheme = phonestheme_embedding
            self.phoneme = phoneme_embedding
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error FeatureExtraction: {0}'.format(e))

    def fit(self, x, y=None):
        return self

    def transform(self, list_messages):
        try:
            result = self.get_features(list_messages)
            return result
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error transform: {0}'.format(e))

    def get_feature_word(self, messages):
        try:
            counter = 0
            model = self.word
            num_features = model.vector_size
            index_to_key_set = set(model.wv.index_to_key)
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                num_words = 1
                feature_vec = []
                list_words = [token['text'] for token in self.ta.tagger(msg)]
                for word in list_words:
                    if word in index_to_key_set:
                        vec = model.wv[word]
                        feature_vec.append(vec)
                    else:
                        feature_vec.append(np.zeros(num_features, dtype="float32"))
                    num_words += 1
                feature_vec = np.array(feature_vec, dtype="float32")
                feature_vec = np.sum(feature_vec, axis=0)
                feature_vec = np.divide(feature_vec, num_words)
                msg_feature_vec[counter] = feature_vec
                counter = counter + 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_word: {0}'.format(e))
            return None

    def get_feature_phonestheme(self, messages, syllable_binary='11'):
        try:
            counter = 0
            model = self.phonestheme
            num_features = model.vector_size
            index2phoneme_set = set(model.wv.index_to_key)
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                num_phonemes = 1
                feature_vec = []
                # print('Msg: {0}'.format(msg))
                list_syllable = [token['syllables'] for token in self.ta.tagger(msg) if token['syllables'] is not None]
                for syllable in list_syllable:
                    for s in syllable:
                        syllable_phonetic = self.epi.transliterate(s, normpunc=True)
                        if syllable_phonetic in index2phoneme_set:
                            vec = model.wv[syllable_phonetic]
                            feature_vec.append(vec)
                            num_phonemes += 1
                feature_vec = np.array(feature_vec, dtype="float32")
                feature_vec = np.sum(feature_vec, axis=0)
                feature_vec = np.divide(feature_vec, num_phonemes)
                msg_feature_vec[counter] = feature_vec
                counter += 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_syllable: {0}'.format(e))
            return None

    def get_frequency_phoneme(self, messages):
        try:
            counter = 0
            model = self.phoneme
            index2phoneme = list(model.wv.index_to_key)
            num_features = len(index2phoneme)
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                # print('Msg: {0}'.format(msg))
                feature_vec = np.zeros(num_features, dtype="float32")
                list_syllable = [token['syllables'] for token in self.ta.tagger(msg) if token['syllables'] is not None]
                for syllable in list_syllable:
                    for s in syllable:
                        syllable_phonetic = self.epi.transliterate(s, normpunc=True)
                        if syllable_phonetic in index2phoneme:
                            index = index2phoneme.index(syllable_phonetic)
                            value = feature_vec[index]
                            feature_vec[index] = value + 1
                msg_feature_vec[counter] = feature_vec
                counter += 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_frequency_phoneme: {0}'.format(e))
            return None

    def get_feature_phoneme(self, messages, syllable=False):
        try:
            counter = 0
            model = self.phoneme
            num_features = model.vector_size
            index2phoneme_set = set(model.wv.index_to_key)
            msg_feature_vec = np.zeros((len(messages), num_features), dtype="float32")
            for msg in tqdm(messages):
                size = 1
                feature_vec = []
                list_syllable = [token['syllables'] for token in self.ta.tagger(msg) if token['syllables'] is not None]
                if syllable:
                    try:
                        first_syllable = str(list_syllable[0][0])
                        first_syllable = first_syllable[0] \
                            if (first_syllable is not None) and (len(first_syllable) > 0) else ''
                        syllable_phonetic = self.epi.transliterate(first_syllable)
                        if syllable_phonetic in index2phoneme_set:
                            vec = model.wv[syllable_phonetic]
                            feature_vec.append(vec)
                        else:
                            feature_vec.append(np.zeros(num_features, dtype="float32"))
                    except Exception as e_epi:
                        print('Error transliterate: {0}'.format(e_epi))
                        pass
                else:
                    list_phoneme = self.epi.trans_list(msg)
                    size = len(list_phoneme)
                    for phoneme in list_phoneme:
                        if phoneme in index2phoneme_set:
                            vec = model.wv[phoneme]
                            feature_vec.append(vec)
                        else:
                            feature_vec.append(np.zeros(num_features, dtype="float32"))
                # print('Vector: {0}'.format(feature_vec))
                feature_vec = np.array(feature_vec, dtype="float32")
                feature_vec = np.sum(feature_vec, axis=0)
                feature_vec = np.divide(feature_vec, size)
                msg_feature_vec[counter] = feature_vec
                counter += 1
            return msg_feature_vec
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error get_feature_phoneme: {0}'.format(e))
            return None