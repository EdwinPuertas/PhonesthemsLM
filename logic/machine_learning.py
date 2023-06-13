import sys
import time
import warnings
from tqdm import tqdm
from math import log
from sklearn.metrics import log_loss
import numpy as np
from numpy import asarray
import pandas as pd
from scipy.stats import kruskal, pearsonr, ttest_ind
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from collections import Counter
from logic.feature_extraction import FeatureExtraction
from logic.text_analysis import TextAnalysis
from logic.utils import Utils
warnings.filterwarnings("ignore")


class MachineLearning(object):

    def __init__(self, lang='es', text_analysis=None):
        try:
            print('Load Machine Learning')
            if text_analysis is None:
                self.ta = TextAnalysis(lang=lang)
            else:
                self.ta = text_analysis
            self.features = FeatureExtraction(lang=lang, text_analysis=self.ta)
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error MachineLearning: {0}'.format(e))

    # calculate cross entropy
    @staticmethod
    def cross_entropy(p, q, ets=1e-15):
        return -sum([p[i] * log(q[i] + ets) for i in range(len(p))])

    @staticmethod
    def model_name(model_type, binary_vad):

        dict_type = {'W': int(model_type[0]), 'S': int(model_type[1]),
                     'FP': int(model_type[2]), 'OP': int(model_type[3]),
                     'AP': int(model_type[4])}

        model_name = '+'.join([k for k, v in dict_type.items() if v == 1 and len(model_type) > 0])

        result = {'model_name': model_name + '@' + binary_vad,
                  'word': int(model_type[0]),
                  'syllable': int(model_type[1]),
                  'freq_phoneme': int(model_type[2]),
                  'one_phoneme': int(model_type[3]),
                  'all_phoneme': int(model_type[4]),
                  'valence': int(binary_vad[0]),
                  'arousal': int(binary_vad[1]),
                  'dominance': int(binary_vad[2]),
                  'sum_vad': int(binary_vad[3])}
        return result

    def Data_Pearson_Embeddings(self, model_type, x):
        #Word              0
        #Syllable          1
        #Freq Phoneme      2
        #One Phoneme       3
        #All Phoneme       4  
        try:
            
            corr_w_s = []
            corr_w_fp = []
            corr_w_of = []
            corr_w_af = []
            corr_s_fp = []
            corr_s_of = []
            corr_s_af = []
            corr_fp_of = []
            corr_fp_af = []
            corr_of_af = []       
            
            #Word features Shape (1350, 300)    W
            #Syllables features Shape (1350, 150) S
            #Phoneme features Shape (1350, 30) FP
            #One Syllable features Shape (1350, 150) OF
            #All Syllable features Shape (1350, 150) AF
            if model_type == '11111':
                x_w = x[:, 0:300]
                x_s = x[:,300:450]
                x_fp= x[:,450:480]
                x_of= x[:,480:630]
                x_af= x[:,630:780]
                
            if model_type == '01111':
                x_s = x[:,0:150]
                x_fp= x[:,150:180]
                x_of= x[:,180:330]
                x_af= x[:,330:480]
                
            if model_type == '10111':
                x_w = x[:,0:300]
                x_fp= x[:,300:330]
                x_of= x[:,330:480]
                x_af= x[:,480:630]
                
            if model_type == '00111':
                x_fp= x[:,0:30]
                x_of= x[:,30:180]
                x_af= x[:,180:330]
                
            if model_type == '11011':
                x_w = x[:,0:300]
                x_s = x[:,300:450]
                x_of= x[:,450:600]
                x_af= x[:,600:750]
                
            if model_type == '01011':
                x_s = x[:,0:150]
                x_of= x[:,150:300]
                x_af= x[:,300:450]
                
            if model_type == '10011':
                x_w = x[:,0:300]
                x_of= x[:,300:450]
                x_af= x[:,450:600]
                
            if model_type == '00011':
                x_of= x[:,0:150]
                x_af= x[:,150:300]
                
            if model_type == '11101':
                x_w = x[:,0:300]
                x_s = x[:,300:450]
                x_fp= x[:,450:480]
                x_af= x[:,480:630]
                
            if model_type == '01101':
                x_s = x[:,0:150]
                x_fp= x[:,150:180]
                x_af= x[:,180:330]
                
            if model_type == '10101':
                x_w = x[:,0:300]
                x_fp= x[:,300:330]
                x_af= x[:,330:480]
                
            if model_type == '00101':
                x_fp= x[:,0:30]
                x_af= x[:,30:180]
                
            if model_type == '11001':#Data
                x_w = x[:,0:300] 
                x_s = x[:,300:450]
                x_af= x[:,450:600]
                
            if model_type == '01001':
                x_s = x[:,0:150]
                x_af= x[:,150:300]
                
            if model_type == '10001':
                x_w = x[:,0:300]
                x_af= x[:,300:450]
                
            #if model_type == '00001':
            #    x_af=
                
            if model_type == '11110':
                x_w = x[:,0:300]  
                x_s = x[:,300:450]
                x_fp= x[:,450:480]
                x_of= x[:,480:630]
                
            if model_type == '01110':
                x_s = x[:,0:150]
                x_fp= x[:,150:180]
                x_of= x[:,180:330]
                
            if model_type == '10110':
                x_w = x[:,0:300]
                x_fp= x[:,300:330]
                x_of= x[:,330:480]
                
            if model_type == '00110':
                x_fp= x[:,0:30]
                x_of= x[:,30:180]
                
            if model_type == '11010':
                x_w = x[:,0:300]
                x_s = x[:,300:450]
                x_of= x[:,450:600]
                
            if model_type == '01010':
                x_s = x[:,0:150]
                x_of= x[:,150:300]
                
            if model_type == '10010':
                x_w = x[:,0:300]
                x_of= x[:,300:450]
                
            #if model_type == '00010':
            #    x_of=
                
            if model_type == '11100':
                x_w = x[:,0:300]
                x_s = x[:,300:450]
                x_fp= x[:,450:480]
                
            if model_type == '01100':
                x_s = x[:,0:150]
                x_fp= x[:,150:180]
                
            if model_type == '10100':
                x_w = x[:,0:300]
                x_fp= x[:,300:330]
                
            #if model_type == '00100':
            #    x_fp=
                
            if model_type == '11000':
                x_w = x[:,0:300]
                x_s = x[:,300:450]
                
            #if model_type == '01000':
            #    x_s =
                
            #if model_type == '10000':
            #    x_w = 
                
                
            #0 - W - 300
            #1 - S - 150
            #2 - FP - 30
            #3 - OF - 150
            #4 - AF - 150
            
            
            #WORD
            
            
            if int(model_type[0]) == 1 & int(model_type[1]) == 1:
                
                
                x_s_f = np.append(x_s, x_s, axis=1)
                corr_w_s, _ = np.abs(pearsonr(x_w.flatten('F'), x_s_f.flatten('F')))
                
            if int(model_type[0]) == 1 & int(model_type[2]) == 1:
                
                x_fp_f = x_fp
                for i in range(9):
                    x_fp_f= np.append(x_fp_f, x_fp, axis=1)
                               
                corr_w_fp, _ = np.abs(pearsonr(x_w.flatten('F'), x_fp_f.flatten('F')))
                
                
            if int(model_type[0]) == 1 & int(model_type[3]) == 1:
                
                
                x_of_f = np.append(x_of, x_of, axis=1)
                corr_w_of, _ = np.abs(pearsonr(x_w.flatten('F'), x_of_f.flatten('F')))
                
                
            if int(model_type[0]) == 1 & int(model_type[4]) == 1:
                
                x_af_f = np.append(x_af, x_af, axis=1)
                corr_w_af, _ = np.abs(pearsonr(x_w.flatten('F'), x_af_f.flatten('F')))
            
            
            
            #SYLLABLE
            if int(model_type[1]) == 1 & int(model_type[2]) == 1:
                x_fp_f = x_fp
                for i in range(4):
                    x_fp_f= np.append(x_fp_f, x_fp, axis=1)
            
                corr_s_fp, _ = np.abs(pearsonr(x_s.flatten('F'), x_fp_f.flatten('F')))
                
            if int(model_type[1]) == 1 & int(model_type[3]) == 1:
                
                corr_s_of, _ = np.abs(pearsonr(x_s.flatten('F'), x_of.flatten('F')))
                
            if int(model_type[1]) == 1 & int(model_type[4]) == 1:
                
                corr_s_af, _ = np.abs(pearsonr(x_s.flatten('F'), x_af.flatten('F')))
            
            
            #PHONEME
            if int(model_type[2]) == 1 & int(model_type[3]) == 1:
                x_fp_f = x_fp
                for i in range(4):
                    x_fp_f= np.append(x_fp_f, x_fp, axis=1)
                    
                corr_fp_of, _ = np.abs(pearsonr(x_fp_f.flatten('F'), x_of.flatten('F')))
                
            if int(model_type[2]) == 1 & int(model_type[4]) == 1:
                x_fp_f = x_fp
                for i in range(4):
                    x_fp_f= np.append(x_fp_f, x_fp, axis=1)
                corr_fp_af, _ = np.abs(pearsonr(x_fp_f.flatten('F'), x_af.flatten('F')))
                
            #One Syllable     
            if int(model_type[3]) == 1 & int(model_type[4]) == 1:
                
                corr_of_af, _ = np.abs(pearsonr(x_of.flatten('F'), x_af.flatten('F')))
            
            return corr_w_s, corr_w_fp, corr_w_of, corr_w_af, corr_s_fp, corr_s_of, corr_s_af, corr_fp_of, corr_fp_af, corr_of_af
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error replica: {0}'.format(e))
            return None    
    
    def replica(self, dict_model, classifier_name, classifier, fold, rep, target,
                x, y, x_test, y_test, sample_train, sample_test, model_type):
        
        try:
            dict_model.update({'classifier': classifier_name,
                               'sample_train': sample_train,
                               'sample_test': sample_test})
            k_fold = StratifiedShuffleSplit(n_splits=fold, test_size=0.30, random_state=42)
            start_time = time.time()

            accuracies_scores = []
            recalls_scores = []
            precisions_scores = []
            f1_scores = []
            ll_score = []
            ce_score = []
            stat_kw_score = []
            p_kw_score = []
            #New Scores
            pearson_scores=[]
            studentst_Abs_scores=[]
            studentst_scores=[]
            
            corr_w_s_scores =[]
            corr_w_fp_scores =[]
            corr_w_of_scores =[]
            corr_w_af_scores =[]
            corr_s_fp_scores =[]
            corr_s_of_scores =[]
            corr_s_af_scores =[]
            corr_fp_of_scores =[]
            corr_fp_af_scores =[]
            corr_of_af_scores =[]
            
            
            corr_w_s_test_scores =[]
            corr_w_fp_test_scores =[]
            corr_w_of_test_scores =[]
            corr_w_af_test_scores =[]
            corr_s_fp_test_scores =[]
            corr_s_of_test_scores =[]
            corr_s_af_test_scores =[]
            corr_fp_of_test_scores =[]
            corr_fp_af_test_scores =[]
            corr_of_af_test_scores =[]
            
            
            clf = classifier
            for train_index, test_index in k_fold.split(x, y):
                data_train = x[train_index]
                target_train = y[train_index]

                data_test = x[test_index]
                target_test = y[test_index]

                clf.fit(data_train, target_train)
                predict = classifier.predict(data_test)
                # Accuracy
                accuracy = accuracy_score(target_test, predict)
                accuracies_scores.append(accuracy)
                # Recall
                recall = recall_score(target_test, predict, average='macro')
                recalls_scores.append(recall)
                # Precision
                precision = precision_score(target_test, predict, average='weighted')
                precisions_scores.append(precision)
                # F1
                f1 = f1_score(target_test, predict, average='weighted')
                f1_scores.append(f1)

                # Log Loss function
                # prepare classification data
                probability = classifier.predict_proba(data_test)
                y_true = asarray(target_test)
                y_pred = np.nan_to_num(asarray(probability))

                # calculate the average log loss
                ll = log_loss(y_true, y_pred)
                ll_score.append(ll)

                # cross-entropy for predicted probability distribution
                ents = np.nan_to_num([self.cross_entropy(target, d) for d in probability])
                ce = abs(np.mean(ents))
                ce_score.append(ce)

                # Kruskal-Wallis H Test
                stat_kw, p_kw = kruskal(y_true, predict)
                stat_kw_score.append(stat_kw)
                p_kw_score.append(p_kw)
                
                #Pearson
                pearson_corr , _ = np.abs(pearsonr(target_test, predict))
                pearson_scores.append(pearson_corr)
                
                #Student's T test Abs
                StudentTtest_stat_Abs , _ = np.abs(ttest_ind(target_test, predict))
                studentst_Abs_scores.append(StudentTtest_stat_Abs)
                
                #Student's T test 
                StudentTtest_stat , _ = ttest_ind(target_test, predict)
                studentst_scores.append(StudentTtest_stat)
                
                #Corr embeddings
                       
                if (accuracy*100)>60:
                #Data Train
                    corr_w_s, corr_w_fp, corr_w_of, corr_w_af, corr_s_fp, corr_s_of, corr_s_af, corr_fp_of, corr_fp_af, corr_of_af = self.Data_Pearson_Embeddings(model_type=model_type, x=data_train)
                
                    corr_w_s_scores.append(corr_w_s)
                    corr_w_fp_scores.append(corr_w_fp)
                    corr_w_of_scores.append(corr_w_of)
                    corr_w_af_scores.append(corr_w_af)
                    corr_s_fp_scores.append(corr_s_fp)
                    corr_s_of_scores.append(corr_s_of)
                    corr_s_af_scores.append(corr_s_af)
                    corr_fp_of_scores.append(corr_fp_of)
                    corr_fp_af_scores.append(corr_fp_af)
                    corr_of_af_scores.append(corr_of_af)
                
                #Data Test
                    corr_w_s_test, corr_w_fp_test, corr_w_of_test, corr_w_af_test, corr_s_fp_test, corr_s_of_test, corr_s_af_test, corr_fp_of_test, corr_fp_af_test, corr_of_af_test = self.Data_Pearson_Embeddings(model_type=model_type, x=data_test)
                
                    corr_w_s_test_scores.append(corr_w_s_test)
                    corr_w_fp_test_scores.append(corr_w_fp_test)
                    corr_w_of_test_scores.append(corr_w_of_test)
                    corr_w_af_test_scores.append(corr_w_af_test)
                    corr_s_fp_test_scores.append(corr_s_fp_test)
                    corr_s_of_test_scores.append(corr_s_of_test)
                    corr_s_af_test_scores.append(corr_s_af_test)
                    corr_fp_of_test_scores.append(corr_fp_of_test)
                    corr_fp_af_test_scores.append(corr_fp_af_test)
                    corr_of_af_test_scores.append(corr_of_af_test)
                

            average_recall = round(np.mean(recalls_scores) * 100, 2)
            dict_model['recall'] = average_recall

            average_precision = round(np.mean(precisions_scores) * 100, 2)
            dict_model['precision'] = average_precision

            average_f1 = round(np.mean(f1_scores) * 100, 2)
            dict_model['f1'] = average_f1

            average_accuracy = round(np.mean(accuracies_scores) * 100, 2)
            dict_model['accuracy'] = average_accuracy
            #dict_model['accuracy'] = PP

            # calculate the average cross entropy
            mean_ll = round(float(np.mean(ll_score)), 2)
            dict_model['log_loss'] = mean_ll

            mean_ce = round(float(np.mean(ce_score)), 2)
            dict_model['cross_entropy'] = mean_ce

            mean_p_kw = round(float(np.mean(p_kw_score)), 2)
            dict_model['kruskal_wallis'] = mean_p_kw
            
            #New Metrics
            average_pearson = round(np.mean(pearson_scores), 2)
            dict_model['pearson'] = average_pearson
            
            average_students = round(np.mean(studentst_scores), 2)
            dict_model['students t-test'] = average_students
            
            average_students_abs = round(np.mean(studentst_Abs_scores), 2)
            dict_model['|students t-test|'] = average_students_abs

            
            #Metric Pearson Corr
            average_corr_w_s = round(np.mean(corr_w_s_scores), 2)
            dict_model['corr w-s'] = average_corr_w_s
            
            average_corr_w_fp = round(np.mean(corr_w_fp_scores), 2)
            dict_model['corr w-fp'] = average_corr_w_fp
            
            average_corr_w_of = round(np.mean(corr_w_of_scores), 2)
            dict_model['corr w-of'] = average_corr_w_of
            
            average_corr_w_af = round(np.mean(corr_w_af_scores), 2)
            dict_model['corr w-af'] = average_corr_w_af
            
            average_corr_s_fp = round(np.mean(corr_s_fp_scores), 2)
            dict_model['corr s-fp'] = average_corr_s_fp
            
            average_corr_s_of = round(np.mean(corr_s_of_scores), 2)
            dict_model['corr s-of'] = average_corr_s_of
            
            average_corr_s_af = round(np.mean(corr_s_af_scores), 2)
            dict_model['corr s-af'] = average_corr_s_af
            
            average_corr_fp_of = round(np.mean(corr_fp_of_scores), 2)
            dict_model['corr fp-of'] = average_corr_fp_of
            
            average_corr_fp_af = round(np.mean(corr_fp_af_scores), 2)
            dict_model['corr fp-af'] = average_corr_fp_af
            
            average_corr_of_af = round(np.mean(corr_of_af_scores), 2)
            dict_model['corr of-af'] = average_corr_of_af
            
            #Test 

            average_corr_w_s_test = round(np.mean(corr_w_s_test_scores), 2)
            dict_model['corr w-s test'] = average_corr_w_s_test
            
            average_corr_w_fp_test = round(np.mean(corr_w_fp_test_scores), 2)
            dict_model['corr w-fp test'] = average_corr_w_fp_test
            
            average_corr_w_of_test = round(np.mean(corr_w_of_test_scores), 2)
            dict_model['corr w-of test'] = average_corr_w_of_test
            
            average_corr_w_af_test = round(np.mean(corr_w_af_test_scores), 2)
            dict_model['corr w-af test'] = average_corr_w_af_test
            
            average_corr_s_fp_test = round(np.mean(corr_s_fp_test_scores), 2)
            dict_model['corr s-fp test'] = average_corr_s_fp_test
            
            average_corr_s_of_test = round(np.mean(corr_s_of_test_scores), 2)
            dict_model['corr s-of test'] = average_corr_s_of_test
            
            average_corr_s_af_test = round(np.mean(corr_s_af_test_scores), 2)
            dict_model['corr s-af test'] = average_corr_s_af_test
            
            average_corr_fp_of_test = round(np.mean(corr_fp_of_test_scores), 2)
            dict_model['corr fp-of test'] = average_corr_fp_of_test
            
            average_corr_fp_af_test = round(np.mean(corr_fp_af_test_scores), 2)
            dict_model['corr fp-af test'] = average_corr_fp_af_test
            
            average_corr_of_af_test = round(np.mean(corr_of_af_test_scores), 2)
            dict_model['corr of-af test'] = average_corr_of_af_test
            
            y_predict = []
            for features in x_test:
                features = features.reshape(1, -1)
                value = clf.predict(features)[0]
                y_predict.append(value)

            classification = classification_report(y_test, y_predict)
            dict_model['classification'] = classification
            confusion = confusion_matrix(y_predict, y_test)
            dict_model['confusion'] = confusion

            dict_model['predict_model'] = clf

            # Calculated Time processing
            t_sec = round(time.time() - start_time)
            (t_min, t_sec) = divmod(t_sec, 60)
            (t_hour, t_min) = divmod(t_min, 60)
            time_processing = '{} hour:{} min:{} sec'.format(t_hour, t_min, t_sec)
            #Error name here
            dict_model['time_processing'] = time_processing

            #PRINT TEST TRAINING OBJECT
            #print(np.shape(x))
            #print(x.get_feature_phoneme)
            
            # print result
            print('{0} | Begin {1} - Replica #{2} | {0}'.format("#" * 12, classifier_name, rep))
            output_result = {'F1-score': average_f1, 'Accuracy': average_accuracy,
                             'Recall': average_recall, 'Precision': average_precision,
                             'Log Loss': mean_ll, 'Cross Entropy': mean_ce,
                             'Kruskal - Wallis': mean_p_kw, 'pearson':average_pearson, 'students t-test' :average_students,
                             '|students t-test|': average_students_abs , 'corr w-s':average_corr_w_s, 'corr w-fp':average_corr_w_fp, 'corr w-of':average_corr_w_of, 'corr w-af':average_corr_w_af, 'corr s-fp':average_corr_s_fp, 'corr s-of':average_corr_s_of, 'corr s-af':average_corr_s_af, 'corr fp-of':average_corr_fp_of, 'corr fp-af':average_corr_fp_af, 'corr of-af':average_corr_of_af, 'corr w-s test':average_corr_w_s_test ,'corr w-fp test':average_corr_w_fp_test, 'corr w-of test' :average_corr_w_of_test,'corr w-af test':average_corr_w_af_test, 'corr s-fp test':average_corr_s_fp_test, 'corr s-of test':average_corr_s_of_test, 'corr s-af test':average_corr_s_af_test, 'corr fp-of test':average_corr_fp_of_test, 'corr fp-af test':average_corr_fp_af_test, 'corr of-af test':average_corr_of_af_test,'Time Processing': time_processing,
                             'Classification Report': classification, 'Confusion Matrix\n': confusion}
            for item, val in output_result.items():
                print('{0}: {1}'.format(item, val))
            print('{0} | End {1} - Replica #{2} | {0}'.format("#" * 12, classifier_name, rep))
            return dict_model
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error replica: {0}'.format(e))
            return None

    def train(self, model_type='11111', classifier_name=None, classifier=None, train_data=None, test_data=None,
              binary_vad='0000', over_sampler=True, iteration=10, fold=10, target=None):
        try:
            dict_model = self.model_name(model_type=model_type, binary_vad=binary_vad)
            if train_data is not None:
                print("#" * 15 + '| Start Model: ' + dict_model['model_name'] + ' |' + "#" * 15)
                print('Training {0} ....'.format(classifier_name))

                train_data = pd.DataFrame(train_data)
                x = train_data['message'].tolist()
                y = train_data['valence'].to_numpy()

                test_data = pd.DataFrame(test_data)
                x_test = test_data['message'].tolist()
                y_test = test_data['valence'].to_numpy()

                print('***Get training features')
                x = self.features.get_features(x, model_type=model_type, binary_vad=binary_vad)
                
                x = preprocessing.normalize(x)

                print('***Get testing features')
                x_test = self.features.get_features(x_test, model_type=model_type, binary_vad=binary_vad)
                x_test = preprocessing.normalize(x_test)

                # Calculated Over Sample
                print('**Sample train:', sorted(Counter(y).items()))
                print('**Sample test:', sorted(Counter(y_test).items()))
                sample_train = 'Sample train:' + str(sorted(Counter(y).items())) + '\n'
                sample_test = 'Sample test:' + str(sorted(Counter(y_test).items())) + '\n'

                if over_sampler:
                    ros_train = RandomOverSampler(random_state=1000)
                    x, y = ros_train.fit_resample(x, y)
                    print('**RandomOverSampler train:', sorted(Counter(y).items()))
                    sample_train += 'RandomOverSampler train:' + str(sorted(Counter(y).items()))
                    # test
                    ros_test = RandomOverSampler(random_state=1000)
                    x_test, y_test = ros_test.fit_resample(x_test, y_test)
                    print('**RandomOverSampler test:', sorted(Counter(y_test).items()))
                    sample_test += 'RandomOverSampler test:' + str(sorted(Counter(y_test).items()))
                else:
                    ros_train = RandomUnderSampler(random_state=1000)
                    x, y = ros_train.fit_resample(x, y)
                    print('**RandomUnderSampler train:', sorted(Counter(y).items()))
                    sample_train += 'RandomUnderSampler train:' + str(sorted(Counter(y).items()))
                    # test
                    ros_test = RandomOverSampler(random_state=1000)
                    x_test, y_test = ros_test.fit_resample(x_test, y_test)
                    print('**RandomUnderSampler test:', sorted(Counter(y_test).items()))
                    sample_test += 'RandomUnderSampler test:' + str(sorted(Counter(y_test).items()))

                result = []
                for i in range(1, iteration+1):
                    rep = int(i)
                    out_put = {'replica': rep}
                    data_dict = self.replica(dict_model=dict_model, classifier_name=classifier_name,
                                             classifier=classifier, fold=fold, target=target, x=x, y=y,
                                             x_test=x_test, y_test=y_test, sample_train=sample_train,
                                             sample_test=sample_test, rep=rep, model_type=model_type)
                    out_put.update(data_dict)
                    result.append(out_put)
                print("#" * 15 + '| End Model: ' + dict_model['model_name'] + ' |' + "#" * 15)
                return result
            else:
                print('ERROR without dataset')
        except Exception as e:
            Utils.standard_error(sys.exc_info())
            print('Error train: {0}'.format(e))
            return None
### esto es una prueba


