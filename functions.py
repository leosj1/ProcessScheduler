import hashlib
import dateparser
import os
import re
import time
import pandas as pd
from langdetect import detect
from textblob import TextBlob
# from mtranslate import translate
from urllib.parse import urlparse
from googleapiclient.discovery import build
import googleapiclient
import json
import requests
import pycountry
from gensim.parsing.preprocessing import strip_tags, strip_punctuation, strip_short
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_numeric, stem_text
from gensim.parsing.preprocessing import strip_non_alphanum, remove_stopwords, preprocess_string
from collections import Counter
import pycountry
from datetime import datetime
import spacy
import socket
from urllib.request import urlopen
import logging
import configparser
import math
import nltk
from nltk import word_tokenize, PorterStemmer
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
import detoxify
from typing import Tuple, Union
from Elastic_s import Es
# from Util.sql import SqlFuncs
tokenizer = RegexpTokenizer(r'\w+')

class Functions(Es):
    """[Functions class contains all the functions for the post-processing scripts]

    Args:
        Es ([class]): [Elasticsarch class]

    Returns:
        [type]: [description]
    """
    def get_stopwords(self):
        stop_words = []
        stop_words = ['è', 'abbã³l', 'acaba', 'acerca', 'aderton', 'ahimã', 'ain', 'akã', 'alapjã', 'alors', 'alã', 'alã³l', 'alã³la', 'alã³lad', 'alã³lam', 'alã³latok', 'alã³luk', 'alã³lunk', 'amã', 'annã', 'appendix', 'arrã³l', 'attã³l', 'azokbã³l', 'azokkã', 'azoknã', 'azokrã³l', 'azoktã³l', 'azokã', 'aztã', 'azzã', 'azã', 'ba', 'bahasa', 'bb', 'bban', 'bbi', 'bbszã', 'belã', 'belã¼l', 'belå', 'bennã¼k', 'bennã¼nk', 'bã', 'bãºcsãº', 'cioã', 'cittã', 'ciã²', 'conjunctions', 'cosã', 'couldn', 'csupã', 'daren', 'didn', 'dik', 'diket', 'doesn', 'don', 'dovrã', 'ebbå', 'effects', 'egyedã¼l', 'egyelå', 'egymã', 'egyã', 'egyã¼tt', 'egã', 'ek', 'ellenã', 'elså', 'elã', 'elå', 'ennã', 'enyã', 'ernst', 'errå', 'ettå', 'ezekbå', 'ezekkã', 'ezeknã', 'ezekrå', 'ezektå', 'ezekã', 'ezentãºl', 'ezutã', 'ezzã', 'ezã', 'felã', 'forsûke', 'fã', 'fûr', 'fûrst', 'ged', 'gen', 'gis', 'giã', 'gjûre', 'gre', 'gtã', 'gy', 'gyet', 'gã', 'gã³ta', 'gã¼l', 'gã¼le', 'gã¼led', 'gã¼lem', 'gã¼letek', 'gã¼lã¼k', 'gã¼lã¼nk', 'hadn', 'hallã³', 'hasn', 'haven', 'herse', 'himse', 'hiã', 'hozzã', 'hurrã', 'hã', 'hãºsz', 'idã', 'ig', 'igazã', 'immã', 'indonesia', 'inkã', 'insermi', 'ismã', 'isn', 'juk', 'jã', 'jã³', 'jã³l', 'jã³lesik', 'jã³val', 'jã¼k', 'kbe', 'kben', 'kbå', 'ket', 'kettå', 'kevã', 'khã', 'kibå', 'kikbå', 'kikkã', 'kiknã', 'kikrå', 'kiktå', 'kikã', 'kinã', 'kirå', 'kitå', 'kivã', 'kiã', 'kkel', 'knek', 'knã', 'korã', 'kre', 'krå', 'ktå', 'kã', 'kã¼lã', 'lad', 'lam', 'latok', 'ldã', 'led', 'leg', 'legalã', 'lehetå', 'lem', 'lennã', 'leszã¼nk', 'letek', 'lettã¼nk', 'ljen', 'lkã¼l', 'll', 'lnak', 'ltal', 'ltalã', 'luk', 'lunk', 'lã', 'lã¼k', 'lã¼nk', 'magã', 'manapsã', 'mayn', 'megcsinã', 'mellettã¼k', 'mellettã¼nk', 'mellã', 'mellå', 'mibå', 'mightn', 'mikbå', 'mikkã', 'miknã', 'mikrå', 'miktå', 'mikã', 'mindenã¼tt', 'minã', 'mirå', 'mitå', 'mivã', 'miã', 'modal',
                    'mostanã', 'mustn', 'myse', 'mã', 'mãºltkor', 'mãºlva', 'må', 'måte', 'nak', 'nbe', 'nben', 'nbã', 'nbå', 'needn', 'nek', 'nekã¼nk', 'nemrã', 'nhetå', 'nhã', 'nk', 'nnek', 'nnel', 'nnã', 'nre', 'nrå', 'nt', 'ntå', 'nyleg', 'nyszor', 'nã', 'nå', 'når', 'også', 'ordnung', 'oughtn', 'particles', 'pen', 'perchã', 'perciã²', 'perã²', 'pest', 'piã¹', 'puã²', 'pã', 'quelqu', 'qué', 'ra', 'rcsak', 'rem', 'retrieval', 'rlek', 'rmat', 'rmilyen', 'rom', 'rt', 'rte', 'rted', 'rtem', 'rtetek', 'rtã¼k', 'rtã¼nk', 'rã', 'rã³la', 'rã³lad', 'rã³lam', 'rã³latok', 'rã³luk', 'rã³lunk', 'rã¼l', 'sarã', 'schluss', 'semmisã', 'shan', 'shouldn', 'sik', 'sikat', 'snap', 'sodik', 'sodszor', 'sokat', 'sokã', 'sorban', 'sorã', 'sra', 'st', 'stb', 'stemming', 'study', 'sz', 'szen', 'szerintã¼k', 'szerintã¼nk', 'szã', 'sã', 'talã', 'ted', 'tegnapelå', 'tehã', 'tek', 'tessã', 'tha', 'tizenhã', 'tizenkettå', 'tizenkã', 'tizennã', 'tizenã', 'tok', 'tovã', 'tszer', 'tt', 'tte', 'tted', 'ttem', 'ttetek', 'ttã¼k', 'ttã¼nk', 'tulsã³', 'tven', 'tã', 'tãºl', 'tå', 'ul', 'utoljã', 'utolsã³', 'utã', 'vben', 'vek', 'velã¼k', 'velã¼nk', 'verbs', 'ves', 'vesen', 'veskedjã', 'viszlã', 'viszontlã', 'volnã', 'vvel', 'vã', 'vå', 'vöre', 'vört', 'wahr', 'wasn', 'weren', 'won', 'wouldn', 'zadik', 'zat', 'zben', 'zel', 'zepesen', 'zepã', 'zã', 'zã¼l', 'zå', 'ã³ta', 'ãºgy', 'ãºgyis', 'ãºgynevezett', 'ãºjra', 'ãºr', 'ð¾da', 'γα', 'البت', 'بالای', 'برابر', 'برای', 'بیرون', 'تول', 'توی', 'تی', 'جلوی', 'حدود', 'خارج', 'دنبال', 'روی', 'زیر', 'سری', 'سمت', 'سوی', 'طبق', 'عقب', 'عل', 'عنوان', 'قصد', 'لطفا', 'مد', 'نزد', 'نزدیک', 'وسط', 'پاعین', 'کنار', 'अपन', 'अभ', 'इत', 'इनक', 'इसक', 'इसम', 'उनक', 'उसक', 'एव', 'ऐस', 'करत', 'करन', 'कह', 'कहत', 'गय', 'जह', 'तन', 'तर', 'दब', 'दर', 'धर', 'नस', 'नह', 'पहल', 'बन', 'बह', 'यत', 'यद', 'रख', 'रह', 'लक', 'वर', 'वग़', 'सकत', 'सबस', 'सभ', 'सर', 'ἀλλ']
        # change this to full path of stop words file
        with open(r"C:\stopwords.txt", "r", encoding="utf-8") as f:
            for line in f:
                stop_words.append(str(line.strip()))

        return stop_words

    # def load_nlp_model(self):
        #run "python -m spacy download en_core_web_lg" for first time run
    nlp = spacy.load('en_core_web_lg') 
    nlp.max_length = 20966640

    # return nlp
    


    def get_config(self):
        """[Gets database configuration]

        Returns:
            [tuple]: [The ip of ther server, username, password and name of the database]
        """
        config = configparser.ConfigParser()
        config.read(r"C:\config.ini")

        DB_MOVER=config["DB_MOVER"]
        ip = DB_MOVER["HOST"]
        user_name = DB_MOVER["USER"] 
        password =  DB_MOVER["PASS"]
        db = DB_MOVER["DB"]

        return ip, user_name, password, db

    def get_config2(self, config_type):
        """[Gets database configuration]

        Returns:
            [tuple]: [The ip of ther server, username, password and name of the database]
        """
        config = configparser.ConfigParser()
        config.read(r"C:\config.ini")

        DB_MOVER=config[config_type]
        ip = DB_MOVER["HOST"]
        user_name = DB_MOVER["USER"] 
        password =  DB_MOVER["PASS"]
        db = DB_MOVER["DB"]

        return ip, user_name, password, db
        

    def get_language(self, text):
        """[detect language of text]

        Args:
            text ([string]): [text to detect language]

        Returns:
            [string]: [language]
        """
        try:
            post_lang = detect(text)
        except:
            post_lang = 'N/A'
        return post_lang

    def convert_to_json(self, string):
        """[Converts string to json]

        Args:
            string ([string]): [string in json format]

        Returns:
            [json]: [key-value json data type]
        """
        return json.dumps(string)

    def get_full_language(self, language):
        """[Get full language e.g 'EN' --> 'English']

        Args:
            language ([string]): [language in short form e.g 'EN']

        Returns:
            [string]: [language in full form  e.g English']
        """
        if language:
            language = pycountry.languages.get(alpha_2=language)
            if language:
                language = language.name
                return language.title()

    # Cleaning text before sentiment / toxicity
    def clean_text(self, text) -> Union[str, None]:
        """Cleans string for processing. Removes bytes, emails, and urls

        Args:
            text ([str]): text to clean

        Returns:
            Union[str, None]: Returns cleaned string, or None if no text remains after cleaning
        """
        if text and ''.join(text.split()):
            if type(text) == bytes: #Decoding byte strings
                text = text.decode('utf-8')
            #Removing emails + ***.com urls
            text = ' '.join([item for item in text.split() if '@' not in item and '.com' not in item])
            text = ' '.join(text.split()) #removing all multiple spaces
            if text: return text
        # UNCLEAN_TEXT.inc()
        return None

    # get sentiment score
    def get_sentiment_score(self, text):
        """[summary]

        Args:
            text ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Cleaning the text
        text = self.clean_text(text)
        if not text:
            return 0  # Returning empty strings
        # Getting Score
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        sentiment_score_rounded = round(sentiment_score, 6)
        return sentiment_score_rounded

    # get toxicity score
    def get_toxicity_score(self, text, length):
        """[summary]

        Args:
            text ([type]): [description]
            length ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Cleaning the text
        text = self.clean_text(text)
        if not text:
            return 0, 0, 0, 0, 0  # Returning empty strings
        if len(text.encode('utf-8')) > 20480: 
            return 0, 0, 0, 0, 0  #str limit for toxicity
        # Sending Request
        API_KEY = 'AIzaSyCdOGjynFqrd5A-gkKKeYjqs0UIMP7FGjc'
        service = build('commentanalyzer', 'v1alpha1', developerKey=API_KEY)
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {'INSULT': {}, 'PROFANITY': {}, 'SEXUALLY_EXPLICIT': {}, 'THREAT': {}, 'IDENTITY_ATTACK': {}}}
        try:
            probability = service.comments().analyze(body=analyze_request).execute()

            result_insult = probability['attributeScores']['INSULT']['summaryScore']['value']
            result_profanity = probability['attributeScores']['PROFANITY']['summaryScore']['value']
            result_sexually_explicit = probability['attributeScores']['SEXUALLY_EXPLICIT']['summaryScore']['value']
            result_threat = probability['attributeScores']['THREAT']['summaryScore']['value']
            result_identity_attack = probability['attributeScores']['IDENTITY_ATTACK']['summaryScore']['value']
            return result_insult, result_profanity, result_sexually_explicit, result_threat, result_identity_attack

        # Catching Exceptions
        except googleapiclient.errors.HttpError as e:
            if 'language' in e.content.decode('utf-8'):
                return 0, 0, 0, 0, 0
            elif "Comment text too long" in e.content.decode('utf-8'):
                # Note that for blogs, lots of content is too long, so I turned off the message
                # print("Message to long for toxiciity to process!")
                return None, None, None, None, None
            elif "Comment text was too many bytes" in e.content.decode('utf-8'):
                # length-=100
                # return self.get_toxicity_score(text[:length], length)
                return None, None, None, None, None
            else:
                # print("Uncaught Http error on Toxicity Score: {}".format(e))
                return None, None, None, None, None
        except Exception as e:
            print("Uncaught error on Toxicity Score: {}".format(e))
            return None, None, None, None, None

    def get_toxicity(self, text, detoxify: detoxify.detoxify.Detoxify) ->dict:
        """Generates toxicity scores using Detoxify model
        https://github.com/unitaryai/detoxify

        Args:
            text (str): Text to use
            detoxify (detoxify.detoxify.Detoxify): Detoxify model

        Returns:
            [dict]: Key of score name (toxicity), value of score. 
        """
        if text:
            results = detoxify.predict(text)
            #Rounding the results
            return results

    # Get entities
    def get_entity_sentiment(self, record):
        """[summary]

        Args:
            record ([type]): [description]

        Returns:
            [type]: [description]
        """
        content = record['post']
        nlp_model = self.nlp
        doc = nlp_model(content)

        entities = doc.ents
        
        # result = list(map(lambda x: (x.text, func_type(x.label_)), entities))
        # logging.basicConfig(filename=f'New_Logs//Entity_Sentiment_Logs//entity_sentiment{str(time.localtime()[0]) + "-" + str(time.localtime()[1]) + "-" + str(time.localtime()[2])}.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
        # data = json.dumps({"blogpost_id": record['blogpost_id'], "permalink":record['permalink']})
        # if entities:
            
        #     logging.debug(f"{data} processed...")
        # else:
        #     logging.debug(f"{data} has empty entities...")
        return entities
        # pass

    # Clean text
    def remove(self, text):
        """[summary]

        Args:
            text ([type]): [description]

        Returns:
            [type]: [description]
        """
        # if text:
        CUSTOM_FILTERS = [lambda x: x.lower(),  # lowercase
                          strip_multiple_whitespaces,
                          strip_non_alphanum,
                          strip_numeric,
                          remove_stopwords,
                          strip_short,
                          #                   stem_text
                          ]
        text = text.lower()
        example_sent = preprocess_string(text, CUSTOM_FILTERS)
        filtered_sentence = [
            w for w in example_sent if not w in self.get_stopwords()]

        return filtered_sentence

    # Get top terms
    def counter(self, text):
        """[summary]

        Args:
            text ([type]): [description]

        Returns:
            [type]: [description]
        """
        if text:
            st = self.remove(text)
            if st:
                terms_dict_final = []
                counter_obj = Counter(st)
                most_common_100 = counter_obj.most_common(100)
                most_common_1 = counter_obj.most_common(1)
                for term, occurrence in most_common_100:
                    terms_dict_final.append({"term":term,"occurrence":occurrence})
                return str(most_common_100), str(most_common_1[0][0]), dict(counter_obj), str(most_common_1[0][0]), terms_dict_final
            else:
                return None, None, None, None, None
        else:
            return None, None, None, None, None

    # Get location from url
    def get_location(self, url):
        """[summary]

        Args:
            url ([type]): [description]

        Returns:
            [type]: [description]
        """
        domain = urlparse(url).netloc

        ip = ''
        try:
            ip = socket.gethostbyname(domain) if domain else None
        except:
            pass
        url = 'http://ipinfo.io/' + ip  if ip else None
        response = urlopen(url) if url else None
        data = json.load(response) if response else None
        location = data['country'] if data else None

        return location

    def load_indices(self):
        indices = [
                        {
                            "index": "blogposts",
                            "column": "blogpost_id",
                            "key": "blogpost_id"
                        }
                        ,
                        {
                            "index": "liwc",
                            "column": "blogpostid",
                            "key": "blogpostid"
                        }
                        ,
                        {
                            "index": "outlinks",
                            "column": "blogpost_id",
                            "key": "outlink_id"
                        }
                        ,
                        {
                            "index": "blogpost_entitysentiment",
                            "column": "blogpost_id",
                            "key": "id"
                        }
                        ,
                        {
                            "index": "entity_narratives",
                            "column": "blogpost_id",
                            "key": "blogpost_id"
                        }
                    ]
        return indices

    def func_type(self, x):
        return x.replace('ORG', 'ORGANIZATION').replace('LOC', 'LOCATION').replace('GPE', 'COUNTRY').replace('NORP', 'NATIONALITY').replace('GPE', 'COUNTRY')

    """ pos_tag_narratives accept sentences from blogpost and with the help of grammar rules, extract VerbPhrases, NounPhrases, and Triplets from each sentence """
    def pos_tag_narratives(self,textSentString):
        token = word_tokenize(textSentString)
        tags = nltk.pos_tag(token)
        grammar = r"""
        NP: {<DT|JJ|NN.*>+}
            {<IN>?<NN.*>}
        VP: {<TO>?<VB.*>+<IN>?<RB.*>?}
        CLAUSE: {<CD>?<NP><VP>+<NP>?<TO>?<NP>?<IN>?<NP>?<VP>?<NP>?<TO>?<NP>+}
        """
        a = nltk.RegexpParser(grammar)
        result = a.parse(tags)
        tfidf_string = ''
        for a in result:
            if type(a) is nltk.Tree:
                str1= ''
                if a.label() == 'CLAUSE':
                # This climbs into your NVN tree
                    for b in a:
                        if(isinstance(b, tuple)):
                            #print(b[0])
                            str1 += str(b[0])+ ' '
                        else:
                            for elem in b:
                                str1 += str(elem[0])+ ' '
                            #print(b.leaves()) # This outputs your "NP"
                    str1 = str1.strip() + str('.') + str(' ')
                    tfidf_string += str1
        return tfidf_string   

    """ This method will comprehensively do all the required things for us. First accept sentences and
        does Frequency matrix operation, TF, IDF, TF-IDF, scoring sentences """
    def run_comprehensive(self, text, stop_words):
        # 1 Sentence Tokenize
        sentences = tokenize.sent_tokenize(text)
        #print(sentences)
        total_documents = len(sentences)
        #print(total_documents)
        #print(sentences)

        # 2 Create the Frequency matrix of the words in each sentence.
        freq_matrix = self._create_frequency_matrix(sentences, stop_words)
        #print(freq_matrix)

        # 3 Calculate TermFrequency and generate a matrix
        tf_matrix = self._create_tf_matrix(freq_matrix)
        #print(tf_matrix)

        # 4 creating table for documents per words
        count_doc_per_words = self._create_documents_per_words(freq_matrix)
        #print(count_doc_per_words)

        # 5 Calculate IDF and generate a matrix
        idf_matrix = self._create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
        #print(idf_matrix)

        # 6 Calculate TF-IDF and generate a matrix
        tf_idf_matrix = self._create_tf_idf_matrix(tf_matrix, idf_matrix)
        #print(tf_idf_matrix)

        # 7 Important Algorithm: score the sentences
        sentence_scores = self._score_sentences(tf_idf_matrix)
        #print(sentence_scores)

        # 8 Find the threshold
        threshold = self._find_average_score(sentence_scores)
        
        #z_value = _find_z_score(sentence_scores)
        #print(z_value)
        
        # 9 Important Algorithm: Generate the narratives
        narratives = self._generate_narratives(sentences, sentence_scores, threshold)
        #narratives_z = _generate_narratives(sentences, sentence_scores, z_value)
        return narratives

    """ _create_frequency_matrix creates a matrix of sentences for a given blogpost """
    def _create_frequency_matrix(self, sentences, stop_words):
        frequency_matrix = {}
        # stopWords = set(stopwords.words("english"))
        

        ps = PorterStemmer()
        for sent in sentences:
            freq_table = {}
            words = word_tokenize(sent)
            for word in words:
                word = word.lower()
                word = ps.stem(word)
                if word in stop_words:
                    continue
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
            #frequency_matrix[sent[:15]] = freq_table
            frequency_matrix[sent] = freq_table
        return frequency_matrix

    
    """ Using freq_matrix creates a TermFreq Matrix """
    def _create_tf_matrix(self, freq_matrix):
        tf_matrix = {}
        for sent, f_table in freq_matrix.items():
            tf_table = {}
            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence
            tf_matrix[sent] = tf_table
        return tf_matrix

    """ Using freq_matrix created words for document """
    def _create_documents_per_words(self, freq_matrix):
        word_per_doc_table = {}
        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1
        return word_per_doc_table

    """ this method creates Inverse Document Freq matrix """
    def _create_idf_matrix(self, freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}
        for sent, f_table in freq_matrix.items():
            idf_table = {}
            for word in f_table.keys():
                #Considering DF only
                idf_table[word] = math.log10(float(count_doc_per_words[word])/total_documents)
            idf_matrix[sent] = idf_table
        return idf_matrix

    def _create_tf_idf_matrix(self, tf_matrix, idf_matrix):
        tf_idf_matrix = {}
        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
            tf_idf_table = {}
            for (word1, value1), (word2, value2) in zip(f_table1.items(),f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)
            tf_idf_matrix[sent1] = tf_idf_table
        return tf_idf_matrix

    def _score_sentences(self, tf_idf_matrix):
        sentenceValue = {}
        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0
            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score
            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
        return sentenceValue

    def _find_average_score(self, sentenceValue):
        """
        Find the average score from the sentence value dictionary
        :rtype: int
        """
        sumValues = 0
        for entry in sentenceValue:    
            sumValues += sentenceValue[entry]
        
        try:
            average = (sumValues / len(sentenceValue))
        except:
            average = 0
        return average

    def _generate_narratives(self, sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''
        dict_sent = {}
        for sentence in sentences:
            #if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            if sentence in sentenceValue:
                dict_sent[sentence] = sentenceValue[sentence]
        sentences_sorted_values = sorted(dict_sent, key=dict_sent.get, reverse=True)
        count =0
        for r in sentences_sorted_values:
            #This is tuning parameter to display only top sentences.
            #print(r)
            if(count ==100):
                break
            count = count + 1
            #print(r, dict_sent[r])
            summary += str(r) + " "
            sentence_count += 1
        return summary

    def entity_narratives(self, sentences_scoredList, record, objectEntitiesList, storage_type, entity_count = []):
        entity_narratives_dict = {}
        actions = []
        for narr in sentences_scoredList:
            for entity in objectEntitiesList:
                temp_array = [" " + entity.lower() + " ", entity.lower() + " ", " " + entity.lower()] 
                s = set()
                for temp in temp_array:
                    if temp in narr.lower() and len(entity) > 1 and temp.strip() not in s:
                        s.add(temp.strip())
                        json_body = {
                            "_index": record['index'],
                            "_id": self.hash_function(str(record['blogpost_id']) + '_' + entity + '_' + narr),
                            "_source": {
                                "blogpost_id": record['blogpost_id'],
                                "blogsite_id": record['blogsite_id'],
                                "narrative": narr,
                                "entity": entity,
                                "date": record['date'],
                                "narrative_keyword": narr
                            }
                        }
                        actions.append(json_body)
                        
        if actions:
            client = self.get_client("144.167.35.89")
            bulk_action = self.bulk_request(client, actions)
            client.transport.close()
            if bulk_action[0] != len(actions):
                print('here')

        # return entity_narratives_dict
        return len(actions)

    def hash_function(self, text):
        hash_object = hashlib.md5(text.encode())
        return hash_object.hexdigest()


class Time():
    def __init__(self):
        self.start = time.time()
        self.end = None
        self.runtime_mins = None
        self.runtime_secs = None

    def finished(self):
        self.end = time.time()
        self.runtime_mins, self.runtime_secs = divmod(
            self.end - self.start, 60)
        self.runtime_mins = round(self.runtime_mins, 0)
        self.runtime_secs = round(self.runtime_secs, 0)
        print("Time to complete: {} Mins {} Secs".format(self.runtime_mins,
                                                         self.runtime_secs), f'AT - {datetime.today().isoformat()}')
        return "Time to complete: {} Mins {} Secs".format(self.runtime_mins,
                                                         self.runtime_secs), f'AT - {datetime.today().isoformat()}'


# x = Functions().get_location("https://balkaniumblog.wordpress.com/2016/12/03/re-emergence-of-the-balkans-independence-from-the-ottoman-empire/")