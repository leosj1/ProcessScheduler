# store last sql blogpost_id value
# Check if data exists after that id
# Update tables based on this data
# from TableUpdates.blogsites import main
from tqdm import tqdm
import sys
import pycountry
from nltk import tokenize
import re
from sql import SqlFuncs
from functions import Functions, Time

class ProcessSchedule(Functions, SqlFuncs):
    def __init__(self, connect):
        self.connect = connect

    def update_trigger_table(self):
        """[Function to update blogposts_trigger table]
        """
        for record in self.records:
            blogpost_id = record['blogpost_id']
            self.update_item('''
                    UPDATE blogposts_triggers 
                    SET updated=%s
                    WHERE blogpost_id = %s''', (1, blogpost_id), self.connect)

    def get_data_after_last_sql(self):
        """[Getting newly inserted records from blogposts table from trigger (blogposts_triggers table)]
        """
        self.BLOGGER = ""
        self.BLOGPOST_IDS = ""

        records = []
        # sql_func = SqlFuncs(self.connect)
        connection = self.get_connection(self.connect)

        with connection.cursor() as cursor:
            cursor.execute(f"""
                    select * from blogposts_triggers bt, blogposts bs
                    WHERE bt.blogpost_id = bs.blogpost_id
                    AND bt.updated = 0;
                """)
            records = cursor.fetchall()
        connection.close()
        cursor.close()

        self.records = records
        self.BLOGGER = "\"" + '\",\"'.join(list(set([record['blogger'] for record in records]))) + "\""
        self.BLOGPOST_IDS = ','.join([str(record['blogpost_id']) for record in records])
        self.BLOGSITE_IDS = ','.join([str(record['blogsite_id']) for record in records])
        self.process = True if records else False
        print('done getting sql')

    def process_bloggers(self):
        """[Function to update blogger table]
        """
        # sql_func = SqlFuncs(self.connect)
        connection = self.get_connection(self.connect)
        # if self.process:
        with connection.cursor() as cursor:
            cursor.execute(f"""
                    SELECT *
                    FROM liwc limit 1000
                    """)

            records = cursor.fetchall()

            for record in tqdm(records, desc="Blogger", ascii=True,  file=sys.stdout):
                self.update_item('''UPDATE liwc SET analytic=%s WHERE blogpostid = %s ; ''', (record['analytic'], record['blogpostid']), self.connect)
                # return record

        connection.close()
        cursor.close()

        print('done processing bloggers')
        return True

    def process_entity_sentiments(self):
        """[Function to update blogpost_entitysentiment table]
        """
        if self.process:
            for record in self.records:
                if record['post']:
                    res = self.get_entity_sentiment(record)
                    if res:
                        connection = self.get_connection(self.connect)
                        for x in res:
                            entity, entity_type = x.text, self.func_type(x.label_)
                            with connection.cursor() as cursor:
                                try:
                                    query = '''INSERT INTO 
                                                blogpost_entitysentiment 
                                                (blogpost_id, entity, sentiment, blogsite_id, type) 
                                                values (%s, %s, %s, %s,%s)  
                                            '''
                                    entity_blogpost_id, entity_name, entity_sentiment, entity_blogsite_id, entity_type = record['blogpost_id'], entity, 0, record['blogsite_id'], entity_type
                                    data = (entity_blogpost_id, entity_name, entity_sentiment, entity_blogsite_id, entity_type)
                                    cursor.execute(query, data)
                                    cursor.close()
                                    connection.commit()
                                except Exception as e:
                                    if 'Duplicate entry' in str(e):
                                        cursor.close()
                                        connection.close()
                                        return record
                                    elif 'Data too long for column' in str(e):
                                        cursor.close()
                                        connection.close()
                                        return record
                                    else:
                                        cursor.close()
                                        connection.close()
                                        return record
                        
                        connection.close()
                        return record

    def process_posts(self):
        """[Function to update blogpost_terms and blogpost_terms_api tables]
        """
        for record in self.records:
            if record['post']:
                terms, topterm, terms_, topterm_, final_dict = self.counter(record['post'])
                if not terms:
                    terms, topterm = "BLANK", "BLANK"
                if not terms_:
                    terms_, topterm_ = self.convert_to_json("{'NO KEYWORD': 1}"), self.convert_to_json("{'NO KEYWORD': 1}")

                if not record['num_comments']:
                    record['num_comments'] = 0

                if 'Duplicate entry' in self.update_item('''INSERT INTO blogpost_terms (blogpost_id, blogsiteid, terms, topterm, date, blogger, post, title, num_comments) 
                values (%s, %s, %s, %s,%s, %s, %s, %s, %s) ''', 
                (record['blogpost_id'], record['blogsite_id'], terms, topterm, record['date'], record['blogger'], record['post'], record['title'], record['num_comments']), self.connect):
                    self.update_item('''UPDATE blogpost_terms
                    SET terms=%s, topterm=%s
                    WHERE blogpost_id=%s;  ''', (terms, topterm, record['blogpost_id']), self.connect)

                if 'Duplicate entry' in self.update_item('''INSERT INTO blogpost_terms_api (blogpost_id, blogsiteid, terms, topterm, date, blogger, post, title, num_comments, terms_test) 
                values (%s, %s, %s, %s,%s, %s, %s, %s, %s, %s) ''', 
                (record['blogpost_id'], record['blogsite_id'], self.convert_to_json(terms_), topterm_, record['date'], record['blogger'], record['post'], record['title'], record['num_comments'], self.convert_to_json(final_dict)), self.connect):
                    self.update_item('''
                    UPDATE blogpost_terms_api
                    SET terms=%s, topterm=%s, terms_test=%s
                    WHERE blogpost_id=%s;  
                    ''', (self.convert_to_json(terms_), topterm_, self.convert_to_json(final_dict), record['blogpost_id']), self.connect)

    def process_narratives(self):
        self.process_entity_sentiments()
        stop_words = []
        with open(r"C:\BT-PostProcessing\stopwords.txt", "r", encoding="utf-8") as f:
            for line in f:
                if line != '':
                    stop_words.append(str(line.strip()))
                
        new_stp_wrds = []
        
        final_stp_wrds = stop_words + new_stp_wrds
        stopWords = final_stp_wrds

        for record in self.records:
            blogpostID = record['blogpost_id']
            post = record['post']
            ListSentences_Unique = []
            entity_narrative_dict_list = []
            countSentTotal = 0
            countSentFiltered = 0
            countSentFilteredTriplet = 0
            textSentString = ''

            connection = self.get_connection(self.connect)
            with connection.cursor() as cursor:
                # Declaring the entities for the blogpost
                cursor.execute(f"SELECT distinct entity from blogpost_entitysentiment where blogpost_id = {blogpostID}")
                records_entity = cursor.fetchall()
                objectEntitiesList = [x['entity'] for x in records_entity if x['entity'].lower() not in stopWords]

            connection.close()
            cursor.close()

            for everyPost in tokenize.sent_tokenize(post):
                countSentTotal = countSentTotal + 1
                everyPost = everyPost.replace("’s", "s")
                """ Clean up activity"""
                everyPost = re.sub(r"[-()\"#/@;:<>{}`'’‘“”+=–—_…~|!?]", " ", everyPost)
                if('on Twitter' not in everyPost and 'or e-mail to:' not in everyPost and 'd.getElementsByTagName' not in everyPost and len(everyPost)>10 and 'g.__ATA.initAd' not in everyPost and 'document.body.clientWidth' not in everyPost):
                    countSentFiltered = countSentFiltered +1
                    if everyPost not in ListSentences_Unique:
                        ListSentences_Unique.append(everyPost)
                        textSentString += str(' ') + str(everyPost)
                        countSentFilteredTriplet = countSentFilteredTriplet + 1
            
            ListSentences_Unique = []       
            tfidf_string = self.pos_tag_narratives(textSentString) #takes so much time
            result_scored = self.run_comprehensive(tfidf_string, stopWords)
            sentences_scored = tokenize.sent_tokenize(result_scored)
            
            entity_count = []
            record['index'] = "entity_narr_trigger_test"
            data_narratives = self.entity_narratives(sentences_scored, record, objectEntitiesList, "elastic", entity_count)
            return data_narratives

def main():
    connection_credentials = Functions().get_config2("BLOGTRACKERS")
    # connection_credentials = Functions().get_config2("DB_MOVER")

    ps = ProcessSchedule(connection_credentials)
    # ps.get_data_after_last_sql()

    ps.process_bloggers()

    # if ps.process_bloggers():
        # ps.update_trigger_table()
    # ps.process_posts()
    # ps.process_blogsites()
    # ps.process_languages()

    # entity_sentiment has to run before narratives
    # # # ps.process_entity_sentiments()
    # ps.process_narratives()

    # 

    print('seun is testing')
    

    # query = f"""
    #     SELECT * 
    #     FROM blogposts 
    #     WHERE blogpost_id in ({BLOGPOST_IDS})
    # """

import sched, time
import time
import atexit

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

run_async = False
if __name__ == "__main__":
    # s = sched.scheduler(time.time, time.sleep)

    # s.enter(60, 1, main, (s,))

    if run_async:

        executors = {
            'default': ThreadPoolExecutor(60),
            'processpool': ProcessPoolExecutor(60)
       }

        job_defaults = {
            'coalesce': False,
            'max_instances': 50
        }
        scheduler = BackgroundScheduler(job_defaults = job_defaults)
        # scheduler = BackgroundScheduler()
        scheduler.add_job(func=main, trigger="interval", seconds=3)
        scheduler.start()

        # Shut down the scheduler when exiting the app
        atexit.register(lambda: scheduler.shutdown())
    
        print('here')
    else:
        main()

    
