from dateutil.relativedelta import relativedelta
from urllib.parse import urlparse
from dateutil.parser import parse
from tld import get_tld
from tqdm import tqdm
import pandas as pd
import urllib.parse
import requests
import datetime
import pymysql
import json
import time
import string 
import random
import re
import os
import socket
import pygeoip
from langdetect import detect
from pycountry import languages
from textblob import TextBlob
from urllib.request import urlopen
import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from nltk import tokenize
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
import detoxify


# import sys
# sys.path.append("C:\\COSMOS\\ProcessScheduler")
from functions import Functions, Time

# token = "fc7ea3a02234f4589f5042bfcf9d637f"
token = "a85593889ea045d5a3a828b5df4278ce"
LOGS = []
funct = Functions()

def main():
    #Run the url crawl if True, Search by keywords if False
    articles = True 
    #GENERAL Settings
    fname = "Testing_concat_errors"
    domain = False #For crawling by domain instead of url
    force_diff = False #bypass db check 
    #-KEYWORD SETTINGS
    keywords = "covid"
    domains = "test"
    size = 20  #Total return size, will be divded up by keyword groups
    
    
    if not fname: raise ValueError("You forgot to set a file name")
    if articles and not domain:
        # urls = open_file()
        # main_time = Time()
        user_blog_records = get_records()
        data = get_article(user_blog_records, force_diff)
        # Run blogpost post processing here
        
        # main_time.finished()
        # save_excel(data, fname)
        # if LOGS: save_excel(LOGS, "Logs_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    elif articles and domain:
        domains = open_file()
        for d in domains:
            if d.count('/') > 3: raise ValueError(f"Make sure {d} is a domain, not a url, and make sure the options in main() are set properly")
            data = get_domain(d)
            if data: save_excel(data, fname)
        if LOGS: save_excel(LOGS, "Logs_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    else:
        return_data = []
        diff_bot=True
        if keywords: data = open_keywords(keywords)
        else: data = open_file()
        words = format_keywords(data)      
        for keyword in tqdm(words, desc="Keywords"):
            per_size = round(size/len(words))
            if per_size < 1: raise ValueError('''Your size of {} is too small for this keyword search.
                We have {} pairs of keywords to run. Please set a size limit at least equal to {}'''.format(size,
                len(words), len(words)))
            if any([x for x in keyword if ".com" in x or "http" in x]): 
                raise ValueError("Looks like we are supposed to read keywords from the input file, but there are urls in there...")
            domains = open_keywords(use=domains, fname="domains.json")
            return_data += get_keyword(keyword, domains, diff_bot=diff_bot, size=per_size)
        
        if LOGS: save_excel(LOGS, "Logs_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
        save_excel(return_data, fname)




def get_article(user_blog_records, force_diff=False):
    articles = []
    if force_diff: articles += [diff_get_article(x['url']) for x in tqdm(user_blog_records, desc="Articles")]
    else:
        connection = get_connection()
        for user_blog_record in tqdm(user_blog_records, desc="Articles"):
            url = user_blog_record['url']
            with connection.cursor() as cursor:
                # Update status of blogs to be crawled to -1 to avoid recrawling
                # values for processed field
                # 0 -> not processed at all, needs to be run by diffbot
                # -1 -> has been processed by diffbot and need to undergo post processing
                # with connection.cursor() as cursor:
                cursor.execute(f"""
                        UPDATE user_blog
                        SET processed = -1
                        WHERE url ="{url}"
                    """)
                connection.commit()
                cursor.execute("SELECT * FROM blogposts where permalink = %s;", url)
                record = cursor.fetchall()
                if record:
                    articles.append(record[0])
                else:
                    articles += diff_get_article(url, user_blog_record)
                    # if diff_get_article(url, user_blog_record) == "Processed
        connection.close()
    return articles

def get_keyword(keywords, domains=None, diff_bot=False, size=50):
    if diff_bot:
        return diff_get_keyword(keywords, domains=domains, size=size)
    else:
        terms = "+" + " +".join(keywords)
        connection = get_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM blogs.posts WHERE Match(content) against (%s IN BOOLEAN MODE) order by published_date desc limit %s;", (terms, size))
        connection.close()
        return cursor.fetchall()      

def get_domain(domain: str):
    total_usage = diff_check_account_balance()
    print(f"""\nWARNING: the total diffbot API usage this month is {total_usage:,}
    We will not be able to check the cost as you crawl a domain. 
    Make sure you have enough credits to complete the crawl. 
    --- Type YES to continue ---""")
    if input().lower() == "yes":
        name = urlparse(domain).netloc.replace("www.","").replace('.', '_')
        domain = DiffbotCrawl(token, name, domain, api='article')
        while True:
            time.sleep(10)
            status = domain.status()['jobs'][0]
            print(f"Processed {status['pageCrawlSuccesses']} pages")
            if status['jobStatus']['status'] == 9 or status['jobStatus']['status'] == 10: 
                print("Finished!")
                break
        data = domain.download()
        p_data = process_diff_data(data)
        domain.delete()
        return p_data
    else: return None



def process_diff_data(diff_data, record):
    articles = []
    if len(diff_data) > 5:
        pbar = tqdm(total=len(diff_data), desc="Processing Data")
    for data in diff_data:
        try: 
            if 'pageUrl' in data:
                #Cleaning
                domain = urlparse(data['pageUrl']).netloc.replace("www.","")
                if 'date' in data:
                    if 'timestamp' in data['date']:
                        try: 
                            published_date = datetime.datetime.fromtimestamp(data['date']['timestamp']/1000) 
                        except OSError:
                            published_date = None
                    else: 
                        published_date = parse(data['date'])
                else: 
                    published_date = None
                html_content = data['html'] if 'html' in data else None
                links = get_links(html_content) if html_content else None
                author = data['author'] if 'author' in data else domain
                # tags = tags_to_json([x['label'] for x in data['tags']]) if 'tags' in data else None

                tags = ','.join([x['label'] for x in data['tags']] )if 'tags' in data else None
                post_length = len(data['text'])
                outlinks = set(json.loads(links)["links"]) if (links is not None and  links!= 'null')  else None
                # num_outlinks = len(set(json.loads(links)["links"])) if (links is not None and  links!= 'null')  else 0
                num_outlinks = len(outlinks) if outlinks else 0
                num_comments = 0

                blogsite_url = urlparse(data['pageUrl']).scheme + '://' + urlparse(data['pageUrl']).netloc + '/'
                blogsite_id = get_blogsite_id(blogsite_url)

                location = get_location(data['pageUrl'])
                sentiment = get_sentiment_score(data['text'])
                language = get_language(data['text'])
                user_id = record['userid']

                #Checking for comments
                if 'discussion' in data:
                    num_comments = len(data['discussion']['posts'])

                influence_score = get_influence_score(post_length, num_outlinks, num_comments)

                #adding Post
                if 'text' in data and '<?xml' not in data['text'] and good_url(data['pageUrl']):
                    sql_query = """INSERT INTO blogposts 
                    (title, date, blogger, post, post_length, num_outlinks, num_inlinks, num_comments, comments_url, permalink, 
                    blogsite_id, tags, location, sentiment, language, influence_score, user_id) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY 
                    UPDATE title = %s, date = %s, blogger = %s, post = %s, post_length = %s, num_outlinks = %s, 
                    num_inlinks = %s, num_comments =%s, comments_url = %s, permalink = %s, blogsite_id = %s, tags = %s, 
                    location = %s, sentiment = %s, language = %s, influence_score = %s, user_id = %s, last_modified_time = CURRENT_TIMESTAMP()"""
                    sql_data = (data['title'], published_date, author, data['text'], post_length, num_outlinks, 0, num_comments,None,
                    data['pageUrl'], blogsite_id, tags, location, sentiment, language, influence_score, user_id, data['title'], 
                    published_date, author, data['text'], post_length, num_outlinks, 0, num_comments, None,
                    data['pageUrl'], blogsite_id, tags, location, sentiment, language, influence_score, user_id)

                    commit_to_db(sql_query, sql_data)
                    update_process_stat(data['pageUrl'], record['id'])
                    update_blogsite_crawled_time(blogsite_id)

                    # Post-processing
                    connection = get_connection()
                    with connection.cursor() as cursor:
                        cursor.execute(f"""
                                SELECT blogpost_id, blogsite_id, post, date
                                FROM blogposts
                                WHERE permalink = "{data['pageUrl']}";
                        """)
                        post_records = cursor.fetchall()
                    cursor.close()
                    connection.close()

                    if post_records:
                        # inserting putlinks
                        if outlinks:
                            list_outlinks = map_oultink_entry(outlinks,post_records[0]['blogsite_id'],post_records[0]['blogpost_id'], domain, published_date)
                            insert_outlink_entries(list_outlinks)
                        
                        # entity_sentiment
                        entity_sentiment_results = funct.get_entity_sentiment(post_records[0])
                        if entity_sentiment_results:
                            connection = get_connection()
                            for x in entity_sentiment_results:
                                entity, entity_type = x.text, funct.func_type(x.label_)
                                with connection.cursor() as cursor:
                                    try:
                                        query = '''INSERT INTO 
                                                    blogpost_entitysentiment 
                                                    (blogpost_id, entity, sentiment, blogsite_id, type) 
                                                    values (%s, %s, %s, %s,%s)  
                                                '''
                                        entity_blogpost_id, entity_name, entity_sentiment, entity_blogsite_id, entity_type = post_records[0]['blogpost_id'], entity, 0, post_records[0]['blogsite_id'], entity_type
                                        sql_data = (entity_blogpost_id, entity_name, entity_sentiment, entity_blogsite_id, entity_type)
                                        cursor.execute(query, sql_data)
                                        # cursor.close()
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

                        # narratives
                        process_narratives_posts(post_records[0])
                        # liwc
                        # toxicity
                        process_toxicity(post_records[0])

                        # done will all the processing
                        connection = get_connection()
                        with connection.cursor() as cursor:
                            cursor.execute(f"""
                                UPDATE user_blog
                                SET processing_status = 100, processed = 1
                                WHERE url ="{data['pageUrl']}"
                            """)
                        connection.commit()
                        cursor.close()
                        connection.close()

                        
                    
                    
                    
                    # update_valid(data['pageUrl'])
                    #Formating for return
                    # articles.append({
                    #     'domain': domain, 
                    #     'url':data['pageUrl'],
                    #     'author':author,
                    #     'title':data['title'], 
                    #     'title_sentiment':None,
                    #     'title_toxicity':None,
                    #     'published_date':published_date,
                    #     'content':data['text'],
                    #     'content_sentiment':None,
                    #     'content_toxicity':None,
                    #     'content_html':html_content,
                    #     'language':None, 
                    #     'links':links,
                    #     'tags':tags,
                    #     'crawled_time':datetime.datetime.now()
                    # })
                else:
                    update_invalid(data['pageUrl'])
            else:
                update_invalid(data['pageUrl'])
                    
            if len(diff_data) > 5:pbar.update()
        except Exception as e:
            print(e)
            update_invalid(data['pageUrl'])
            

    if len(diff_data) > 5:pbar.close()
    return "Processed"

def update_valid(url):
    connection = get_connection()
    with connection.cursor() as cursor:
        cursor.execute(f"""
            UPDATE user_blog
            SET processed = -1
            WHERE url ="{url}"
        """)
    connection.commit()
    cursor.close()
    connection.close()

def update_process_stat(url, id):
    connection = get_connection()
    with connection.cursor() as cursor:
        cursor.execute(f"""
            UPDATE user_blog
            SET processing_status = 50
            WHERE url ="{url}"
            AND id = {id}
        """)
    connection.commit()
    cursor.close()
    connection.close()  

def update_invalid(url):
    # Update to 2; because not a valid URL or no data
    connection = get_connection()
    with connection.cursor() as cursor:
        cursor.execute(f"""
            UPDATE user_blog
            SET processed = 2
            WHERE url ="{url}"
        """)
    connection.commit()
    cursor.close()
    connection.close()


def map_oultink_entry(outlinks,blogsite_id,blogpost_id,domain,pub_date):
    outlink_batch = []
    for link in outlinks :
        outlink_dict = {}
        outlink_dict['link'] = link.encode('utf-8')
        outlink_dict['blogpost_id'] = blogpost_id
        outlink_dict['blogsite_id'] = blogsite_id
        outlink_dict['domain'] = urlparse(link).netloc
        outlink_dict['date'] = pub_date.strftime("%Y-%m-%d")
        outlink_batch.append(outlink_dict)
    return outlink_batch

def insert_outlink_entries(outlinks_list):
    inserted_rows = 0 
    connection = get_connection()
    columns = ', '.join(outlinks_list[0].keys())
    params = ', '.join(['%s'] * len(outlinks_list[0]))
    link_list = [tuple(d.values()) for d in outlinks_list]
    query = ''' INSERT INTO OUTLINKS (%s) VALUES (%s) ''' %(columns,params)
    try :
        with connection.cursor() as cursor:
            try : 
                inserted_rows = cursor.executemany(query, link_list)
            except pymysql.err.IntegrityError as e:
                if e.args[0] == 1062  :
                    pass #Ignore Duplicate Entry 
                else :
                    raise Exception('{}, Error No. is {}'.format(e.args[1], e.args[0]))
            except Exception as e:
                raise Exception(e)
        connection.commit()    
    except Exception as e:
        raise Exception(e)
    finally:
        if (connection.open):
            cursor.close()
            connection.close()
    return inserted_rows

def process_narratives_posts(record):
    blogpostID = record['blogpost_id']
    post = record['post']
    ListSentences_Unique = []
    entity_narrative_dict_list = []
    countSentTotal = 0
    countSentFiltered = 0
    countSentFilteredTriplet = 0
    textSentString = ''

    stopWords = funct.get_stopwords()

    connection = get_connection()
    with connection.cursor() as cursor:
        # Declaring the entities for the blogpost
        cursor.execute(f"SELECT distinct entity from blogpost_entitysentiment where blogpost_id = {blogpostID}")
        records_entity = cursor.fetchall()
        objectEntitiesList = [x['entity'] for x in records_entity if x['entity'].lower() not in stopWords]

    connection.close()
    cursor.close()

    for everyPost in tokenize.sent_tokenize(post):
        countSentTotal = countSentTotal + 1
        everyPost = everyPost.replace("???s", "s")
        """ Clean up activity"""
        everyPost = re.sub(r"[-()\"#/@;:<>{}`'????????????+=??????_???~|!?]", " ", everyPost)
        if('on Twitter' not in everyPost and 'or e-mail to:' not in everyPost and 'd.getElementsByTagName' not in everyPost and len(everyPost)>10 and 'g.__ATA.initAd' not in everyPost and 'document.body.clientWidth' not in everyPost):
            countSentFiltered = countSentFiltered +1
            if everyPost not in ListSentences_Unique:
                ListSentences_Unique.append(everyPost)
                textSentString += str(' ') + str(everyPost)
                countSentFilteredTriplet = countSentFilteredTriplet + 1
    
    ListSentences_Unique = []       
    tfidf_string = funct.pos_tag_narratives(textSentString) #takes so much time
    result_scored = funct.run_comprehensive(tfidf_string, stopWords)
    sentences_scored = tokenize.sent_tokenize(result_scored)
    
    entity_count = []
    record['index'] = "test_narrative_entities"
    data_narratives = funct.entity_narratives(sentences_scored, record, objectEntitiesList, "elastic", entity_count)

    # if 'Duplicate entry' in self.update_insert('''INSERT INTO narratives (blogpost_id, blogsite_id, narratives, entity_count) values (%s, %s, %s, %s) ''', (blogpostID, record['blogsite_id'], json.dumps(data_narratives), json.dumps(entity_count)), self.connect):
    #     self.update_insert('''UPDATE narratives SET narratives=%s, entity_count = %s, blogsite_id = %s WHERE blogpost_id=%s;  ''', (json.dumps(data_narratives), json.dumps(entity_count), record['blogsite_id'], blogpostID), self.connect)

    return data_narratives

def process_toxicity(record):
    clean_text = funct.clean_text(record['post'])
    toxicity_result = funct.get_toxicity(clean_text, detoxify.detoxify.Detoxify('unbiased'))
    if toxicity_result:
        # for xx, post, blogpost_id, blogsite_id, result_insult, result_profanity, result_sexually_explicit, result_threat, result_identity_attack in zip(tqdm(chunk['post'], desc="Processing Toxicity Chunk", ascii=True,  file=sys.stdout), chunk['post'], chunk['blogpost_id'], chunk['blogsite_id'], toxicity_result['insult'], toxicity_result['obscene'], toxicity_result['sexual_explicit'], toxicity_result['threat'], toxicity_result['insult']):
        result_insult = round(float(toxicity_result['insult']), 6)
        result_profanity = round(float(toxicity_result['obscene']), 6)
        result_sexually_explicit = round(float(toxicity_result['sexual_explicit']), 6)
        result_threat = round(float(toxicity_result['threat']), 6)
        result_identity_attack = round(float(toxicity_result['identity_attack']), 6)

        language = funct.get_language(record['post'])
        count = 0
        if language == 'en':
            commit_to_db('''INSERT INTO liwc 
                                (blogpostid, insult, profanity, sexually_explicit, threat, identity_attack, blogsite_id) 
                                values (%s, %s, %s, %s, %s, %s, %s) 
                                ON DUPLICATE KEY UPDATE
                                insult=%s, profanity=%s, sexually_explicit=%s, threat=%s, identity_attack=%s, blogsite_id = %s''', (record["blogpost_id"], result_insult, result_profanity, result_sexually_explicit, result_threat, result_identity_attack, record['blogsite_id'], result_insult, result_profanity, result_sexually_explicit, result_threat, result_identity_attack, record['blogsite_id']))
                
        else:
            result_insult, result_profanity, result_sexually_explicit, result_threat, result_identity_attack = funct.get_toxicity_score(record['post'], len(record['post']))
            commit_to_db('''INSERT INTO liwc 
                                (blogpostid, insult, profanity, sexually_explicit, threat, identity_attack, blogsite_id) 
                                values (%s, %s, %s, %s, %s, %s, %s)
                                ON DUPLICATE KEY UPDATE
                                insult=%s, profanity=%s, sexually_explicit=%s, threat=%s, identity_attack=%s, blogsite_id = %s''', (record['blogpost_id'], result_insult, result_profanity, result_sexually_explicit, result_threat, result_identity_attack, record['blogsite_id'], result_insult, result_profanity, result_sexually_explicit, result_threat, result_identity_attack, record['blogsite_id']))

    
def get_reply_to(parent_comment, url):
    username = parent_comment['author'] if 'author' in parent_comment else None 
    date = parse(parent_comment['date'], ignoretz=True) if 'date' in parent_comment else None
    connection = get_connection()
    with connection.cursor() as cursor:
        if date: cursor.execute('''Select comment_id from comments where
                url=%s and username=%s and published_date=%s ''',(url, username, date))
        else: cursor.execute('''Select comment_id from comments where
                url=%s and username=%s and published_date is NULL and comment=%s ''',(url, username, parent_comment['text']))
        record = cursor.fetchall()
    connection.close()
    if len(record) > 1: raise IOError("Identified multiple parent comments, we only want 1. Modify the get_repy_to query. ")
    return record[0]['comment_id']


def insert_comment(comment):
    """Checks if the comment is already in database, then updates. 
        If not, generate unique key and update"""
    #Getitng id
    connection = get_connection()
    with connection.cursor() as cursor:
        if comment['published_date']: cursor.execute('''Select comment_id from comments where
                         url=%s and username=%s and published_date=%s ''', 
                         (comment['url'], comment['username'], comment['published_date']))
        else: cursor.execute('''Select comment_id from comments where
                url=%s and username=%s and published_date is NULL and comment=%s ''',
                (comment['url'], comment['username'], comment['comment']))
        record = cursor.fetchall()
        c_id = record[0]['comment_id'] if record else gen_comment_id()
    connection.close()
    #Adding to database
    sql_query = """INSERT INTO comments (domain, url, comment_id, username, comment, 
                    comment_original, links, published_date, reply_count, reply_to) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE domain=%s, username=%s, reply_count=%s,
                    crawled_time = CURRENT_TIMESTAMP()"""
    sql_data = (comment['domain'], comment['url'], c_id, comment['username'], 
                comment['comment'], comment['comment_original'], comment['links'],
                comment['published_date'], comment['reply_count'], comment['reply_to'],
                comment['domain'], comment['username'], comment['reply_count'])
    commit_to_db(sql_query, sql_data)

def diff_billing_cycle(day_renews: int=8, dnow: datetime=None,) -> int:
    """Caculates the number of days in the current diffbot billing cycle. 
    Currently, we renew on the 8th. dnow should be left as None, exposed only for testing

    Args:
        day_renews (int, optional): Day the billing cycle resets each month. Defaults to 8.
        dnow (datetime, optional): Current day. Leave as NONE, exposed only for testing

    Returns:
        int: The number of days in the current billing cycle. Can be used for queriing the diffbot accounts API. 
    """
    #Getting billing cycle timeframe (currently 8th)
    day_renews = 8
    dnow = datetime.datetime.now() if not dnow else dnow
    if dnow.day >= day_renews:
        #same month
        drenew = datetime.datetime(dnow.year, dnow.month, day_renews)
    else:
        #last month
        drenew = datetime.datetime(dnow.year, dnow.month, day_renews) - relativedelta(months=1)
    delta = dnow - drenew
    days = delta.days + 1 #+1 for inclusive days
    return days

def diff_check_account_balance():
    days = diff_billing_cycle(day_renews=8)
    r = requests.get(f"https://api.diffbot.com/v4/account?token={token}&days={days}")
    api_calls = r.json()
    total_usage = sum([x['credits'] for x in api_calls['usage']])
    percentage = (total_usage/api_calls["planCredits"])*100 if "planCredits" in api_calls else 0
    planCredits = api_calls["planCredits"] if "planCredits" in api_calls else total_usage +1
    if total_usage >= planCredits:
        raise Exception(f"""You have exceeded the API limit for this key!!
        Over the last 30 days you have used {total_usage:,} credits. Stop Now!!!""")
    elif percentage in [50,60,70,80,85,90,95,97,98,99]:
        print(f"\nCurrent Diffbot credit usage: {int(percentage)}%")
    return total_usage        

def diff_get_keyword(keywords, domains=None, size=50, page_num=0):
    terms = []
    for term in keywords['and']:
        terms.append('text:"{}"'.format(term))
    or_terms = "".join([f'"{x}", ' for x in keywords['or']]).strip(", ")
    f_terms = " ".join(terms) + " text:or(" + or_terms + ")" if len(keywords['or']) > 1 \
        else " ".join(terms)
    if len(domains) > 1: 
        site_names = "siteName:or(" + \
            "".join(['"{}", '.format(x) for x in domains]).strip(", ") + ")"
    else: site_names = 'siteName:"{}"'.format(domains[0]) if domains else None
    params = {
        "token":token,
        "type":"query",
        "size":size,
        "from":page_num,
        "query": 'type:Article {} {} sortBy:date'.format(site_names, f_terms) \
                    if site_names else 'type:Article {} sortBy:date'
    }
    diff_endpoint = "https://kg.diffbot.com/kg/dql_endpoint"
    diff_check_account_balance()
    r = requests.get(diff_endpoint, params=params)
    request = r.json()
    
    LOGS.append({"Keywords":keywords, 'domains':domains, "Hits":request['hits'],
        "request_size":size})
    diff_data = request['data']

    return process_diff_data(diff_data)

def diff_get_article(url, record, paging=True, count=0):
    #Checking for redirects
    try:
        r = requests.get(url)
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
        update_invalid(url)
        LOGS.append({"url":url, "error": "Max retries exceeded with url. Caused by SSLError(SSLError('bad handshake: SysCallError(10054, 'WSAECONNRESET')"})
        return []
    except requests.exceptions.InvalidSchema:
        update_invalid(url)
        LOGS.append({"url":url, "error": "Invalid URL Schema. Check the URL"})
        return []
    #Sending Diffbot request
    diff_check_account_balance()
    if get_tld(r.url, as_object=True).domain in url:
        diff_endpoint = "https://api.diffbot.com/v3/analyze?token={}&url={}&paging={}".format(token, urllib.parse.quote(url), paging)
        try:
            r = requests.get(diff_endpoint, timeout=60*5)
        except requests.exceptions.Timeout:
            update_invalid(url)
            LOGS.append({"url":url, "error": "The diffbot request timed out."})
            return []
        if r.status_code in (504,):
            if count > 2:
                update_invalid(url)
                LOGS.append({"url":url, "error": r.text})
                return []
            else: return diff_get_article(url, count=count+1)
        request = r.json()
    else:
        print("\nThis URL is redirecting: {}".format(url))
        update_invalid(url)
        LOGS.append({"url":url, "error":"Redirecting URL"})
        return []
    #Catching errors  
    if 'error' in request: 
        if "Automatic page concatenation exceeded timeout" in request['error']:
            if count >= 1: LOGS.append({"url":url, "error":request['error']})
            else:
                print(f"\nRetrying with paging off due to failed page concatenation: {url}")
                return diff_get_article(url, paging=False, count=count+1)
        elif '404' in request['error'] or '502' in request['error']:
            update_invalid(url)
            LOGS.append({"url":url, "error":request['error']})
        else: 
            update_invalid(url)
            print("\n{} url: {}".format(request['error'], url))
            LOGS.append({"url":url, "error":request['error']})
        return []
    #Returing Data
    diff_data = request['objects']
    return process_diff_data(diff_data, record)
    
class DiffbotClient(object):
    base_url = 'http://api.diffbot.com/'

    def request(self, url, token, api, fields=None, version=3, **kwargs):
        """
        Returns a python object containing the requested resource from the diffbot api
        """
        params = {"url": url, "token": token}
        if fields:
            params['fields'] = fields
        params.update(kwargs)
        response = requests.get(self.compose_url(api, version), params=params)
        response.raise_for_status()
        return response.json()

    def compose_url(self, api, version_number):
        """
        Returns the uri for an endpoint as a string
        """
        version = self.format_version_string(version_number)
        return '{}{}/{}'.format(self.base_url, version, api)

    @staticmethod
    def format_version_string(version_number):
        """
        Returns a string representation of the API version
        """
        return 'v{}'.format(version_number)

class DiffbotJob(DiffbotClient):
    """
    Various calls for managing a Crawlbot or Bulk API job.
    """
    def request(self,params):
        response = requests.get(self.compose_url(self.jobType,3),params=params)
        response.raise_for_status
        try:
            return response.json()
        except:
            print(response.text)

    def start(self,params):
        response = self.request(params)
        return response

    def status(self):
        response = self.request(self.params)
        return response

    def update(self,**kwargs):
        temp_params = self.params
        temp_params.update(kwargs)
        response = self.request(self.params)
        return response

    def delete(self):
        temp_params = self.params
        temp_params['delete'] = 1
        response = self.request(temp_params)
        return response

    def restart(self):
        temp_params = self.params
        temp_params['restart'] = 1
        response = self.request(temp_params)
        return response

    def download(self,data_format="json"):
        """
        downloads the JSON output of a crawl or bulk job
        """

        download_url = '{}/v3/{}/download/{}-{}_data.{}'.format(
            self.base_url,self.jobType,self.params['token'],self.params['name'],data_format
            )
        download = requests.get(download_url)
        download.raise_for_status()
        if data_format == "csv":
            return download.content
        else:
            return download.json()

class DiffbotCrawl(DiffbotJob):
    """
    Initializes a Diffbot crawl. Pass additional arguments as necessary.
    """

    def __init__(self,token,name,seeds=None,api=None,apiVersion=3,**kwargs):
        self.params = {
            "token": token,
            "name": name,
        }
        startParams = dict(self.params)
        if seeds:
            startParams['seeds'] = seeds
        if api:
            startParams['apiUrl'] = self.compose_url(api,apiVersion)
        startParams.update(kwargs)
        self.jobType = "crawl"
        self.start(startParams)

def good_url(url:str) -> bool:
    """Checks the URL to see if is a valid blog URL. URL's that could point to invalid blog pages will return False

    Args:
        url (str): The blog URL to be checked

    Returns:
        bool: True for a valid blog URL, False for invalid URL
    """
    if any([
        x for x in [
            "archive.html", "/author/","/category/", "/tag/",
            "search?updated-max", "&max-results=", "index.php?","/tagged/",
            "html?page=", "/page/", "/search/", "index.html", "/profile/",
            "?ak_action=reject_mobile", "subscribe.html", "/publications/"
        ] if x in url]): return False
    elif "archive" in url and url[-1].isdigit(): return False
    else: return True


def save_excel(data, fname, loc="export"):
    df = pd.DataFrame(data)
    path = os.path.join(os.getcwd(), loc+"\\"+fname+".xlsx")
    #Removing Timezones
    if "published_date" in df: df['published_date'] = df['published_date'].apply(lambda x: \
        datetime.datetime.replace(x,tzinfo=None) if x else None)
    df.to_excel(path, header=True, index=False, encoding='utf-8', na_rep='None', engine='xlsxwriter')
    del df
    return fname

def tags_to_json(tags):
    if tags:
        df = {'tags':tags}
        return json.dumps(df)
    else:
        return None

def links_to_json(links):
    if links: 
        df = {'links':links}
        return json.dumps(df)
    else:
        return None

def get_links(html):
    return links_to_json(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+',html.replace('};', '')))

def gen_comment_id():
    return ''.join(random.choices(string.ascii_letters.lower() + string.digits, k=16))

def format_keywords(keywords):
    results = []
    for groups in keywords:
        if "+( " in groups or " )" in groups: raise ValueError(f"Don't leave a space before or after '+( ', ' )' instead bump the word right next to it. \n{groups}")
        #Grouping
        if groups.startswith("("):
            grouped = re.findall('\[[^\]]*\]|\([^\)]*\)|\"[^\"]*\"|\S+',groups)
            grouped = [x.replace("(+","+") for x in grouped if x != ")"]
            for pos, i in enumerate(grouped):
                if i.endswith(")") and "(" not in i:
                    grouped[pos] = i.replace(")","")
        else: grouped = [groups]
        for keyword in grouped:
            #+ Keywords
            required = [x.lower().replace("+","") for x in keyword.split() if "+" in x and '(' not in x]
            or_words = [x.lower().replace("+(","").replace(")","") for x in keyword.split() if "+" not in x or "+(" in x]        
            results.append({"and":required, "or":or_words})
    return results

def open_file(fname='input.txt'):
    f = open(fname, 'r')
    data = f.read().splitlines()
    f.close()
    return data

def get_records():
    """[Getting urls from user_blog for processing]

    Returns:
        [type]: [description]
    """
    urls = []
    connection = get_connection()
    with connection.cursor() as cursor:
        # Getting urls from user_blog
        # values for processed field
        # 0 -> not processed at all, needs to be run by diffbot
        # -1 -> has been processed by diffbot and need to undergo post processing
        # 2 -> not a valid URL
        cursor.execute(f"""
                SELECT *
                FROM user_blog
                WHERE processed = 0
                AND status = "Not crawled"
                AND blogpost_id is null
                AND processed != 2;
            """)
        records = cursor.fetchall()
        # for record in records:
        #     urls.append(record['url'])

    connection.close()
    cursor.close()
    return records

def get_blogsite_id(blogsite_url):
    connection = get_connection()
    blogsite_id = 0
    with connection.cursor() as cursor:
        try:
            cursor.execute("select `blogsite_id` from `blogsites` where `blogsite_url` = %s", blogsite_url)
            blogsite_id = cursor.fetchone()['blogsite_id']
        except:
            blogsite_name = urlparse(blogsite_url).netloc
            site_type = 2
            cursor.execute("insert into `blogsites`(blogsite_url, blogsite_name, site_type) values(%s, %s, %s)", (blogsite_url, blogsite_name, site_type))
            cursor.execute("select `blogsite_id` from `blogsites` where `blogsite_url` = %s", blogsite_url)
            blogsite_id = cursor.fetchone()['blogsite_id']
            connection.commit()
        finally:
            if (connection.open):
                cursor.close()
                connection.close()
    return blogsite_id

# get ip address
def get_ip_addr(url):

    domain = urlparse(url).netloc
    try:
        ip_addr = socket.gethostbyname(domain)
    except:
        ip_addr = 'unknown'

    return ip_addr

# get geo-location
def get_location(url):
    try:
        ip_addr = get_ip_addr(url)
        geolib = pygeoip.GeoIP('GeoIP.dat')
        location = geolib.country_code_by_addr(ip_addr)
    except:
        location = None

    return location

def get_language(post_content):
    try:
        lang_detect = detect(post_content)
        post_lang = languages.get(alpha_2=lang_detect).name
    except:
        post_lang = 'unknown'
    return post_lang

def update_blogsite_crawled_time(blogsite_id):
    connection = get_connection()
    with connection.cursor() as cursor:
        try:
            cursor.execute("update blogtrackers.blogsites set last_crawled = CURRENT_TIMESTAMP() where `blogsite_id` = %s", (blogsite_id))
            connection.commit()
        except Exception as e:
            raise Exception('Blogsite Update error')
        finally:
            if (connection.open):
                cursor.close()
                connection.close()

# get influence score
def get_influence_score(post_length, num_outlinks, comment_count):

    length_factor = 0
    num_inlinks = 0

    if (post_length <= 350):
        length_factor = 1
    else:
        length_factor = 2 if post_length <= 1500 else 3

    scaling_factor = length_factor
    outlinks_normalized = num_outlinks / length_factor
    influence_flow = (0.6 * comment_count) + (0.7 * num_inlinks) - (0.3 * outlinks_normalized)
    influence_score = scaling_factor * influence_flow
    influence_score_rounded = round(influence_score,5)

    return influence_score_rounded

# Get location from url
def get_location(url):
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

# Cleaning text before sentiment / toxicity
def clean_text(text):
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
def get_sentiment_score(text):
    """[summary]

    Args:
        text ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Cleaning the text
    text = clean_text(text)
    if not text:
        return 0  # Returning empty strings
    # Getting Score
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    sentiment_score_rounded = round(sentiment_score, 6)
    return sentiment_score_rounded

def open_keywords(use=None, fname='keywords.json',):
    with open(fname, encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not use: return data
    else:
        try: return data[use]
        except KeyError: raise KeyError("You chose a project name that is not in {}: {}".format(fname, use))

def get_connection():
    connection = pymysql.connect(host='cosmos-1.host.ualr.edu',
                                user='ukraine_user',
                                password='summer2014',
                                db='blogtrackers',
                                charset='utf8mb4',
                                use_unicode=True,
                                cursorclass=pymysql.cursors.DictCursor)
    return connection

def commit_to_db(query, data, error=0):
    # while True: 
    try:
        connection = get_connection()
        with connection.cursor() as cursor:
            cursor.execute(query, data)
        connection.commit()
        connection.close()
        return 
    #Error handeling
    except Exception as e:
        if isinstance(e, pymysql.err.IntegrityError) and e.args[0]==1062:
            # Duplicate Entry, already in DB
            # print("There is already duplicate entry in the DB, check the quary: {}".format(query))
            connection.close() 
            return
        elif e.args[0] == 1406:
            # Data too long for column
            print(e)
            print("Good API request, but data is Too Long for DB Column")
            connection.close()
            return 
        elif e.args[0] == 2013:
            if error < 10:
                commit_to_db(query, data, error+1)
                connection.close()
            else:
                raise Exception("Keep loosing connection to the db: {}".format(e))
        else: 
            # Uncaught errors
            raise Exception("We aren't catching this mySql commit_to_db Error: {}".format(e))



# if __name__ == "__main__":
    # diff_data = json.load(open('test.json', 'r'))
    # record = {
    #     "url":"https://www.torontosun.com/news/local-news/toronto-schools-closed-to-in-class-learning",
    #     "userid":"btrackerdemo@gmail.com",
    #     "id":26
    # }
    # process_diff_data(diff_data,record)

    # main()
run_async = True
if __name__ == "__main__":
    if run_async:
        # executors = {
        #     'default': ThreadPoolExecutor(60),
        #     'processpool': ProcessPoolExecutor(60)
        # }

        job_defaults = {
            'coalesce': False,
            'max_instances': 20
        }
        scheduler = BackgroundScheduler(job_defaults = job_defaults)
        # scheduler = BackgroundScheduler()
        scheduler.add_job(func=main, trigger="interval", seconds=1)
        scheduler.start()

        # Shut down the scheduler when exiting the app
        end_script = input("Please input 'Yes' to stop the script or leave the processing running")
        if end_script.lower() == "yes":
            atexit.register(lambda: scheduler.shutdown())
    
        print('here')
    else:
        main()