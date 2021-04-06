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

token = "fc7ea3a02234f4589f5042bfcf9d637f"
LOGS = []

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
        urls = open_file()
        data = get_article(urls, force_diff)
        save_excel(data, fname)
        if LOGS: save_excel(LOGS, "Logs_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
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




def get_article(urls, force_diff=False):
    articles = []
    if force_diff: articles += [diff_get_article(x) for x in tqdm(urls, desc="Articles")]
    else:
        connection = get_connection()
        for url in tqdm(urls, desc="Articles"):
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM posts where url = %s;", url)
                record = cursor.fetchall()
                if record:
                    articles.append(record[0])
                else:
                    articles += diff_get_article(url)
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



def process_diff_data(diff_data):
    articles = []
    if len(diff_data) > 5:
        pbar = tqdm(total=len(diff_data), desc="Processing Data")
    for data in diff_data: 
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
            author = data['author'] if 'author' in data else None
            tags = tags_to_json([x['label'] for x in data['tags']]) if 'tags' in data else None
            #adding Post
            if 'text' in data and '<?xml' not in data['text'] and good_url(data['pageUrl']):
                sql_query = """INSERT INTO posts (domain, url, author, title, published_date, content, content_html, links, tags) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE domain = %s, author = %s, title = %s, 
                        published_date = %s, content = %s, content_html = %s, links = %s, tags = %s, crawled_time = CURRENT_TIMESTAMP()"""
                sql_data = (domain, data['pageUrl'], author, data['title'], published_date, data['text'], html_content, links, tags, 
                            domain, author, data['title'], published_date, data['text'], html_content, links, tags)
                commit_to_db(sql_query, sql_data)
                #Formating for return
                articles.append({
                    'domain': domain, 
                    'url':data['pageUrl'],
                    'author':author,
                    'title':data['title'], 
                    'title_sentiment':None,
                    'title_toxicity':None,
                    'published_date':published_date,
                    'content':data['text'],
                    'content_sentiment':None,
                    'content_toxicity':None,
                    'content_html':html_content,
                    'language':None, 
                    'links':links,
                    'tags':tags,
                    'crawled_time':datetime.datetime.now()
                })
                #Checking for comments
                if 'discussion' in data:
                    #Doing all non-reply comments first, then adding reply comments sorted by id (so we can get the comment_id from the db)
                    comment_data = [x for x in data['discussion']['posts'] if 'parentId' not in x] + \
                        sorted([x for x in data['discussion']['posts'] if 'parentId' in x], key=lambda k: k['id'])
                    for c in comment_data:
                        comment = {}
                        comment['domain'] = domain
                        comment['url'] = data['pageUrl']
                        comment['username'] = c['author'] if 'author' in c else None
                        comment['comment'] = c['text'] if 'text' in c else ""
                        comment['comment_original'] = c['html'] if 'html' in c else None
                        if 'date' not in c: comment['published_date'] = None
                        elif type(c['date']) == dict: comment['published_date'] = parse(c['date']['str'].replace("d",""))
                        else: comment['published_date'] = parse(c['date'])
                        comment['links'] = get_links(c['html']) if 'html' in c else None
                        comment['reply_count'] = len([x for x in comment_data if 'parentId' in x and c['id'] == x['parentId']])
                        parent_comment = [x for x in comment_data if 'parentId' in c and c['parentId'] == x['id']]
                        comment['reply_to'] = get_reply_to(parent_comment[0], data['pageUrl']) if parent_comment else None
                        insert_comment(comment)
        if len(diff_data) > 5:pbar.update()
    if len(diff_data) > 5:pbar.close()
    return articles

    
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
    percentage = (total_usage/api_calls["planCredits"])*100
    if total_usage >= api_calls["planCredits"]:
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

def diff_get_article(url, paging=True, count=0):
    #Checking for redirects
    try:
        r = requests.get(url)
    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError):
        LOGS.append({"url":url, "error": "Max retries exceeded with url. Caused by SSLError(SSLError('bad handshake: SysCallError(10054, 'WSAECONNRESET')"})
        return []
    except requests.exceptions.InvalidSchema:
        LOGS.append({"url":url, "error": "Invalid URL Schema. Check the URL"})
        return []
    #Sending Diffbot request
    diff_check_account_balance()
    if get_tld(r.url, as_object=True).domain in url:
        diff_endpoint = "https://api.diffbot.com/v3/analyze?token={}&url={}&paging={}".format(token, urllib.parse.quote(url), paging)
        try:
            r = requests.get(diff_endpoint, timeout=60*5)
        except requests.exceptions.Timeout:
            LOGS.append({"url":url, "error": "The diffbot request timed out."})
            return []
        if r.status_code in (504,):
            if count > 2: 
                LOGS.append({"url":url, "error": r.text})
                return []
            else: return diff_get_article(url, count=count+1)
        request = r.json()
    else:
        print("\nThis URL is redirecting: {}".format(url))
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
            LOGS.append({"url":url, "error":request['error']})
        else: 
            print("\n{} url: {}".format(request['error'], url))
            LOGS.append({"url":url, "error":request['error']})
        return []
    #Returing Data
    diff_data = request['objects']
    return process_diff_data(diff_data)
    
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

def open_keywords(use=None, fname='keywords.json',):
    with open(fname, encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not use: return data
    else:
        try: return data[use]
        except KeyError: raise KeyError("You chose a project name that is not in {}: {}".format(fname, use))

def get_connection():
    connection = pymysql.connect(host='144.167.35.221',
                                user='diffbot',
                                password='Cosmos1',
                                db='blogs',
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



if __name__ == "__main__":
    main()