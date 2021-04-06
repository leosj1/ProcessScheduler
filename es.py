from elasticsearch import Elasticsearch, AsyncElasticsearch
import json
import time
from elasticsearch import helpers
import requests


# import elasticsearch
# es = elasticsearch.client.SqlClient("144.167.35.89")
# q = es.query(body ="select * from entity_narratives_pool order by blogpost_id desc limit 10", format=json)
# print('here')


class Es():
    def get_client(self, ip):
        count = 0
        while True:
            count += 1
            try:
                client = Elasticsearch([
                    {'host': ip},
                ])
                return client
            #Error handeling
            except Exception as e:
                time.sleep(3)
                count += 1
                if count > 10:
                    print(
                        "Failed to connect to Elasticsearch {} times in a row".format(count))
                else:
                    # Uncaught errors
                    raise Exception(
                        "We aren't catching this Elasticsearch get_client Error: {}".format(e))

    def bulk_request(self, client, actions):
        count = 0
        while True:
            count += 1
            try:
                bulk_action = helpers.bulk(
                    client, actions, request_timeout=500)
                # bulk_action = helpers.parallel_bulk(client, actions, max_chunk_bytes=10485760000, chunk_size=10000, thread_count=10)
                return bulk_action
            #Error handeling
            except Exception as e:
                time.sleep(1)
                count += 1
                if count > 100:
                    print("Failed to work on Bulk {} times in a row, error is {}".format(
                        count, str(e)))
                # else:
                #     # Uncaught errors
                #     raise Exception("We aren't catching this Elasticsearch bulk_request Error: {}".format(e))

    # Delete
    def delete_record(self, client, index, record_id, doc_type):
        try:
            x = client.delete(
                index=index,
                # doc_type=doc_type,
                id=record_id
            )
            return True
        except Exception as e:
            # Catching error
            # err = json.loads(e.error)
            # error_message = err['result'] if 'result' in err else e.error
            # print(error_message)
            return False
    
    def delete_by_query(self, client, index, body, params=None, headers=None):
        try:
            delete_result = client.delete_by_query(index, body, params=None, headers=None, request_timeout=30)
            if 'failures' in delete_result:
                if not delete_result['failures']:
                    return True
            return delete_result
        except Exception as e:
            return False

    # Update
    def update_record(self, client, index, record_id, doc_type, json_body):
        try:
            client.update(
                index=index,
                # doc_type=doc_type,
                id=record_id,
                body={
                    "doc": json_body
                },
                request_timeout=30
            )
            return True
        except Exception as e:
            # Catching error
            if 'document_missing_exception' in str(e.error):
                return False

    # Insert
    def insert_record(self, client, index, record_id, doc_type, json_body):
        response = client.index(
            index=index,
            # doc_type=doc_type,
            id=record_id,
            body=json_body,
            request_timeout=30
        )
        result = response['result'] if 'result' in response else None
        if result == 'created':
            return True
        else:
            return False

    def index_record(self, client, index, record_id, doc_type, json_body):
        response = client.index(
            index=index,
            doc_type=doc_type,
            id=record_id,
            body=json_body,
            request_timeout=30
        )
        result = response['result'] if 'result' in response else None
        if result == 'created':
            return True
        else:
            return False

    # Search
    def search_record(self, client, index, json_request):
        count = 0
        while True:
            count += 1
            try:
                response = client.search(
                    index=index,
                    body=json_request,
                    request_timeout=30
                )
                return response
            except Exception as e:
                time.sleep(1)
                count += 1
                if count > 100:
                    print(
                        "Failed to get search result {} times in a row and error is --- {}".format(count, str(e)))

    # Scroll

    def scroll(self, client, index, json_request):
        result = []
        errors = []

        response = client.search(
            index=index,
            body=json_request,
            scroll='2s',  # length of time to keep search context,
            request_timeout=30
        )

        while True:
            result_hits = response['hits']['hits']
            if result_hits:
                for x in result_hits:
                    source = x['_source']
                    if 'blogpost_id' in source:
                        if source['blogpost_id'] not in result:
                            result.append(source)

                scroll_id = response['_scroll_id']

                try:
                    response = client.scroll(
                        scroll_id=scroll_id,
                        scroll='10s'
                    )
                except Exception as e:
                    errors.append(e)

            else:
                break

        return result

# es = Es()
# client = es.get_client("144.167.35.89")

# b = es.bulk_request(client, actions)
# print('done')

# es.delete_record(client, "entity_narratives", "fs6GDHcBzeEK5SP2XyM3", "")

class AsyncEs():
    async def get_client_async(self, ip):
        count = 0
        while True:
            count += 1
            try:
                client = AsyncElasticsearch([
                    {'host': ip},
                ])
                return client
            #Error handeling
            except Exception as e:
                time.sleep(3)
                count += 1
                if count > 10:
                    print(
                        "Failed to connect to Elasticsearch {} times in a row".format(count))
                else:
                    # Uncaught errors
                    raise Exception(
                        "We aren't catching this Elasticsearch get_client Error: {}".format(e))