import pymysql
import json
import time
import os

class SqlFuncs(): 
    def __init__(self, conn):
        self.conn = conn

    def get_connection(self, conn):
        count = 0
        while True:
            count += 1
            try:
                host, user, password, db = conn
                connection = pymysql.connect(host=host,user=user,password=password,db=db,
                                    charset='utf8mb4',
                                    use_unicode=True,
                                    cursorclass=pymysql.cursors.DictCursor)
                # connection = pymysql.connect(host='144.167.35.73',
                #                     user='wale',
                #                     password='abcd1234!',
                #                     db='blogtrackers',
                #                     charset='utf8mb4',
                #                     use_unicode=True,
                #                     cursorclass=pymysql.cursors.DictCursor)
                # connection = pymysql.connect(host='cosmos-1.host.ualr.edu',
                #                     user='ukraine_user',
                #                     password='summer2014',
                #                     db='blogtrackers',
                #                     charset='utf8mb4',
                #                     use_unicode=True,
                #                     cursorclass=pymysql.cursors.DictCursor)
                return connection
            #Error handeling
            except Exception as e:
                if isinstance(e, pymysql.err.OperationalError): 
                    # print('trying')
                    # Unable to access port (Windows Error), trying again
                    # See https://docs.microsoft.com/en-us/biztalk/technical-guides/settings-that-can-be-modified-to-improve-network-performance
                    # print("Socket error uploading to db. Trying again... {}".format(count))
                    time.sleep(2)
                    count += 1
                    if count > 10: print("Failed to connect to db {} times in a row".format(count))
                else: 
                    # Uncaught errors
                    raise Exception("We aren't catching this mySql get_connection Error: {}".format(e))


    def commit_to_db(self, query, data, db_dame):
        # while True: 
        try:
            connection = self.get_connection(db_dame)
            with connection.cursor() as cursor:
                cursor.execute(query, data)
                connection.commit()
                connection.close()
                return            
        #Error handeling
        except Exception as e:
            if isinstance(e, pymysql.err.IntegrityError) and e.args[0]==1062:
                # Duplicate Entry, already in DB
                print(e)
                connection.close() 
                return
            elif e.args[0] == 1406:
                # Data too long for column
                print(e)
                connection.close()
                return 
            else: 
                # Uncaught errors
                raise Exception("We aren't catching this mySql commit_to_db Error: {}".format(e))

    def update_item(self, query, data, conn):
        connection = self.get_connection(conn)
        with connection.cursor() as cursor:
            try:
                cursor.execute(query, data)
                connection.commit()
                cursor.close()
                connection.close()
                return ''
            except Exception as e:
                if 'Duplicate entry' in str(e):
                    cursor.close()
                    connection.close()
                    return str(e)
                else:
                    cursor.close()
                    connection.close()
                    return False
            
        

    def update_insert(self, query, data, conn):
        connection = self.get_connection(conn)
        with connection.cursor() as cursor:
            try:
                cursor.execute(query, data)
                connection.commit()
            except Exception as e:
                if 'Duplicate entry' in str(e):
                    connection.close()
                    return str(e)
                else:
                    connection.close()
                    return ''
            
        connection.close()
        return ''