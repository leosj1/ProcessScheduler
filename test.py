# import time
# import atexit

# from apscheduler.schedulers.background import BackgroundScheduler
# from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor


# def print_date_time():
#     time.sleep(5)
#     print(time.strftime("%A, %d. %B %Y %I:%M:%S %p"))

# executors = {
#     'default': ThreadPoolExecutor(60),
#     'processpool': ProcessPoolExecutor(60)
# }

# job_defaults = {
#     'coalesce': False,
#     'max_instances': 50
# }
# scheduler = BackgroundScheduler(job_defaults = job_defaults)
# # scheduler = BackgroundScheduler()
# scheduler.add_job(func=print_date_time, trigger="interval", seconds=3)
# scheduler.start()

# # Shut down the scheduler when exiting the app
# atexit.register(lambda: scheduler.shutdown())

# import requests
# r = requests.get('https://www.walmart.com/ip/Xbox-Series-X/443574645')

import json
data = json.load(open('test2.json', 'r'))
print('here')