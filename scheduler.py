from tqdm import trange
import subprocess
import time
import os
if __name__ == "__main__": 
    #makes the log dir if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), "Logs"), exist_ok=True)
    while True:
        try:
            subprocess.call([
                "python",
                "C:\COSMOS\ProcessScheduler\processes.py", 
                "|", 
                "tee", 
                "Logs\\Log_%date:~4,2%-%date:~7,2%-%date:~10,4%.log"])
            for i in trange(1, desc="Sleeping"):
                time.sleep(1)
        #Raises system exit if the API key has expired
        except SystemExit:
            print("\nStopping!")
            break
        except Exception as e:
            print(e)
            for i in trange(60*5, desc="Sleeping"):
                time.sleep(1)