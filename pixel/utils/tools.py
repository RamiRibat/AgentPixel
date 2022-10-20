import os, signal

def kill_process(name):
    """
    Source: https://www.geeksforgeeks.org/kill-a-process-by-name-using-python/
    """
    try:
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
            fields = line.split()
            pid = fields[0]
            os.kill(int(pid), signal.SIGKILL)
        print("Process Successfully terminated")

    except:
        print("Error Encountered while running script")
