# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:12:16 2018

@author: X.X
"""

#%%


import os
import time
import threading
import multiprocessing

#from RB_test import generate_randomized_clifford_sequence
#%%

NUM_WORKERS = 4
 
def only_sleep():
    """ Do nothing, wait for a timer to expire """
    
    print("PID: %s, Process Name: %s, Thread Name: %s" % (
        os.getpid(),
        multiprocessing.current_process().name,
        threading.current_thread().name)
    )
    
#    print('running')
    time.sleep(1)
 
 
def crunch_numbers():
    """ Do some computations """
    print('running')
    print("PID: %s, Process Name: %s, Thread Name: %s" % (
        os.getpid(),
        multiprocessing.current_process().name,
        threading.current_thread().name)
    )
    x = 0
    while x < 10000000:
        x += 1


#%%
NUM_WORKERS = 4
gate = ['I', 'Xp', 'Yp', 'Zp']

start_time = time.time()
for i in range(NUM_WORKERS):
    generate_randomized_clifford_sequence(gate[i])
end_time = time.time()

print("Serial time=", end_time - start_time)
 
# Run tasks using threads
start_time = time.time()
#threads = [threading.Thread(target=generate_randomized_clifford_sequence, args = ('Zp')) for i in range(NUM_WORKERS)]
threads = [threading.Thread(target=generate_randomized_clifford_sequence) for i in range(NUM_WORKERS)]
[thread.start() for thread in threads]
[thread.join() for thread in threads]
end_time = time.time()
 
print("Threads time=", end_time - start_time)
 
# Run tasks using processes
start_time = time.time()
processes = [multiprocessing.Process(target=generate_randomized_clifford_sequence(gate[i])) for i in range(NUM_WORKERS)]
[process.start() for process in processes]
[process.join() for process in processes]
end_time = time.time()
 
print("Parallel time=", end_time - start_time)






#%%
## Run tasks serially
start_time = time.time()
for _ in range(NUM_WORKERS):
    only_sleep()
end_time = time.time()
 
print("Serial time=", end_time - start_time)
 
# Run tasks using threads
start_time = time.time()
threads = [threading.Thread(target=only_sleep) for _ in range(NUM_WORKERS)]
[thread.start() for thread in threads]
[thread.join() for thread in threads]
end_time = time.time()
 
print("Threads time=", end_time - start_time)
 
# Run tasks using processes
start_time = time.time()
processes = [multiprocessing.Process(target=only_sleep) for _ in range(NUM_WORKERS)]
[process.start() for process in processes]
[process.join() for process in processes]
end_time = time.time()
 
print("Parallel time=", end_time - start_time)

#%%
'''
start_time = time.time()
for _ in range(NUM_WORKERS):
    crunch_numbers()
end_time = time.time()
 
print("Serial time=", end_time - start_time)
 
start_time = time.time()
threads = [threading.Thread(target=pulsar.program_awgs, args = (sequence, *elts, AWGs = [awg[i]])) for i in range(NUM_WORKERS)]
[thread.start() for thread in threads]
[thread.join() for thread in threads]
end_time = time.time()
 
print("Threads time=", end_time - start_time)
 
 
start_time = time.time()
processes = [multiprocessing.Process(target=crunch_numbers) for _ in range(NUM_WORKERS)]
[process.start() for process in processes]
[process.join() for process in processes]
end_time = time.time()
 
print("Parallel time=", end_time - start_time)

'''
#%%     test loading sequence

awgs = ['awg', 'awg2']
NUM_WORKERS = 4
for i in range(NUM_WORKERS):
#    pulsar.program_awgs(sequence, *elts, AWGs = [awg[i]],)
    crunch_numbers()
end_time = time.time()
 
print("Serial time=", end_time - start_time)
 
start_time = time.time()
threads = [threading.Thread(target=crunch_numbers) for _ in range(NUM_WORKERS)]
[thread.start() for thread in threads]
[thread.join() for thread in threads]
end_time = time.time()
 
print("Threads time=", end_time - start_time)
 
 
start_time = time.time()
processes = [multiprocessing.Process(target=crunch_numbers) for _ in range(NUM_WORKERS)]
[process.start() for process in processes]
[process.join() for process in processes]
end_time = time.time()
 
print("Parallel time=", end_time - start_time)

#%%

#output = multiprocessing.Queue()

elements = experiment.elts

tvals = 1
waveforms = 1
#global tvals, waveforms

def get_wfm():# idx, output):
#    global tvals, waveforms
    i = -22
    print(222)
    print("PID: %s, Process Name: %s, Thread Name: %s" % (
        os.getpid(),
        multiprocessing.current_process().name,
        threading.current_thread().name)
    )
    tvals, waveforms = elements[i].normalized_waveforms()
#    output.put((idx, waveforms))
    print(1)
    return True


start_time = time.time()

length = len(elements)
NUM_WORKERS = 4
for i in range(NUM_WORKERS):
    get_wfm()
end_time = time.time()
 
print("Serial time=", end_time - start_time)
 
start_time = time.time()
#threads = [threading.Thread(target=get_wfm, args = (-22,)) for i in range(NUM_WORKERS)]
threads = [threading.Thread(target=get_wfm) for i in range(NUM_WORKERS)]
[thread.start() for thread in threads]
[thread.join() for thread in threads]
end_time = time.time()

print("Threads time=", end_time - start_time)

start_time = time.time()
#processes = [multiprocessing.Process(target=get_wfm, args = (-22,)) for i in range(NUM_WORKERS)]
processes = [multiprocessing.Process(target=get_wfm) for i in range(NUM_WORKERS)]
[process.start() for process in processes]
[process.join() for process in processes]
end_time = time.time()
 
print("Parallel time=", end_time - start_time)


#results = [output.get() for p in processes]
