from threading import Thread

varnum = 0

def loop1():
    global varnum
    while True:
        print("a", varnum)

def predict():
    global varnum 
    while True:
        print("halo")
        varnum += 1

t1 = Thread(target=loop1)
t2 = Thread(target=predict)
t1.start()
t2.start()
