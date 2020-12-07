
import multiprocessing 
import time 
  
  
class Process(multiprocessing.Process): 
    def __init__(self, id): 
        super(Process, self).__init__() 
        self.id = id
                 
    def run(self): 
        time.sleep(5-self.id*2) 
        print("I'm the process with id: {}".format(self.id)) 
        return "done"
  
if __name__ == '__main__': 
    p0 = Process(0) 
    p1 = Process(1) 
    p0.start() 
    p1.start() 
    p0.join() 
    p1.join() 
    print("hi")
