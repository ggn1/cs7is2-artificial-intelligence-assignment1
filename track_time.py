import time

def track_time(solver):
    ''' This function will return a wrapper function
        that computes and returns both execution time
        as well as memory usage of given solver function.
        Any maze solver function that returns a path that
        solves the corresponding maze may be decorated with
        this function.
    '''
    def wrapper(*args, **kwargs):
        time_start = time.time() # start monitoring
        path = solver(*args, **kwargs)
        seconds = time.time() - time_start # stop tracking memory usage
        return path, seconds
    
    return wrapper