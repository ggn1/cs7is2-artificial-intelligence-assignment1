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
        time_start = time.time() # keep track of time
        res = solver(*args, **kwargs)
        # Add executing time in seconds to the result 
        # that is to be returned.
        res['seconds'] = time.time() - time_start 
        return res
    
    return wrapper