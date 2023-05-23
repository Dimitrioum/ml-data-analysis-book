
def cut_time(timestamp, cut_milisecs=True):
    if cut_milisecs:
        return str(timestamp).split('.')[0]
    else:
        return str(timestamp)
