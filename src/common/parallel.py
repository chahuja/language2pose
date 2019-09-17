import multiprocessing
from multiprocessing import Pool

def parallel(fn, args):
  p = Pool(multiprocessing.cpu_count())
  p.map(fn, args)
