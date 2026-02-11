class s:
 def fib(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return s.fib(n - 1) + s.fib(n - 2)
import numpy as np
from sklearn.pipeline import Pipeline
class c:
   def fib(n):
      return n == np.inf
   

v=Pipeline([("fib",s),("fib2",c)])

print(v.fit([1,2,3,4,5,6,7,8,9,10]))