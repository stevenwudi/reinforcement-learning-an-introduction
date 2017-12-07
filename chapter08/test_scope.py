from __future__ import print_function
import numpy as np

def foo(a):
    a[0] = a[0]+1
    return 0

def flo_c(class_c):
    class_c.a += 1

class floo():
    def __init__(self):
        self.a = 0


def main():
    a = np.array([1])
    a
    for i in range(3):
        print(foo(a))
    print(a)

    # c = floo()
    # for i in range(3):
    #     flo_c(c)
    #     print(c.a)





main()