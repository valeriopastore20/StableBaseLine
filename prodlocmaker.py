from math import e
import numpy as np
import random
import argparse

"""
	Questo programma crea una disposizione iniziale random del numero di prodotti passato come parametro
"""
parser = argparse.ArgumentParser()
parser.add_argument("num_prod", type=int, help="Numero prodotti (deve essere un quadrato)")
args = parser.parse_args()

num_prod = args.num_prod

arr = np.arange(num_prod)
np.random.shuffle(arr)

path = "./prodLocFile"+str(num_prod)+".txt"
file = open(path,"w")
for i in range(num_prod):
	line = str(i)+" : "+str(arr[i])+"\n"
	file.write(line)
file.close()
