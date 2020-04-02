from math import e
import numpy as np
import random
import argparse

"""
	Questo programma genera il file dei frequent item set del numero di prodotti passati come parametro.
"""

parser = argparse.ArgumentParser()
parser.add_argument("num_prod", type=int, help="Numero prodotti (deve essere un quadrato)")
parser.add_argument("max_fq", type=int, help="frequenza piu' alta")
args = parser.parse_args()

dec_fact = 4 #decreasing factor
min_fract = 10
max_fract = 5


num_prod = args.num_prod
max_fq = args.max_fq
path = "./fisFile"+str(num_prod)+".txt"
# Metodo che scrive il file contenente i frequent item sets
fq_disp = np.zeros(num_prod)
file = open(path,"w")
file.write("Num. prods = "+str(num_prod)+"\n")
for i in range(0,num_prod):
	freq = int(max_fq*e**(-i/(num_prod/dec_fact)))
	file.write("{ "+str(i)+" }\t\t"+str(freq)+"\n")
	try:
		fq_disp[i] = freq - random.randrange(int(freq/min_fract),int(freq/max_fract)) #numero massimo di volte in cui e` stato venduto con altri
	except:
		fq_disp[i] = 0
for i in range(0,num_prod):
	low = int(fq_disp[i]/6)
	high = fq_disp[i]  # low,high : range di selezione del numero di altri prodotti con cui e` stato acquistato
	if low < high:
		times = random.randrange(low,high)
	else:
		continue
	if i + times >= num_prod:
		remaining =  num_prod - i
		times = random.randrange(0,remaining)
	if times == 0:
		continue
	previous = np.full(times+1,i) # array che serve per far si che non vengano riscelti gli stessi valori
	for k in range(0,times):
		j = random.randrange(i+1,num_prod)
		while j in previous:
			j = random.randrange(i+1,num_prod)
		previous[k+1] = j
		minor = min(fq_disp[i],fq_disp[j])
		if minor > 1:
			freq = random.randrange(1,minor)
			fq_disp[i]-=freq
			fq_disp[j]-=freq
			file.write("{ "+str(i)+" , "+str(j)+" }\t\t"+str(freq)+"\n")
file.close()