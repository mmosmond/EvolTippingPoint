#Author: Matthew Osmond <mmosmond@zoology.ubc.ca>
#Description: Burger & Lynch 1995 discrete time moving optimum model

import numpy as np
import time
import pickle
#import matplotlib.pyplot as plt
import random

######################################################################
##HELPER FUNCTIONS##
######################################################################

def open_output_files(K,B,omega,mu,alphasqrd,n,k,rep):
	"""
	This function opens the output files and returns file
	handles to each.
	"""

	sim_id = 'K%d_B%d_w%r_u%.4f_alphasqrd%.2f_n%d_k%.3f_rep%d' %(K,B,omega,mu,alphasqrd,n,k,rep)
	data_dir = 'altdata'

	outfile_A = open("%s/n_%s.pkl" %(data_dir,sim_id),"wb")
	outfile_B = open("%s/genos_%s.pkl" %(data_dir,sim_id),"wb")
	outfile_C = open("%s/phenos_%s.pkl" %(data_dir,sim_id),"wb")
	outfile_D = open("%s/gens_%s.pkl" %(data_dir,sim_id),"wb")
	outfile_E = open("%s/numparents_%s.pkl" %(data_dir,sim_id),"wb")

	return [outfile_A, outfile_B, outfile_C, outfile_D, outfile_E]

def write_data_to_output(fileHandles, data):
	"""
	This function writes a (time, data) pair to the
	corresponding output file. We write densities
	not abundances.
	"""
	
	for i in range(0,len(fileHandles)):
		pickle.dump(data[i],fileHandles[i])

	# for i in range(0,len(fileHandles)):
	#   pickle.dump(data[i],fileHandles[i])

def close_output_files(fileHandles):
	"""
	This function closes all output files.
	"""

	for i in range(0,len(fileHandles)):
		fileHandles[i].close()


######################################################################
##PARAMETERS##
######################################################################

K = 512 # maximum population size (positive integer)
B = 2 #number of offspring per generation per parent (positive integer)
omega = 3 #square root of the inverse of the strength of selection (positive real)
mu = 0.0002 #mutation probability per locus (0<u<1) (not really: n*u is the mutation probability per gamete, only one locus per gamete mutates)
alphasqrd = 0.05 #mutation variance (positive real)
n = 50 #number of loci
ks = [i * 0.01 for i in range(0,26)] #rates of change in optimum trait (non-negative reals)

burngens = 1000 #number of generations before environment begins to change
maxgens = 10000 + burngens #maximum number of generations (positive integer)
outputFreq = 100 #record and print update this many generations (positive integer)
nReps = 10 #number of replicates to run

running = True #run simulations
plotting = False #plots pop up when done
transforming = True #convert pkl files to csv for plotting in mathematica

######################################################################
##SIMULATION##
######################################################################

def main():

	for i in range(len(ks)):
		k = ks[i]

		rep = 0
		while rep < nReps:

			randalleles = np.random.normal(0, 0.1*alphasqrd**0.5, (n,5)) #5 random allelic effects for each locus
			genos = np.array([]).reshape(0,2,n)
			for j in range(0,K): #for each of K initial individuals
				idx1 = np.random.randint(5, size=n) #index of randomly chosen allele for each locus
				chromo1 = [randalleles[i][idx1[i]] for i in range(0,n)] #random chromosome
				idx2 = np.random.randint(5, size=n) #index of randomly chosen allele for each locus
				chromo2 = [randalleles[i][idx2[i]] for i in range(0,n)] #random chromosome
				genos = np.vstack([genos,[[chromo1,chromo2]]])

			#genos = np.array([[[0]*n] * 2] * K) #genotypes (each of 2n loci (diploid) begins with effect 0 for all K individuals)

			# open output files
			fileHandles = open_output_files(K,B,omega,mu,alphasqrd,n,k,rep) 

			gen = 0 #generation
			while gen < maxgens + 1:

				# 1. population regulation
				if len(genos) > K:
					genos = genos[random.sample([i for i in range(len(genos))], K)] #randomly choose K individuals if more than K
				numparents = len(genos) #number of parents

				# 2. production of offspring
				pairs = np.resize(random.sample([i for i in range(len(genos))], len(genos)), (int(len(genos)/2), 2)) #random mate pairs (each mates at most once and not with self)
				#make 2B offspring from each pair
				off = np.array([]).reshape(0,2,n)
				for i in range(2*B):
					rand1 = np.random.randint(2, size=(len(genos), n)) #random number giving chromosome each locus inherited from on each gamete
					rec = np.resize(np.append(rand1, 1-rand1, axis=1),(len(pairs), 2, 2, n)) #reshape for next step
					newoff = np.sum(genos[pairs] * rec, axis=2) #combine chosen loci into diploid offspring
					off = np.vstack([off, newoff])
				#mutations
				rand2 = np.random.uniform(size = (len(off),2,1)) #random uniform number in [0,1] for each chromosome (gamete)
				whomuts = rand2 < n*mu # mutate if random number is below mutation rate; returns which chromosomes mutate
				nmuts = np.sum(whomuts) #number of mutations
				newmuts = np.random.normal(0, alphasqrd**0.5, nmuts) #genotypic effect of new mutations
				wheremut = np.where(whomuts) #indices of chromosomes that mutate
				wheremut2 = (wheremut[0], wheremut[1], np.random.randint(n, size=nmuts)) #individuals, chromosomes, and random locus chosen to mutate (only one per chromosomes)
				off[wheremut2] = off[wheremut2] + newmuts #add mutation

				# 3. viability selection
				if gen < burngens: # in burn-in period keep optimum constant
					opt = 0
				else:	# after burn-in increase optimum by k each gen
					opt = k * (gen - burngens) 
				phenos = np.sum(np.sum(off, axis=1), axis=1) + np.random.normal(0, 1, len(off)) #phenotypes of offspring (add random normal environmental effect)
				dist = phenos - opt #phenotypic distance from optimum
				w = np.exp(-(1-np.exp(-(dist**2)/(2*omega**2)))) #probability of survival
				rand3 = np.random.uniform(size = len(off)) #random uniform number in [0,1] for each individual
				genos = off[rand3 < w] #survivors make next generations parents
						
				#end simulation if extinct (need at least two individuals for mating)        
				if len(genos) < 2: 
					#record first
					genotypes = np.sum(np.sum(off, axis=1), axis=1) #genotypic values before selection (because that is when phenotypes calculated)
					write_data_to_output(fileHandles, [len(genos),genotypes,phenos,gen,numparents])
					print("k %.3f     rep %d    gen %d    N %d" %(k, rep, gen, len(genos))) 
					#end
					print("Extinct")              
					break 
					
				#otherwise continue
				# dump data every outputFreq iteration
				# also print a short progess message (generation and number of parents)
				if (gen % outputFreq) == 0:
					genotypes = np.sum(np.sum(off, axis=1), axis=1) #genotypic values before selection (because that is when phenotypes calculated)
					write_data_to_output(fileHandles, [len(genos),genotypes,phenos,gen,numparents])
					print("k %.3f     rep %d    gen %d    N %d" %(k, rep, gen, len(genos)))  

				# go to next generation
				gen += 1

			# cleanup
			close_output_files(fileHandles)

			# next replicate run
			rep += 1

######################################################################
##RUNNING##
######################################################################    

if running:	
	
	#run (with timer)
	start = time.time()
	main()
	end = time.time()
	print(end-start)

######################################################################
##PLOTTING##
######################################################################   

if plotting:

	rep = 0
	sim_id = 'K%d_B%d_omegasqrd%r_u%r_alphasqrd%r_n%d_k%r_rep%d' %(K,B,omega,mu,alphasqrd,n,k,rep)
	data_dir = 'altdata'

	# load abundance data
	g = open('%s/n_%s.pkl' %(data_dir,sim_id), 'rb')
	N = []
	while 1:
	    try:
	        N.append(pickle.load(g))
	    except EOFError:
	        break

	# load genotype data
	f = open('%s/genos_%s.pkl' %(data_dir,sim_id), 'rb')
	genos = []
	while 1:
	    try:
	        genos.append(pickle.load(f))
	    except EOFError:
	        break

	# load phenotype data
	g = open('%s/phenos_%s.pkl' %(data_dir,sim_id), 'rb')
	phenos = []
	while 1:
	    try:
	        phenos.append(pickle.load(g))
	    except EOFError:
	        break

	# load generation data
	f = open('%s/gens_%s.pkl' %(data_dir,sim_id), 'rb')
	gens = []
	while 1:
	    try:
	        gens.append(pickle.load(f))
	    except EOFError:
	        break

	opt = [k * (gens[i] - burngens) if gens[i] > burngens else 0 for i in range(len(gens))] #optimum
	mean_pheno = [np.mean(phenos[i]) for i in range(len(phenos))] #mean phenotype
	mean_lag = [opt[i] - mean_pheno[i] for i in range(len(phenos))] #mean lag

	gen_var = [np.var(genos[i]) for i in range(len(genos))] #variance in genotypic value

	fig1 = plt.figure()

	plt1 = plt.subplot(311)
	plt1.plot(gens, N)
	plt1.set_ylabel('# survivors')

	plt2 = plt.subplot(312)
	plt2.plot(gens, mean_lag)
	plt2.set_ylabel('lag')

	plt3 = plt.subplot(313)
	plt3.plot(gens, gen_var)
	plt3.set_ylabel('variance')
	plt3.set_xlabel('generations')

	plt.show()
	fig1.savefig('%s/plot_%s.png' %(data_dir,sim_id))

######################################################################
##TRANSFORMING TO CSV##
######################################################################   

if transforming:

	for i in range(len(ks)):
		k = ks[i]

		for j in range(nReps):
			rep = j

			sim_id = 'K%d_B%d_w%r_u%.4f_alphasqrd%.2f_n%d_k%.3f_rep%d' %(K,B,omega,mu,alphasqrd,n,k,rep)
			data_dir = 'altdata'

			# load abundance data
			g = open('%s/n_%s.pkl' %(data_dir,sim_id), 'rb')
			N = []
			while 1:
			    try:
			        N.append(pickle.load(g))
			    except EOFError:
			        break

			# load genotype data
			f = open('%s/genos_%s.pkl' %(data_dir,sim_id), 'rb')
			genos = []
			while 1:
			    try:
			        genos.append(pickle.load(f))
			    except EOFError:
			        break

			# load phenotype data
			g = open('%s/phenos_%s.pkl' %(data_dir,sim_id), 'rb')
			phenos = []
			while 1:
			    try:
			        phenos.append(pickle.load(g))
			    except EOFError:
			        break

			# load generation data
			f = open('%s/gens_%s.pkl' %(data_dir,sim_id), 'rb')
			gens = []
			while 1:
			    try:
			        gens.append(pickle.load(f))
			    except EOFError:
			        break

			# load number of parents data
			f = open('%s/numparents_%s.pkl' %(data_dir,sim_id), 'rb')
			numparents = []
			while 1:
			    try:
			        numparents.append(pickle.load(f))
			    except EOFError:
			        break

			import csv
			with open("%s/n_%s.csv" %(data_dir,sim_id), "w") as f:
				writer = csv.writer(f)
				writer.writerows([N])

			with open("%s/genos_%s.csv" %(data_dir,sim_id), "w") as f:
				writer = csv.writer(f)
				writer.writerows(genos)

			with open("%s/phenos_%s.csv" %(data_dir,sim_id), "w") as f:
				writer = csv.writer(f)
				writer.writerows(phenos)

			with open("%s/gens_%s.csv" %(data_dir,sim_id), "w") as f:
				writer = csv.writer(f)
				writer.writerows([gens])

			with open("%s/numparents_%s.csv" %(data_dir,sim_id), "w") as f:
				writer = csv.writer(f)
				writer.writerows([numparents])
