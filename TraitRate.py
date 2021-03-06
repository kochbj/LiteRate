#!/usr/bin/env python 
import argparse,sys
import os, csv, glob
from numpy import *
import numpy as np
from scipy.special import gamma
from scipy.special import beta as f_beta
import random as rand
import platform, time
import multiprocessing, thread
import multiprocessing.pool
import csv
from scipy.special import gdtr, gdtrix
from scipy.special import betainc
import scipy.stats
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)  
shape_beta_RJ = 10.
print "\n\n             LiteRate - 20180201\n"

####### BEGIN FUNCTIONS for RJMCMC #######
def prior_sym_beta(x,a): 
	return scipy.stats.beta.logpdf(x, a,a)

def random_choice(vector):
	ind = np.random.choice(range(len(vector)))
	return [vector[ind], ind]

def add_shift_RJ_weighted_mean(rates,times): # args: vector of rates and vector of times (t_origin, t_shifts, t_end)
	r_time, r_time_ind = random_choice(np.diff(times))
	delta_t_prime           = np.random.uniform(0,r_time)
	t_prime                 = times[r_time_ind] + delta_t_prime
	times_prime             = np.sort(np.array(list(times)+[t_prime]))[::-1]
	time_i1                 = times[r_time_ind]
	time_i2                 = times[r_time_ind+1]
	p1 = (time_i1-t_prime)/(time_i1-time_i2)
	p2 = (t_prime-time_i2)/(time_i1-time_i2)
	u = np.random.beta(shape_beta_RJ,shape_beta_RJ)  #np.random.random()
	rate_i                  = rates[r_time_ind]
	rates_prime1            = exp( log(rate_i)-p2*log((1-u)/u) )
	rates_prime2            = exp( log(rate_i)+p1*log((1-u)/u) )	
	rates_prime             = np.insert(rates,r_time_ind+1,rates_prime2)
	rates_prime[r_time_ind] = rates_prime1
	log_q_prob              = log(abs(r_time))-prior_sym_beta(u,shape_beta_RJ) # prob latent parameters: Gamma pdf
	Jacobian                = 2*log(rates_prime1+rates_prime2)-log(rate_i)
	# return new rate vector, new time vector, prod between hastings ratio and Jacobian
	return rates_prime,times_prime,log_q_prob+Jacobian

def remove_shift_RJ_weighted_mean(rates,times):
	random_indx = np.random.choice(range(1,len(times)-1))
	rm_shift_ind  = random_indx
	t_prime       = times[rm_shift_ind]
	time_i1       = times[rm_shift_ind-1]
	time_i2       = times[rm_shift_ind+1]
	dT            = abs(times[rm_shift_ind+1]-times[rm_shift_ind-1]) # if rm t_i: U[t_i-1, t_i+1]
	times_prime   = times[times != t_prime]	
	p1 = (time_i1-t_prime)/(time_i1-time_i2) # weights based on amount of time assigned to each rate
	p2 = (t_prime-time_i2)/(time_i1-time_i2)	
	rate_i1       = rates[rm_shift_ind-1] # remove rates from rate vector
	rate_i2       = rates[rm_shift_ind]
	rate_prime    = exp(p1 *log(rate_i1) + p2 *log(rate_i2)) # new rate based on wehgted mean of old rates	
	rm_rate       = rates[rm_shift_ind]
	rates_prime   = rates[rates != rm_rate]
	rates_prime[rm_shift_ind-1] = rate_prime	
	u             = 1./(1+rate_i2/rate_i1) # == rate_i1/(rate_i1+rate_i2)
	log_q_prob    = -log(dT)+prior_sym_beta(u,shape_beta_RJ) # log_q_prob_rm = 1/(log_q_prob_add)
	Jacobian      = log(rate_prime)-(2*log(rate_i1+rate_i2))
	# return new rate vector, new time vector, prod between hastings ratio and Jacobian
	return rates_prime,times_prime,log_q_prob+Jacobian
	
def RJMCMC(arg):
	# args = birth-rate vector (L), death rates (M), rate shifts for L and M 
	[L,M, timesL, timesM]=arg
	r=np.random.random(2)
	newL,newtimesL,log_q_probL = L,timesL,0
	newM,newtimesM,log_q_probM = M,timesM,0
	sample_shift_mu = 0.5
	min_allowed_n_rates = 1
	# update birth model with 50% chance
	if r[0]>sample_shift_mu:
		# ADD/REMOVE SHIFT LAMBDA
		if r[1]>0.5: 
			newL,newtimesL,log_q_probL = add_shift_RJ_weighted_mean(L,timesL)
		# if 1-rate model this won't do anything, keeping the frequency of add/remove equal
		elif len(L)> min_allowed_n_rates: # defined for the edgeShift model
			newL,newtimesL,log_q_probL = remove_shift_RJ_weighted_mean(L,timesL) 
		update_L = 1 # indicator (which par was changed by RJ)
	# update death model with 50% chance
	else:
		# ADD/REMOVE SHIFT MU
		if r[1]>0.5: 
			newM,newtimesM,log_q_probM = add_shift_RJ_weighted_mean(M,timesM)
		# if 1-rate model this won't do anything, keeping the frequency of add/remove equal
		elif len(M)> min_allowed_n_rates: # defined for the edgeShift model
			newM,newtimesM,log_q_probM = remove_shift_RJ_weighted_mean(M,timesM) 
		update_L = 0
	# return new rates (newL, newM), new times of rate shift (newtimesL/M), hastngs ratio times Jacobian
	return newL,newtimesL,newM,newtimesM,log_q_probL+log_q_probM,update_L

def get_post_rj_HP(xl,xm): # returns rate parameter for the Poisson distribution sampled directly from the posterior
	# Gamma hyper-prior on the rate parameter of the Poisson priors on number of rates
	G_shape_rjHP = 2. # 1.1
	G_rate_rjHP  = 1. # 0.1 # mode at 1
	n = 2 # sp, ex
	a = G_shape_rjHP + xl + xm
	b = G_rate_rjHP + n
	Poi_lambda_rjHP = np.random.gamma(a,1./b)
	#print "Mean Poi_lambda:", a/b
	return Poi_lambda_rjHP


####### LIKELIHOOD FUNCTIONS #######

def BD_partial_lik(arg):
	# time window defined by up and lo (max/min ages)
	# ts = times of speciation, te = times of extinction
	# par="l" calc likelihood for speciation
	[up,lo,rate,par]=arg
	# indexes of the species within time frame
	if par=="l": i_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
	else: i_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])	
	# get total time lived (or tot branch length) within time window
	n_S = get_sp_in_frame_br_length(up,lo)
	# number of events (n. sp or ex events)
	no_events = len(i_events)
	lik= log(rate)*no_events -rate*sum(n_S)
	return lik

def get_BDlik(times,rates,par):
	lik =0
	for i in range(len(rates)):
		up = times[i]
		lo = times[i+1]
		rate = rates[i]
		lik += BD_partial_lik([up,lo,rate,par])
	return lik

# VECTORIZED LIK FUNCTIONS
def get_sp_in_frame_br_length(up,lo):
	# index species present in time frame
	n_all_inframe = np.intersect1d((ts >= lo).nonzero()[0], (te <= up).nonzero()[0])
	# tot br length within time frame
	n_t_ts,n_t_te=zeros(len(ts)),zeros(len(ts))
	n_t_ts[n_all_inframe]= ts[n_all_inframe]   # speciation events before time frame
	n_t_ts[(n_t_ts>up).nonzero()]=up           # for which length is accounted only from $up$ rather than from $ts$	
	n_t_te[n_all_inframe]= te[n_all_inframe]   # extinction events in time frame
	n_t_te[np.intersect1d((n_t_te<lo).nonzero()[0], n_all_inframe)]=lo     # for which length is accounted only until $lo$ rather than to $te$
	# vector of br lengths within time frame  #(scaled by rho)
	n_S=((n_t_ts[n_all_inframe]-n_t_te[n_all_inframe])) #*rhos[n_all_inframe])
	return n_S

def precompute_events(arg):
	[up,lo]=arg
	# indexes of the species within time frame
	L_events=np.intersect1d((ts <= up).nonzero()[0], (ts > lo).nonzero()[0])
	M_events=np.intersect1d((te <= up).nonzero()[0], (te > lo).nonzero()[0])	
	# get total time lived (or tot branch length) within time window
	n_S = get_sp_in_frame_br_length(up,lo)
	return len(L_events), len(M_events), sum(n_S)
	
def get_rate_index(times):
	if len(times)==2: 
		ind =np.zeros(n_bins).astype(int)
	else:
		times = np.round(times+0)
		dT = abs(np.diff(times)).astype(int)
		#print dT, sum(dT), times
		ind = []
		[ ind.extend([i]*dT[i]) for i in range(len(times)-1) ]
		ind = np.array(ind)
		
		# alternative way to compute it
		#shifts = times[1:-1]
		#h = np.histogram(shifts,bins =rev_bins)[0][::-1]
		#ind = np.cumsum(h)
	return ind

def vect_lik(L_acc_vec,M_acc_vec):
	# BD likelihood
	try:
		Blik = sum(log(L_acc_vec)*sp_events_bin - L_acc_vec*br_length_bin) 
		Dlik = sum(log(M_acc_vec)*ex_events_bin - M_acc_vec*br_length_bin) 
	except:
		print len(L_acc_vec),len(M_acc_vec),len(sp_events_bin)
		sys.exit()
	return sum(Blik)+sum(Dlik)

# TRAIT DEPENDENT LIK FUNCTIONS
def get_likelihood_continuous_trait(l0,m0,alphaL,alphaM,tranf_rate_func):
	lik1 = sum(log(tranf_rate_func(l0,alphaL,tr_birth_events)))
	lik2 = -sum(tranf_rate_func(l0,alphaL,tr_waiting_times))
	lik3 = sum(log(tranf_rate_func(m0,alphaM,tr_death_events)))
	lik4 = -sum(tranf_rate_func(m0,alphaM,tr_waiting_times))
	#print lik1,lik2,lik3,lik4,l0,m0,alphaL,alphaM
	return lik1+lik2+lik3+lik4


####### PROPOSALS #######
def update_multiplier_freq(q,d=1.1,f=0.75):
	S=np.shape(q)
	ff=np.random.binomial(1,f,S)
	u = np.random.uniform(0,1,S)
	l = 2*log(d)
	m = exp(l*(u-.5))
	m[ff==0] = 1.
	# new vector of rates
 	new_q = q * m
	# Hastings ratio
	U=sum(log(m))
	return new_q,U

def update_multiplier(q,d=1.1):
	u = np.random.uniform(0,1)
	l = 2*log(d)
	m = exp(l*(u-.5))
	# new vector of rates
 	new_q = q * m
	# Hastings ratio
	U=sum(log(m))
	return new_q,U

def update_sliding_win(i, d = 1): 
	# boundaries within which we can have a rate shift
	m, M = min_time, max_time
	ii = i+(np.random.random()-.5)*d
	if ii<m: ii=(ii-m)+m
	if ii>M: ii=(M-(ii-M))
	if ii<m: ii=i
	else: ii=i
	return ii

def update_sliding_win_unbounded(i, d = 1): 
	ii = i+(np.random.random()-.5)*d
	return ii

def update_sliding_win_unbounded_vec(i, d = 1): 
	ii = i+(np.random.random(len(i))-.5)*d
	return ii

def update_times(times):
	rS= times+0.
	indx = np.random.choice(range(1,len(times)-1))
	rS[indx] = update_sliding_win(rS[indx])
	#
	#for i in : 
	#	rS[i]=update_parameter(times[i])	
	return np.sort(rS)[::-1]

####### PRIORS #######
def Poisson_prior(k,rate):
	return k*log(rate) - rate - sum(log(np.arange(1,k+1)))

def prior_gamma(L,a=2,b=2):  
	return sum(scipy.stats.gamma.logpdf(L, a, scale=1./b,loc=0))

def prior_normal(L,sd): 
	#print L, sd, scipy.stats.norm.logpdf(L,loc=0,scale=sd)
	return sum(scipy.stats.norm.logpdf(L,loc=0,scale=sd))

def prior_exponential(L,rate): 
	return sum(scipy.stats.expon.logpdf(L, scale=1./rate))

def get_rate_HP(l,m): 
	rates = np.array(list(l)+list(m))
	post_rate_prm = np.random.gamma( shape=hpGamma_shape+Gamma_shape*len(rates), scale=1./(hpGamma_rate+sum(rates)) )
	return post_rate_prm

####### MCMC looop #######
def runMCMC(arg):
	# initial values of rates, and times
	[L_acc, M_acc, timesLA, timesMA]  = arg
	Poi_lambda_rjHP = 1
	Gamma_rate = 1
	# init lik
	indLA = get_rate_index(timesLA)
	indMA = get_rate_index(timesMA)
	likA = sum(vect_lik(L_acc[indLA],M_acc[indMA]))
	#likA = get_BDlik(timesLA,L_acc,"l") + get_BDlik(timesMA,M_acc,"m")
	priorA = prior_gamma(L_acc) + prior_gamma(M_acc)
	priorA += -log(max_time-min_time)*(len(L_acc)-1+len(M_acc)-1)
	priorPoiA = Poisson_prior(len(L_acc),Poi_lambda_rjHP)+Poisson_prior(len(M_acc),Poi_lambda_rjHP)
	priorA += priorPoiA
	
	iteration = 0
	while iteration < n_iterations:		
		r = np.random.random(2)
		L,timesL = L_acc+0,timesLA+0
		M,timesM = M_acc+0,timesMA+0
		indL =indLA
		indM =indMA
		hasting = 0
		gibbs=0
		priorPoi = 0
		if r[0]< 0.4:
			# update birth part
			if r[1] < .5 or len(L_acc)==1:
				# update rates
				L, hasting = update_multiplier_freq(L_acc)
			else:
				# update times (hastings = 0 because we are doing symmetric update)
				timesL = update_times(timesLA)
				indL = get_rate_index(np.floor(timesL))
			
		elif r[0] < 0.8:
			# update M 
			if r[1] < .5 or len(M_acc)==1:
				# update rates
				M, hasting = update_multiplier_freq(M_acc)
			else:
				# update times (hastings = 0 because we are doing symmetric update)
				timesM = update_times(timesMA)
				indM = get_rate_index(np.floor(timesM))
			
		# elif r[0] < 0.99:
		# 	# do RJ
		# 	L,timesL, M,timesM, hasting, update_L = RJMCMC([L_acc,M_acc, timesLA, timesMA])
		# 	if update_L==1: indL = get_rate_index(np.floor(timesL))
		# 	else: indM = get_rate_index(np.floor(timesM))
		# 	priorPoi = Poisson_prior(len(L),Poi_lambda_rjHP)+Poisson_prior(len(M),Poi_lambda_rjHP)
			
		else: 
			# update HPs 
			Poi_lambda_rjHP = get_post_rj_HP(len(L_acc),len(M_acc))
			Gamma_rate = get_rate_HP(L_acc,M_acc)
			gibbs=1
		
		# prevent super small time frames
		if min(abs(np.diff(timesL)))<=min_allowed_t or min(abs(np.diff(timesM)))<=min_allowed_t: 
			prior = -np.inf	
			lik =  -np.inf			
		else:
			# calc acceptance ratio
			# prior on rate
			prior = prior_gamma(L,Gamma_shape,Gamma_rate) + prior_gamma(M,Gamma_shape,Gamma_rate)
			# prior on times of rate shift
			prior += -log(max_time-min_time)*(len(L)-1+len(M)-1)
			# prior on 
			if priorPoi != 0: 
				prior += priorPoi
			else: 
				prior += priorPoiA
				priorPoi = priorPoiA
			if gibbs==0:
				lik = sum(vect_lik(L[indL],M[indM]))
			else: 
				lik = likA
		
		# debug
		if check_lik==1:
			lik_old = get_BDlik(np.floor(timesL),L,"l") + get_BDlik(np.floor(timesM),M,"m")
			if iteration % 100==0: 
				print lik_old-lik 
		
		
		# print lik, likA, prior, priorA
		
		if lik-likA + prior-priorA + hasting >= log(np.random.random()) or gibbs==1:
			# update accepted values to proposed ones
			L_acc, M_acc, timesLA, timesMA = L,M,timesL, timesM
			# update lik, prior
			likA,priorA = lik, prior
			indLA,indMA = indL, indM
			priorPoiA = priorPoi
		
		if iteration % s_freq ==0:
			# MCMC log
			log_state = map(str,[iteration,likA+priorA,likA,priorA,mean(L_acc),mean(M_acc),len(L_acc),len(M_acc),max_time,min_time,Gamma_rate,Poi_lambda_rjHP])
			mcmc_logfile.write('\t'.join(log_state)+'\n')
			mcmc_logfile.flush()
			# log marginal rates/times
			log_state = map(str,list(L_acc) + list(timesLA[1:len(timesLA)-1]))
			sp_logfile.write('\t'.join(log_state)+'\n')
			sp_logfile.flush()
			log_state = map(str,list(M_acc) + list(timesMA[1:len(timesMA)-1]))
			ex_logfile.write('\t'.join(log_state)+'\n')
			ex_logfile.flush()
		
		if iteration % p_freq ==0:
			print iteration, likA, priorA
			# print on screen
			print "\tsp.times:", timesLA
			print "\tex.times:", timesMA
			print "\tsp.rates:", L_acc
			print "\tex.rates:", M_acc
		
		iteration +=1 

####### parse arguments #######
p = argparse.ArgumentParser() #description='<input file>') 

p.add_argument('-v',       action='version', version='%(prog)s')
p.add_argument('-d',       type=str, help='data file', default="", metavar="") 
p.add_argument('-n',       type=int, help='n. MCMC iterations', default=10000000, metavar=10000000)
p.add_argument('-p',       type=int, help='print frequency', default=1000, metavar=1000) 
p.add_argument('-s',       type=int, help='sampling frequency', default=1000, metavar=1000) 
p.add_argument('-seed',    type=int, help='seed (set to -1 to make it random)', default= 1, metavar= 1)
p.add_argument('-present_year',    type=int, help="""set to: -1 for standard pyrate datasets (time BP), \
0: time AD and present set to most recent TE, 1: time AD present user defined """, default= 0, metavar= 0)

args = p.parse_args()

if args.seed==-1:
	rseed=np.random.randint(0,9999)
else: rseed=args.seed	
random.seed(rseed)
np.random.seed(rseed)


n_iterations = args.n
s_freq = args.s
p_freq = args.p


####### Parse DATA #######
#f = args.d
f = "/Users/danielesilvestro/Software/LiteRate/example_dataTAD.txt"
t_file=np.loadtxt(f, skiprows=1)
ts_years = t_file[:,2]
te_years = t_file[:,3]

te_years = np.round(np.random.uniform(1950,2017,1000))
ts_years = np.round(np.random.uniform(1950,te_years,1000))
#


if args.present_year== -1: # to load regular pyrate input
	ts = ts_years
	te = te_years
elif args.present_year==0: # find max year and set to present
	ts = max(te_years) - ts_years 
	te = max(te_years) - te_years 
else: # user-spec present year
	ts = args.present_year - ts_years 
	te = args.present_year - te_years 

ts,te = np.round(ts),np.round(te)
max_time = max(ts)
min_time = min(te)



# make up some trait data
species_durations = (ts+1)-te # consider year of origination as a lived year

trait_list_of_arrays  = []

tr_waiting_times =[]
tr_birth_events =[]
tr_death_events =[]

list_all_values = []
for i in species_durations:
	species_trait_array = np.sort(np.random.normal(0,2,int(i)) )           # skewed values
	species_trait_array = np.sort(np.random.uniform(-5,5,int(i)) )          # severely skewed values
	#species_trait_array = np.random.normal(0,2,int(i))                     # no effects	
	#species_trait_array = np.random.uniform(0,1,int(i))                    # no effects	
	#species_trait_array = np.abs(np.sort(np.random.uniform(-1,1,int(i)) )) # large values in the extremes
	#species_trait_array = 1-np.abs(np.sort(np.random.uniform(-1,1,int(i)) )) # small values in the extremes
	#
	#
	# sp/ex happen at large or small values (no intermediate)
	#if np.random.random()<0.5:
	#	species_trait_array = np.abs(np.sort(np.random.uniform(-1,1,int(i)) )) # large values in the extremes
	#else:
	#	species_trait_array = 1-np.abs(np.sort(np.random.uniform(-1,1,int(i)) )) # small values in the extremes
	#	
	
	# sp/ex happen at large or small values (no intermediate)
	#if np.random.random()<0.5:
	#	species_trait_array = np.abs(np.sort(np.random.uniform(-1,0,int(i)) )) # large then small
	#else:
	#	species_trait_array = np.sort(np.random.uniform(0,1,int(i)) ) # small then large
	#	
	
	
	#a1 = np.sort(np.random.uniform(0,1,max(1,int(i/2.))))
	#a2 = np.sort(np.random.uniform(0,1,int(i/2.)))[::-1]
	#species_trait_array = np.concatenate((a1,a2))         # highest values in the middle
	
	list_all_values += list(species_trait_array)
	trait_list_of_arrays.append(species_trait_array)
	tr_waiting_times += list(species_trait_array) # all trait values
	tr_birth_events  += [species_trait_array[0]] # trait value at origination
	tr_death_events  += [species_trait_array[-1]] # trait value at extinction




def tranform_rate_exp(r0,alpha,trait):
	return r0 * np.exp(alpha*trait)

def tranform_rate_lin(r0,alpha,trait):
	new_rate=r0 + r0*alpha*trait
	new_rate[new_rate<0.00001] = 0.00001
	return new_rate

def transform_rate_beta(r0,prm,trait):
	alpha,beta = prm
	return r0 + r0*scipy.stats.beta.pdf(trait, alpha,beta)

def transform_rate_beta_rescaled(r0,prm,trait):
	alpha,beta = prm
	beta_pdf = scipy.stats.beta.pdf(trait, alpha,beta)
	mode = (alpha-1)/((alpha+beta-2.))
	pdf_mode = scipy.stats.beta.pdf(mode, alpha,beta)
	rescaled_pdf = pdf_mode - beta_pdf/pdf_mode
	return r0 + r0*rescaled_pdf

def transform_rate_beta_rescaled_truncated(r0,prm,trait):
	alpha,beta = prm
	beta_pdf = scipy.stats.beta.pdf(trait, alpha,beta)
	if min(prm)>1:
		mode = (alpha-1)/((alpha+beta-2.))
		pdf_mode = scipy.stats.beta.pdf(mode, alpha,beta)
		rescaled_pdf = beta_pdf/pdf_mode
	else:
		rescaled_pdf = beta_pdf # Y-axis truncation
		rescaled_pdf = rescaled_pdf/10.
		rescaled_pdf[beta_pdf>1] = 1.
	return  r0 + r0*rescaled_pdf

def transform_rate_beta_rescaled_truncated_unnormalized(r0,prm,trait):
	alpha,beta = prm
	beta_pdf = trait**(alpha-1)*(1-trait)**(beta-1)
	#print sum((beta_pdf)), alpha,beta,min(trait),max(trait)
	#quit()
	if min(prm)>1:
		m = (alpha-1)/((alpha+beta-2.))
		pdf_mode = m**(alpha-1)*(1-m)**(beta-1)
		rescaled_pdf = beta_pdf/pdf_mode
	else:
		rescaled_pdf = beta_pdf # Y-axis truncation
		rescaled_pdf = rescaled_pdf/10.
		rescaled_pdf[beta_pdf>1] = 1.
	return  r0 + r0*rescaled_pdf

def transform_rate_beta_rescaled_truncated_indicator(r0,prm,trait):
	alpha,beta,convex = prm
	if convex==1:
		beta_pdf = scipy.stats.beta.pdf(trait, alpha,beta)
		mode = (alpha-1)/((alpha+beta-2.))
		pdf_mode = scipy.stats.beta.pdf(mode, alpha,beta)
		rescaled_pdf = beta_pdf/pdf_mode
	else:
		beta_pdf = scipy.stats.beta.pdf(trait, 1/alpha,1/beta)
		rescaled_pdf = beta_pdf # Y-axis truncation
		#rescaled_pdf = rescaled_pdf/10.
		rescaled_pdf[beta_pdf>1] = 1.
	return  r0 + r0*rescaled_pdf

def transform_rate_normal(r0,prm,trait):
	mu,sd,sign = prm
	pdf_at_mode = 1./sqrt( 2*np.pi*(sd**2) )
	return abs(r0 * sign-(scipy.stats.norm.pdf(trait, loc=mu,scale=sd)/pdf_at_mode))

def transform_rate_logistic(r0,prm,trait):
	# r0 is the max rate
	x0,k = prm # mid point and steepness
	rate_at_trait = r0 / ( 1. + exp( -k * (trait-x0) )    )
	return rate_at_trait




runBETA=0
runLogistic=1
tranform_rate_func = tranform_rate_exp
if runLogistic: tranform_rate_func = transform_rate_logistic
#tranform_rate_func = tranform_rate_lin
#tranform_rate_func = transform_rate_normal
#transform_rate_beta_rescaled_truncated
if runBETA: 
	tranform_rate_func = transform_rate_beta_rescaled_truncated_unnormalized
	trait_list_of_arrays_rescaled = []
	# rescale trait data (so they range between 0 and 1)
	tr_waiting_times =[]
	tr_birth_events =[]
	tr_death_events =[]
	margin = 0.001 # rescaled so the never are exactly 0 or 1
	denom = (max(list_all_values)-min(list_all_values))
	for i in range(len(species_durations)):
		#species_trait_array_original = trait_list_of_arrays[i]- min(list_all_values)
		species_trait_array = trait_list_of_arrays[i]/(margin +denom)
		species_trait_array = species_trait_array-min(species_trait_array) # curve rescaled between 0 and 1
		#print min(species_trait_array),max(species_trait_array)
		trait_list_of_arrays_rescaled.append(species_trait_array)
		tr_waiting_times += list(species_trait_array) # all trait values
		tr_birth_events  += [species_trait_array[0]] # trait value at origination
		tr_death_events  += [species_trait_array[-1]] # trait value at extinction

#covert to array
tr_waiting_times = np.array(tr_waiting_times)
tr_birth_events  = np.array(tr_birth_events )
tr_death_events  = np.array(tr_death_events )[te>0]



# init params
l0A=np.array([0.2])
m0A=np.array([0.1])
alphaLA=np.array([0.])
alphaMA=np.array([0.])

if runBETA:
	alphaLA=np.array([1.1,1.1])
	alphaMA=np.array([1.1,1.1])
elif runLogistic:
	alphaLA=np.array([0,1.])
	alphaMA=np.array([0,1.])

#alphaLA=np.array([0.,1.,0])
#alphaMA=np.array([0.,1.,0])
out_dir= os.getcwd()
file_name = "test"
out_log = "%s/%s_mcmc.log" % (out_dir, file_name)
mcmc_logfile = open(out_log , "w",0) 
if runBETA:
	mcmc_logfile.write('\t'.join(["it","posterior","likelihood","prior","lambda_0","mu_0","alpha_l","beta_l","alpha_m","beta_m"])+'\n')
elif runLogistic:
	mcmc_logfile.write('\t'.join(["it","posterior","likelihood","prior","lambda_0","mu_0","x0_l","kappa_l","x0_m","kappa_m"])+'\n')


Gamma_shape,Gamma_rate = 1.,1.
priorA =  prior_gamma(l0A,Gamma_shape,Gamma_rate) + prior_gamma(m0A,Gamma_shape,Gamma_rate)
#priorA += prior_normal(log(alphaLA),.1) + prior_normal(log(alphaMA),.1)

likA = get_likelihood_continuous_trait(l0A,m0A,alphaLA,alphaMA,tranform_rate_func)

print min(tranform_rate_func(m0A,alphaMA,tr_waiting_times)),max(tranform_rate_func(m0A,alphaMA,tr_waiting_times))


x= tranform_rate_func(l0A,alphaLA,tr_waiting_times)
print x, l0A,alphaLA
#np.argwhere(np.isnan(x)), tr_waiting_times[np.argwhere(np.isnan(x))]

#quit()

print likA,priorA

n_iterations=100000
iteration = 0
while iteration < n_iterations:
	l0,m0,alphaL,alphaM = l0A+0,m0A+0,alphaLA+0,alphaMA+0
	hasting = 0 
	rr = np.random.random(3)
	if rr[0] < 0.2:
		l0,hasting= update_multiplier_freq(l0A,f=1)
	elif rr[0] < 0.4:
		m0,hasting= update_multiplier_freq(m0A,f=1)
	else:
		
		if runBETA:
			if rr[2]<0.5:
				alphaL,hasting= update_multiplier_freq(alphaLA,d=1.05,f=1)
			else:
				alphaM,hasting= update_multiplier_freq(alphaMA,d=1.05,f=1)
		elif runLogistic:
			d_win = np.array([0.5,0.1])
			if rr[2]<0.5:
				alphaL= update_sliding_win_unbounded_vec(alphaLA,d=d_win)
			else:
				alphaM= update_sliding_win_unbounded_vec(alphaMA,d=d_win)
			
		else:
			alphaL= np.abs(update_sliding_win_unbounded_vec(alphaLA,d=0.1))
			alphaM= np.abs(update_sliding_win_unbounded_vec(alphaMA,d=0.1))
		
		
			
	prior =  prior_gamma(l0,Gamma_shape,Gamma_rate) + prior_gamma(m0,Gamma_shape,Gamma_rate)	
	#prior += prior_normal(log(alphaL),.5) + prior_normal(log(alphaM),.5)
	if runBETA==2:
		if min(alphaL)< 1 or min(alphaM)< 1: 
			prior = -np.inf
			lik = -np.inf
		else:
			lik = get_likelihood_continuous_trait(l0,m0,alphaL,alphaM,tranform_rate_func)
	
	else:
		lik = get_likelihood_continuous_trait(l0,m0,alphaL,alphaM,tranform_rate_func)
		
	
	if lik-likA + prior-priorA + hasting >= log(np.random.random()):
		likA=lik
		priorA= prior
		l0A= l0
		m0A= m0
		alphaLA= alphaL
		alphaMA= alphaM
		
	if iteration % 100==0: 
		print iteration,likA,lik,l0A,m0A,alphaLA,alphaMA #,convexA
		log_state = map(str,[iteration,likA+priorA,likA,priorA]+list(l0A)+list(m0A)+list(alphaLA)+list(alphaMA))
		mcmc_logfile.write('\t'.join(log_state)+'\n')
		mcmc_logfile.flush()
		
	iteration+=1
	






quit()




out_dir= os.path.dirname(f)

print out_dir
if out_dir=="": 
	out_dir= os.getcwd()
file_name = os.path.splitext(os.path.basename(f))[0]

####### MCMC log files #######
out_dir = "%s/pyrate_mcmc_logs" % (out_dir)
try: os.mkdir(out_dir) 
except: pass

out_log = "%s/%s_mcmc.log" % (out_dir, file_name)
mcmc_logfile = open(out_log , "w",0) 
mcmc_logfile.write('\t'.join(["it","posterior","likelihood","prior","lambda_avg","mu_avg","K_l","K_m","root_age","death_age","gamma_rate_hp","poisson_rate_hp"])+'\n')
out_log = "%s/%s_sp_rates.log" % (out_dir, file_name)
sp_logfile = open(out_log , "w",0) 
out_log = "%s/%s_ex_rates.log" % (out_dir, file_name)
ex_logfile = open(out_log , "w",0) 

####### PRECOMPUTE VECTORS #######
sp_events_bin = []
ex_events_bin = []
br_length_bin = []
bins = np.arange(min_time,max_time+1)[::-1]
for i in range(len(bins)-1):
	a,b,c = precompute_events([bins[i],bins[i+1]])
	sp_events_bin.append(a)
	ex_events_bin.append(b)
	br_length_bin.append(c)

sp_events_bin = np.array(sp_events_bin)
ex_events_bin = np.array(ex_events_bin)
br_length_bin = np.array(br_length_bin)

# remove first bin
sp_events_bin = sp_events_bin[1:]
ex_events_bin = ex_events_bin[1:]
br_length_bin = br_length_bin[1:]
max_time -= 1


n_bins = len(sp_events_bin)

####### init parameters #######
L_acc= np.random.gamma(2,2,1)
M_acc= np.random.gamma(2,2,1)
timesLA = np.array([max_time, min_time])
timesMA = np.array([max_time, min_time])

####### GLOBAL variables #######
min_allowed_t = 1   # minimum allowed distance between shifts (to avoid numerical issues)
Gamma_shape = 2.    # shape parameter of Gamma prior on B/D rates
hpGamma_shape = 1.2 # shape par of Gamma hyperprior on rate of Gamma priors on B/D rates
hpGamma_rate =  0.1 # rate par of Gamma hyperprior on rate of Gamma priors on B/D rates
rev_bins = bins[::-1]+0.1

check_lik = 0 # debug (set to 1 to compare vectorized likelihood against 'traditional' one)
runMCMC([L_acc,M_acc,timesLA,timesMA])