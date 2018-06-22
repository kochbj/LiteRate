#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:57:01 2018

@author: bernie
"""
import pandas as pd; import numpy as np
from sys import argv, stdout
import argparse

p = argparse.ArgumentParser(description='This script is used to convert a standard literate input format (clade, species, ts, te) to trait format using age.') #description='<input file>') 
p.add_argument('-t', type=str, help='age since species (s) origin or clade (c)?', default="s", metavar="t",choices=['s','c'])
p.add_argument('-o', type=argparse.FileType('w'), default='-', help='output file')
p.add_argument('input', metavar='f', type=str, help='input file in standard literate format')
args = p.parse_args()

df=pd.read_csv(args.input,sep='\t')
#df=pd.read_csv('~/Documents/Github/LiteRate/example_dataTAD.txt',sep='\t')
min_ts=df.ts.min()
max_te=df.te.max()

years=['clade','species','ts','te',]+list(map(str,list(range(min_ts,max_te+1))))

args.o.write('\t'.join(years)+'\n')

def create_outrow(row,outfile,age_type,min_ts,max_te):
    if age_type=='s':
        age = list(range(0,int(row.te-row.ts+1)))
    else:
        age = list(range(int(row.ts-clade_age[row.clade]), int(row.te-clade_age[row.clade])+1))
        print(age,int(row.te-clade_age[row.clade]))
    species_trait_array= [int(row[0]),int(row[1]),int(row.ts),int(row.te)]+ ["NA"]*int(row.ts-min_ts) + age + ["NA"]*int(max_te-row.te)
    species_trait_array = list(map(str,species_trait_array))
    outfile.write('\t'.join(species_trait_array)+'\n')

clade_age=df.groupby('clade').min().ts    

df.apply(lambda row: create_outrow(row,args.o,args.t,min_ts,max_te),axis=1)
        
