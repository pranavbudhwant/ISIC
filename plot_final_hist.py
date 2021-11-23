# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:49:24 2020

@author: prnvb
"""

import joblib
import matplotlib.pyplot as plt 
import seaborn as sns

match = joblib.load('final_results/recon_errors/v2densenet-0.8-04-isic_val-match_recon.pkl')
nonmatch = joblib.load('final_results/recon_errors/v2densenet-0.8-04-isic_val-nonmatch_recon.pkl')

known = joblib.load('final_results/recon_errors/derm_known.pkl')
unknown = joblib.load('final_results/recon_errors/derm_unknown.pkl')

clinic_known = joblib.load('final_results/recon_errors/derm_clinic_known.pkl')
clinic_unknown = joblib.load('final_results/recon_errors/derm_clinic_unknown.pkl')

f, axes = plt.subplots(1, 2)
f.set_size_inches((12,6))
sns.distplot(match,hist=True,norm_hist=True,label='Match',ax=axes[0])
sns.distplot(nonmatch,hist=True,norm_hist=True,label='Non-match',ax=axes[0])
axes[0].set_xlabel('Reconstruction Errors',fontsize=15)
axes[0].set_ylabel('Normalized Histgram',fontsize=15)
axes[0].legend(loc='best',prop={'size': 15})
sns.distplot(known,hist=True,norm_hist=True,label='Known',color='green',ax=axes[1])
sns.distplot(unknown,hist=True,norm_hist=True,label='Unknown',color='red',ax=axes[1])
axes[1].legend(loc='best',prop={'size': 15})
axes[1].set_xlabel('Reconstruction Errors',fontsize=15)
axes[1].set_ylabel('Normalized Histgram',fontsize=15)
f.savefig('final_results/recon_errors/recon.pdf',bbox_inches='tight')


f = plt.figure()
sns.distplot(match,hist=True,norm_hist=True,label='Match',bins=100)
sns.distplot(nonmatch,hist=True,norm_hist=True,label='Non-match',bins=100)
plt.legend(loc='best',prop={'size': 15})
plt.xlabel('Reconstruction Errors',fontsize=15)
plt.ylabel('Normalized Histgram',fontsize=15)
f.savefig('final_results/recon_errors/mnm.pdf',bbox_inches='tight')


f = plt.figure()
sns.distplot(known,hist=True,norm_hist=True,label='Known',color='green')
sns.distplot(unknown,hist=True,norm_hist=True,label='Unknown',color='red')
sns.distplot(clinic_unknown,hist=True,norm_hist=True,label='Clinical Unknown',
             color='lavender')
plt.legend(loc='best',prop={'size': 15})
plt.xlabel('Reconstruction Errors',fontsize=15)
plt.ylabel('Normalized Histgram',fontsize=15)
f.savefig('final_results/recon_errors/kuk.pdf',bbox_inches='tight')





x=plt.hist(match,bins=50,color='orange',alpha=0.6,label='Match Reconstruction Errors')
plt.hist(nonmatch,bins=50,color='blue',alpha=0.6,label='Non-match Reconstruction Errors')
plt.legend(loc='best')

f = plt.figure()
plt.hist(known,bins=50,color='red',alpha=0.7,
         label='Known',normed=True,
         histtype='bar')
plt.hist(unknown,bins=50,color='green',alpha=0.7,
         label='Unknown',normed=True,
         histtype='bar')
plt.legend(loc='best',prop={'size': 12})
plt.xlabel('Reconstruction Errors')
plt.ylabel('Normalized Histgram')
f.savefig('final_results/recon_errors/kuk.pdf',bbox_inches='tight')

f = plt.figure()
plt.hist(match,bins=50,color='orange',alpha=0.7,
         label='Match',normed=True,
         histtype='bar')
plt.hist(nonmatch,bins=50,color='blue',alpha=0.7,
         label='Non-match',normed=True,
         histtype='bar')
plt.legend(loc='best',prop={'size': 12})
plt.xlabel('Reconstruction Errors')
plt.ylabel('Normalized Histgram')
f.savefig('final_results/recon_errors/mnm.pdf',bbox_inches='tight')


import pandas as pd

recon_errors = pd.DataFrame()