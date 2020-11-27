import os
import numpy as np
import pickle
from math import log, pi
from tools import emcee_perrakis as perry
import celerite
from celerite import terms

import matplotlib.pyplot as pl

import sys
sys.path.append("/Users/sbarros/Documents/work/python/photodynamic/lisa/")

from source.tools.emcee_tools import geweke_multi, geweke_selection, geweke_plot, plot_chains


from  modulo1Knorm import lnlike_gp, lnprior_gp, lnprob_gp



if __name__ == "__main__":

    #HD_179079_N2_cele3kr1.pkl

    filenames = ['HD_179079.pkl', 'HD_42618_LRa04.pkl', 'HD_42618_LRa05.pkl', 'HD_43587_N2.pkl', 'HD_49933IRa01.pkl', 'HD_49933LRa01.pkl', 'HD_52265.pkl']

    name = ['HD_179079', 'HD_42618_LRa04', 'HD_42618_LRa05', 'HD_43587_N2', 'HD_49933IRa01', 'HD_49933LRa01', 'HD_52265']

    #for ll in range(0, len(filenames)):
    for ll in range(0, 1):

        sampler = pickle.load(open("issi_cele32/K1norm1"+filenames[ll], 'rb'))

        zscores, first_steps = geweke_multi(sampler.chain, 0.1, 0.5, 20)

        # geweke_plot(zscores, first_steps=first_steps, l_param_name=None)

        l_burnin, l_walker = geweke_selection(zscores, first_steps)

        '''
        plot_chains(sampler.chain, sampler.lnprobability, l_param_name=None, l_walker=l_walker, l_burnin=l_burnin)
        pl.savefig("figures32/K1norm"+name[ll]+"pl.png", dpi=150)
        pl.close()
        '''
        burnin = max(l_burnin)
        print(name[ll])
        print( "number l_walker", "max burning")
        print( len(l_walker), burnin)
        #new_chain = sampler.chain[l_walker, burnin:, :]

        #sampler.chain = new_chain

        #print(new_chain.shape)

        cind = [ii in l_walker for ii in range(sampler.chain.shape[0])]

        ntrys = 2#400

        #result =  perry(sampler, nsamples=500, bi=burnin,cind=cind)


        result = np.zeros((2, ntrys))

        for ii in range(ntrys):
            result[:,ii] = perry(sampler, nsamples=3000, bi=burnin,cind=cind)
            if ii % 100 == 0: print(ii)


        #print(result)
        for i in range(ntrys):
            print(result[0,i], result[1,i] )

        '''
        fp = open("results32/resK1norm_"+name[ll]+".txt", 'w')
        for i in range(0,ntrys):
            fp.write('%.20f\t%.20f\n'%(result[0,i], result[1,i] ))
        fp.close()
        '''
    # bi is the beguining  (in case we want to cut the burin-in
    # cind are the chain indices to use,  chainindexes must be boolean
