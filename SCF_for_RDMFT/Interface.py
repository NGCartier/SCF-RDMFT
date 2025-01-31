#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:46:35 2022

@author: nicolascartier
"""

import numpy as np
from numpy.typing import ArrayLike
from  pyscf import gto, scf, ao2mo
import scipy as sc
from scipy import optimize as opt 
import Compute_1RDM

def compute_1RDM(nsites=2, nelec=2, t=None, v_ext=None, U=1., func="Muller", disp=0, epsi = 1e-6, Maxiter=100):
    """
    Compute the 1RDM minimising the enrgy (functional func) of a closed shell
    Hubbard model at half filling
    
    Parameters
    ----------
    nsites : int, optional
        Number of sites in the model. Default is 2
    nelec  : int, optional
        Number of electrons in the system. Default is 2
    t : ArrayLike, optional
        Kinetic energy. By default takes a Hubbard ring with t=1
    v_ext : ArrayLike, optional
        (local) External potential. By default takes v=0
    U : float, optional
        One site electron interaction. Default is 1.
    func : string, optional
        Functional to use. The default is "Muller".
    
    Returns
    -------
    E : float
        Enegy of the groud state.
    occ : array_like
        Array of the occupations of the state in natural orbital basis
    NO : matrix_like
        Matrix to go from natural to atomic orbitals   
    """
    if nelec!=nsites or nelec%2!=0:
        raise ValueError("Expect a system at half filling with even number of electrons.")
    model = gto.M()
    model.nelectron = nelec
    model.nao = nsites
    H1 = np.zeros((nsites,nsites))
    H2 = np.zeros((nsites,nsites,nsites,nsites))
    k=0
    for i in range(nsites):
        H2[i,i,i,i] = U
        if v_ext is not None:
            H1[i,i] = v_ext[i]
        if t is None:
            H1[i,(i+1)%nsites] = -1.; H1[(i+1)%nsites,i] =-1.
        else: 
            for j in range(i):
                H1[i,j] = -t[k]; H1[j,i] = -t[k]
                k+=1
    
    mf    = scf.RHF(model)
    mf.nao= nsites 
    ovlp  = np.eye(nsites)
    mf.get_hcore = lambda *args: H1
    mf.get_ovlp  = lambda *args: ovlp
    mf._eri = ao2mo.restore(8,H2,nsites)

    n,no = rdm_guess(mf,model,H1,H2)
    E,n,no = Compute_1RDM.Optimize_1RDM(func, n, no, nelec, 0, ovlp, H1, 
                                      H2.reshape(nsites**2,nsites**2),
                                      disp, epsi, Maxiter)
    return E,n**2,no

def rdm_guess (mf,model,H1,h2,beta=0.4):
    '''Returns the 1RDM of the mol molecule using HF orbitals and Fermi-Dirac distribtution'''
    def FD_occ (E,i, mu):  return 2/(1+np.exp(beta*(E[i]-mu ) ) )

    E = mf.kernel() 
    rdm1 = mf.make_rdm1()
    n,no  = sc.linalg.eigh(rdm1)
    H2 = np.zeros((model.nao,model.nao,model.nao,model.nao))
    for i in range(model.nao):
        for j in range(model.nao):
            for k in range(model.nao):
                for l in range(model.nao):
                    H2[i,j,k,l]= h2[i,j,k,l] -h2[i,l,k,j]
    H = H1+1/2*np.tensordot(H2,rdm1, axes=([2,3],[1,0]) )
    E,_ = sc.linalg.eigh(H)
    
    def Eq_n(mu):
        res = 0
        for i in range(model.nao):
            res += FD_occ(E,i,mu)
        return res - model.nelectron
    mu = opt.fsolve(Eq_n, 0)
    n = np.zeros(mf.nao)
    
    for i in range (model.nao):
        n[i] = FD_occ(E,i, mu)
            
    return (np.sqrt(n), no)

