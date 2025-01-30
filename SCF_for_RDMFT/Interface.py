#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:46:35 2022

@author: nicolascartier
"""

import numpy as np
from  pyscf import gto, scf, cc, ci, mp
import scipy as sc
from scipy import optimize as opt 
import Compute_1RDM

def compute_1RDM(mol, guess="CISD", func="Muller", disp=0, epsi = 1e-6, Maxiter=100 ):
    """
    Parameters
    ----------
    mol : Molecule class (from PySCF)
        molecule to optimize.
    guess : string or None, optional
        Initial guess method. The default is "CISD".
        If None initialises with NO = Id, occ = [2,2,...,1]
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
    if guess=="HF":
        n, no, ne, Enuc, overlap, elec1int, elec2int  = rdm_guess(mol)
    elif guess=="CISD":
        n, no, ne, Enuc, overlap, elec1int, elec2int =  rdm_guess_CISD(mol)
    elif guess=="uniform":
        #to test problematic case
        n, no, ne, Enuc, overlap, elec1int, elec2int =  rdm_guess_uniform(mol)
    else :
        n, no, ne, Enuc, overlap, elec1int, elec2int = No_init(mol)
    l = len(n)
    return Compute_1RDM.Optimize_1RDM(func, n, no, ne, Enuc, overlap, elec1int, elec2int,
                                disp, epsi, Maxiter)
        
def E(mol, guess="CISD", func="Muller", disp=0, epsi = 1e-6, Maxiter=100 ):
   
    if guess=="HF":
        n, no, ne, Enuc, overlap, elec1int, elec2int  = rdm_guess(mol)
    elif guess=="CISD":
        n, no, ne, Enuc, overlap, elec1int, elec2int =  rdm_guess_CISD(mol)
    elif guess=="uniform":
        #to test problematic case
        n, no, ne, Enuc, overlap, elec1int, elec2int =  rdm_guess_uniform(mol)
    else :
        n, no, ne, Enuc, overlap, elec1int, elec2int = No_init(mol)
    l = len(n)
    return Compute_1RDM.E(func, n, no, ne, Enuc, overlap, elec1int, elec2int)


def rdm_guess (mol,  beta=1.6):
    '''Returns the 1RDM of the mol molecule using HF orbitals and Fermi-Dirac distribtution'''
    def FD_occ (E,i, mu):  return 2/(1+np.exp(-beta*(E[i]-mu ) ) )

    mf = scf.RHF(mol) 
    #mf = scf.addons.frac_occ(mf)
    E = mf.kernel() 
    rdm1 = mf.make_rdm1()
    # /!\ for HF make_rdm1 returns the 1RDm in AO basis (!= methodes that give it in NO)
    srS = np.real( sc.linalg.sqrtm( mol.intor('int1e_ovlp') ) )
    iS = np.linalg.inv(srS)   
    gamma = srS@rdm1@srS 
    gamma = np.around(gamma, 12) #control numerical error
    n,No  = sc.linalg.eigh(gamma)
    No = iS@No
    
    H1 = mol.intor('int1e_nuc')+mol.intor('int1e_kin')  
    v  = mol.intor('int2e')
    H2 = np.zeros((mol.nao, mol.nao, mol.nao, mol.nao))
    for i in range(mol.nao):
        for j in range(mol.nao):
            for k in range(mol.nao):
                for l in range(mol.nao):
                    H2[i,j,k,l]= v[i,j,k,l] -v[i,l,k,j]
    H = H1+1/2*np.tensordot(H2,gamma, axes=([2,3],[1,0]) )
    E,_ = sc.linalg.eigh(H)
    
    def Eq_n(mu):
        res = 0
        for i in range(mol.nao):
            res += FD_occ(E,i,mu)
        return res - mol.nelectron
    mu = opt.fsolve(Eq_n, 0)
    n = np.zeros(mol.nao)
    
    for i in range (mol.nao):
        n[i] = FD_occ(E,i, mu)
            
    l = len(n)
    return (np.sqrt(n), No,
            mol.nelectron, mol.energy_nuc(), mol.intor('int1e_ovlp'), 
            mol.intor('int1e_nuc').copy()+mol.intor('int1e_kin').copy(),
            mol.intor('int2e').reshape(l**2,l**2) )


def rdm_guess_CISD(mol):
    '''Return 1RDM for molecule mol using CISD'''
    mf = scf.ROHF(mol)
    #mf = scf.addons.frac_occ(mf)
    mf.kernel()
    mf.mo_coeff
    ci_mol = ci.cisd.CISD(mf)
    _, civec = ci_mol.kernel()
    rdm1_MO = ci_mol.make_rdm1()
    rdm1 = mf.mo_coeff@rdm1_MO@mf.mo_coeff.T
    
    sqS = np.real( sc.linalg.sqrtm( mol.intor('int1e_ovlp') ) )
    iS = np.linalg.inv(sqS)
      
    gamma = sqS@rdm1@sqS 
    n,No  = np.linalg.eigh(gamma)
    No = iS@No
    l = len(n)
    return (np.sqrt(n), No,
            mol.nelectron, mol.energy_nuc(), mol.intor('int1e_ovlp'), 
            mol.intor('int1e_nuc').copy()+mol.intor('int1e_kin').copy(),
            mol.intor('int2e').reshape(l**2,l**2) )

def No_init(mol,epsi=1e-6):
    l = mol.nao
    S = mol.intor('int1e_ovlp')
    No = np.real(sc.linalg.sqrtm( np.linalg.inv( S ) ) )
    ne = mol.nelectron
    if ne > l:
        n = np.full(l,1.+epsi)
        n[range(2*l-ne,l)] = np.sqrt(2)
    else:
        n = np.zeros(l)
        n[range(l-ne,l)] = 1+epsi
    
    return (n, No,
            mol.nelectron, mol.energy_nuc(), S, 
            mol.intor('int1e_nuc').copy()+mol.intor('int1e_kin').copy(),
            mol.intor('int2e').reshape(l**2,l**2) )

def rdm_guess_uniform(mol,epsi=1e-5):
    l = mol.nao
    S = mol.intor('int1e_ovlp')
    No = np.real(sc.linalg.sqrtm( np.linalg.inv( S ) ) )
    ne = mol.nelectron
    n = np.array([ne/l+np.random.uniform(-epsi,epsi) for i in range(l)])
    return (n, No,
            mol.nelectron, mol.energy_nuc(), S, 
            mol.intor('int1e_nuc').copy()+mol.intor('int1e_kin').copy(),
            mol.intor('int2e').reshape(l**2,l**2) )
    
