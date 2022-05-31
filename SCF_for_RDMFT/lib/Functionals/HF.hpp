#ifndef _HF_FUNC_hpp_
#define _HF_FUNC_hpp_
//see HF_func.cpp
class Functional; class RDM1;
VectorXd HF_fJ(RDM1*); VectorXd HF_gJ(RDM1*); VectorXd HF_dfJ(RDM1*); VectorXd HF_dgJ(RDM1*);
VectorXd HF_fK(RDM1*); VectorXd HF_gK(RDM1*); VectorXd HF_dfK(RDM1*); VectorXd HF_dgK(RDM1*);

//Definition of the Hartree Fock functional
static Functional HF_func = Functional(HF_fJ, HF_gJ, HF_fK, HF_gK, HF_dfJ, HF_dgJ, HF_dfK, HF_dgK);
#endif