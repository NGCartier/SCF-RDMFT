#ifndef _MULLER_FUNC_hpp_
#define _MULLER_FUNC_hpp_
//see Muller.cpp
class Functional; class RDM1;
VectorXd Muller_fJ(RDM1*); VectorXd Muller_gJ(RDM1*); VectorXd Muller_dfJ(RDM1*); VectorXd Muller_dgJ(RDM1*);
VectorXd Muller_fK(RDM1*); VectorXd Muller_gK(RDM1*); VectorXd Muller_dfK(RDM1*); VectorXd Muller_dgK(RDM1*);

//Definition of the Muller functional
static Functional Muller_func = Functional(Muller_fJ, Muller_gJ, Muller_fK, Muller_gK, Muller_dfJ, Muller_dgJ, Muller_dfK, Muller_dgK);
#endif