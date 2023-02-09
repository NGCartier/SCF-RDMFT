#ifndef _HF_FUNC_hpp_
#define _HF_FUNC_hpp_

class Functional; class RDM1;
MatrixXd HF_WK (RDM1*); VectorXd HF_dWK (RDM1*);
static Functional HF_func = Functional(HF_WK,HF_dWK);
MatrixXd H_WK (RDM1*); VectorXd H_dWK (RDM1*);
static Functional Hartree_func = Functional(H_WK,H_dWK);
MatrixXd E1_WK (RDM1*); VectorXd E1_dWK (RDM1*);
static Functional E1_func = Functional(E1_WK,E1_dWK,true);
#endif