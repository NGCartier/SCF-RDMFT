#ifndef _HF_FUNC_hpp_
#define _HF_FUNC_hpp_
class Functional; class RDM1;
MatrixXd HF_WK (RDM1*); VectorXd HF_dWK (RDM1*);
static Functional HF_func = Functional(HF_WK,HF_dWK);
#endif