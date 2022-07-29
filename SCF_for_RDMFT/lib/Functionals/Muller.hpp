#ifndef _MULLER_FUNC_hpp_
#define _MULLER_FUNC_hpp_

class Functional; class RDM1;
MatrixXd Muller_WK(RDM1*); VectorXd Muller_dWK(RDM1*); 
static Functional Muller_func = Functional(Muller_WK, Muller_dWK);
#endif