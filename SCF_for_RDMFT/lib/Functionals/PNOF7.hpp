#ifndef _PNOF7_FUNC_hpp_
#define _PNOF7_FUNC_hpp_

class Functional; class RDM1;
MatrixXd PNOF7_WK(RDM1*); VectorXd PNOF7_dWK(RDM1*); VectorXd PNOF7_dWK_subspace(RDM1*,int); 
static Functional PNOF7_func = Functional(PNOF7_WK, PNOF7_dWK, true, PNOF7_dWK_subspace);

#endif
