#ifndef _PNOF7_FUNC_hpp_
#define _PNOF7_FUNC_hpp_

class Functional; class RDM1;
//MatrixXd PNOF7_EK(RDM1*); VectorXd PNOF7_dEK(RDM1*); SLOW IMPLEMENTATION - USED TO TEST

MatrixXd PNOF7_WK(RDM1*); VectorXd PNOF7_dWK(RDM1*); VectorXd PNOF7_dWK_subspace(RDM1*,int); 
MatrixXd PNOF7_old_WK(RDM1*); VectorXd PNOF7_old_dWK(RDM1*); VectorXd PNOF7_old_dWK_subspace(RDM1*,int); 
static Functional PNOF7_func = Functional(PNOF7_WK, PNOF7_dWK, PNOF7_dWK_subspace, true);

//Version with outdated phase in inter-subspace energy
static Functional PNOF7_old_func = Functional(PNOF7_old_WK, PNOF7_old_dWK, PNOF7_old_dWK_subspace, true);
#endif
