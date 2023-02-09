#ifndef _BBC1_FUNC_hpp_
#define _BBC1_FUNC_hpp_

class Functional; class RDM1;
MatrixXd BBC1_WK (RDM1*); VectorXd BBC1_dWK(RDM1*);
static Functional BBC1_func = Functional(BBC1_WK,BBC1_dWK);

#endif