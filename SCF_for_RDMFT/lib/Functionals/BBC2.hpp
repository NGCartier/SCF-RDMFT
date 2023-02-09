#ifndef _BBC2_FUNC_hpp_
#define _BBC2_FUNC_hpp_

class Functional; class RDM1;
MatrixXd BBC2_WK (RDM1*); VectorXd BBC2_dWK(RDM1*);
static Functional BBC2_func = Functional(BBC2_WK,BBC2_dWK);

#endif