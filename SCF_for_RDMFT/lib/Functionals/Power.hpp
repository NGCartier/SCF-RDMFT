#ifndef _POWER_FUNC_hpp_
#define _POWER_FUNC_hpp_
class Functional; class RDM1;
MatrixXd Power_WK(RDM1*); VectorXd Power_dWK(RDM1*); 
static Functional Power_func = Functional(Power_WK, Power_dWK);
#endif
