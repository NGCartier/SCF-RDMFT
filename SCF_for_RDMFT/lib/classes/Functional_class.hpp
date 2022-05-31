#ifndef _FUNCTIONAL_CLASS_hpp_
#define _FUNCTIONAL_CLASS_hpp_

#include "1RDM_class.hpp"
using namespace std;
using namespace Eigen;

class RDM1;
//Definition of the Functional class (for more details see Functional_class.cpp)
class Functional{
    private:
        VectorXd (* f_J)(RDM1*);
        VectorXd (* g_J)(RDM1*);
        VectorXd (* f_K)(RDM1*);
        VectorXd (* g_K)(RDM1*);
        VectorXd (* df_J)(RDM1*);
        VectorXd (* dg_J)(RDM1*);
        VectorXd (* df_K)(RDM1*);
        VectorXd (* dg_K)(RDM1*);
    public:
        Functional(VectorXd (*)(RDM1*), VectorXd (*)(RDM1*), VectorXd (*)(RDM1*), VectorXd (*)(RDM1*), VectorXd (*)(RDM1*), VectorXd (*)(RDM1*), VectorXd (*)(RDM1*), VectorXd (*)(RDM1*));
        ~Functional() {};
        double E(RDM1*) const; VectorXd grad_E(RDM1*,bool only_n=false,bool only_no=false) const; 
        double E(RDM1*, MatrixXd*, MatrixXd*) const; VectorXd grad_E(RDM1*, MatrixXd*, MatrixXd*, MatrixXd*, MatrixXd*, MatrixXd*, MatrixXd*,
                        bool only_n=false, bool only_no=false) const;
        VectorXd dE_Hxc(RDM1*, bool only_n=false, bool only_no=false) const;
        VectorXd dE_Hxc(RDM1*, MatrixXd*, MatrixXd* W_K, MatrixXd*, MatrixXd*, MatrixXd*, MatrixXd*, bool only_n=false, bool only_no=false) const;
        MatrixXd compute_vJ_f(RDM1*) const; MatrixXd compute_vJ_g(RDM1*) const; MatrixXd compute_vK_f(RDM1*) const; MatrixXd compute_vK_g(RDM1*) const; 
        MatrixXd compute_WJ(RDM1*) const; MatrixXd compute_WK(RDM1*) const; MatrixXd compute_WJ(RDM1*, MatrixXd*, MatrixXd*) const; MatrixXd compute_WK(RDM1*, MatrixXd*, MatrixXd* ) const;
        double E_Hxc(RDM1*) const; double E_Hxc(MatrixXd*, MatrixXd*) const;
        
};
//Auxiliary functions to compute the energy
double E1(RDM1*); double compute_E1(MatrixXd*, MatrixXd); VectorXd dE1(RDM1*, bool only_n= false, bool only_no= false);
MatrixXd dU(MatrixXd*,int,int); MatrixXd dN(VectorXd*, MatrixXd*, int);
VectorXd pow(VectorXd, double);

#endif