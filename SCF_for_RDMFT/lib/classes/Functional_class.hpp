#ifndef _FUNCTIONAL_CLASS_hpp_
#define _FUNCTIONAL_CLASS_hpp_

#include "1RDM_class.hpp"
using namespace std;
using namespace Eigen;

class RDM1;


VectorXd None (RDM1*, int);
//Definition of the Functional class (for more details see Functional_class.cpp)
class Functional{
    private:
        bool is_J_func_;                       //Wether the direct term is included in the functional
        MatrixXd (*W_K_) (RDM1*);              //The matrix of E_xc (or E_Hxc if is_J_func_=true) in NO basis.
        VectorXd (*dW_K_)(RDM1*);              //The derivative of W_K_
        VectorXd (*dW_K_subspace_)(RDM1*,int); //W_K_ restricted to one subspace of PNOF omega (for occ only).
    public:
        Functional(MatrixXd(*)(RDM1*), VectorXd(*)(RDM1*), VectorXd (*dW_K_subspace_)(RDM1*,int) = None, bool is_J_func = false);
        bool needs_subspace() const;
        bool operator==(const Functional&); 
        double E(RDM1*) const; VectorXd grad_E(RDM1*,bool only_n=false,bool only_no=false) const; 
        double E(RDM1*, MatrixXd*, MatrixXd*) const; VectorXd grad_E(RDM1*, MatrixXd*, MatrixXd*, bool only_n=false, bool only_no=false) const;
        VectorXd grad_E_subspace(RDM1*, int) const;
        VectorXd dE_Hxc(RDM1*, bool only_n=false, bool only_no=false) const;
        VectorXd dE_Hxc(RDM1*, MatrixXd*, MatrixXd*, bool only_n=false, bool only_no=false) const;
        VectorXd dE_Hxc_subspace(RDM1*, int) const;
        MatrixXd compute_WJ(RDM1*) const;   MatrixXd compute_WK(RDM1*) const; 
        VectorXd compute_dW_J(RDM1*) const; VectorXd compute_dW_K(RDM1*) const;
        double E_Hxc(MatrixXd*, MatrixXd*) const;
        
};
//Auxiliary functions to compute the energy
double E1(RDM1*); double compute_E1(MatrixXd*, MatrixXd*); VectorXd dE1(RDM1*, bool only_n= false, bool only_no= false);
MatrixXd v_J(RDM1*,VectorXd*); MatrixXd v_K(RDM1*,VectorXd*); 

MatrixXd dU(MatrixXd*,int,int); MatrixXd dN(VectorXd*, MatrixXd*, int);
VectorXd pow(const VectorXd*, double);

#endif