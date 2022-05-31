#ifndef _1RDM_CLASS_hpp_
#define _1RDM_CLASS_hpp_
#include "Functional_class.hpp"
#include<tuple>
using namespace std;
class Functional;
// Defines the 1RDM class (see 1RDM_class.cpp for more detail)
class RDM1{
    public:
        int n_elec; 
        VectorXd n; MatrixXd no;
        double E_nuc; MatrixXd ovlp; 
        void set_n (VectorXd n0) {n = n0; };
        void set_no(MatrixXd no0){no=no0; };
        MatrixXd int1e; MatrixXd int2e; MatrixXd int2e_x;
        RDM1(int,double,MatrixXd,MatrixXd,MatrixXd,MatrixXd);
        RDM1(VectorXd,MatrixXd,int,double,MatrixXd,MatrixXd,MatrixXd,MatrixXd);
        RDM1(const RDM1*);
        ~RDM1(){};
        MatrixXd mat() const;
        void opti(Functional,int disp=0,double epsi=1e-6, double epsi_n=1e-3, double epsi_no=1e-3,int maxiter=100);
};
//Auxiliary functions used to minimise the energy of the 1RDM
double norm2(VectorXd x); double norm1(VectorXd x);
tuple<double,int> opti_n(RDM1*,Functional,double epsilon=1e-8,double eta=1e-12,bool disp=false, int maxiter=100);
MatrixXd exp_unit(VectorXd);
tuple<double,int> opti_no(RDM1*,Functional,double epsilon=1e-8,bool disp=false, int maxiter=100);


void print_t(chrono::high_resolution_clock::time_point, chrono::high_resolution_clock::time_point, int iter=1);
#endif