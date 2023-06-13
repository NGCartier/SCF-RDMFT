#ifndef _1RDM_CLASS_hpp_
#define _1RDM_CLASS_hpp_
#include "Functional_class.hpp"
#include<tuple>
#include <vector>
using namespace std;

class Functional;
;

// Defines the 1RDM class (see 1RDM_class.cpp for more detail)
class RDM1{
    private:
        VectorXd V_;                           //Sum of derivatives of erf (used to compute derivatives ; vector if subspaces)
        VectorXi computed_V_;                  //Wether V is up to date  (vector if subspaces)
        VectorXd x_;                           //Vector parametrising the occupations 
    public:
        int n_elec;                            //Number of electrons
        MatrixXd no;                           //Matrix of natural orbitals
        double E_nuc;                          //Nuclear energy
        MatrixXd ovlp;                         //Overlap matrix
        VectorXd mu;                    //EBI variable (vector if subspaces)
        vector<vector<int>> omega;             //Indicies of the subspace repartition
        void set_n (VectorXd);
        void set_n(int,double); 
        void set_no(MatrixXd no0){no=no0; };  // Affect no0 to no
        double get_V(int g=0);
        void solve_mu();
        void subspace();
        double n(int) const;
        VectorXd n() const;
        double x(int i) const { return x_(i);};// Return x_i
        VectorXd x() const { return x_;};      // Return x
        void x (int, double);
        void x (VectorXd);
        int size() const { return x_.size(); };//Return the number of NO
        double dn(int,int); MatrixXd dn(int);
        double dsqrt_n(int,int); MatrixXd dsqrt_n(int);
        MatrixXd int1e; MatrixXd int2e; MatrixXd int2e_x;
        int find_subspace(int) const;
        RDM1();
        RDM1(int,double,MatrixXd,MatrixXd,MatrixXd,MatrixXd);
        RDM1(VectorXd,MatrixXd,int,double,MatrixXd,MatrixXd,MatrixXd,MatrixXd);
        RDM1(const RDM1*);
        ~RDM1();
        MatrixXd mat() const;
        void opti(Functional*,int disp=0,double epsi=1e-6,int maxiter=100);
};
//Auxiliary functions used to minimise the energy of the 1RDM
double norm2(VectorXd* x); double norm1(VectorXd* x);
tuple<double,int> opti_aux(RDM1*,Functional*,double epsilon=1e-8,bool disp=false, int maxiter=100);
MatrixXd exp_unit(VectorXd*);
void print_t(chrono::high_resolution_clock::time_point, chrono::high_resolution_clock::time_point, int iter=1);

double f_n_subspace(const vector<double>& x, vector<double>& grad, void* f_data);
double f_n(const vector<double>& x, vector<double>& grad, void* f_data);
#endif