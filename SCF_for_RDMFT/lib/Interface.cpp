#include <tuple>
#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;


#include "classes/1RDM_class.hpp"
#include "classes/Functional_class.hpp"
#include "classes/Matrix_Tensor_converter.cpp"
#include "Interface.hpp"
#include "Functionals/Muller.hpp"
#include "Functionals/HF.hpp"
#include "Functionals/BBC2.hpp"
#include "Functionals/PNOF7.hpp"

// Wrapper to python code
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
PYBIND11_MODULE(Compute_1RDM, m){
    m.def("Optimize_1RDM", &Optimize_1RDM, " Given occupations, natural orbital matrix, number of electrons, \
    nuclear energy, overlap matrix, tensors of integrals (one and two bodies) and a functional returns the \
    optimized 1RDM (ie corresponding occupation and natural orbital matrix) and prints its energy");
    m.def("E",&E, "Given occupations, natural orbital matrix, number of electrons, \
    nuclear energy, overlap matrix, tensors of integrals (one and two bodies) and a \
    functional returns the corresponding energy");
    m.def("test",&test);
    m.doc() = "C++ extension to compute 1RDMs from PySCF";
}





       
//Used to test the library from Python
void test(VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int){
                                
    int l = overlap.rows(); int ll = pow(l,2);
    // (10)(32) permutation already presente, due to type conversion
    Tensor<double,4> T(l,l,l,l); T = TensorCast(elec2int,l,l,l,l);  Eigen::array<int,4> index({ 1,3,0,2 });
    Tensor<double,4> T2 = T.shuffle(index);
    MatrixXd elec2int_x(ll, ll); elec2int_x = MatrixCast(T2, ll, ll);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x);

    MatrixXd g (l,l); g=gamma.mat();
    
    
    auto t0 = chrono::high_resolution_clock::now(); 
    for (int i = 0;i<1;i++){
    //code to test
        cout<<"E_Muller="<<Muller_func.E(&gamma) <<endl;
        gamma.subspace();
        cout<<"E_PNOF7 ="<<PNOF7_func.E(&gamma)<<endl;
        cout<<"grad E:"<<PNOF7_func.grad_E(&gamma).transpose()<<endl<<endl;
        cout<<"num grad:"<<grad_func(test_E,PNOF7_func,&gamma).transpose()<<endl<<endl;
       
    }
    auto t1 = chrono::high_resolution_clock::now();
    print_t(t1,t0,1); cout<<endl;
    
}

/* 
\param arg    func: functional to use (modify this function to add new functinals)
              disp: if <1 prints details of the computation 
              epsi: relative error required 
              Maxiter: maximum number of iteration for one optimisation of the occupations/NOs
              other arguments are provided by the Python part of Interface and used to build the 1RDM
\param result the vector of the sqrt of the occupations (n) and matrix of the NOs (no) 
              (the 1RDM = no * Diagonal(n) no.T)
              also prints the corresponding ground state energy.
*/
tuple<VectorXd, MatrixXd> Optimize_1RDM(string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
    MatrixXd overlap, MatrixXd elec1int, MatrixXd elec2int,
    int disp, double epsi, int Maxiter) {
    int l = overlap.rows(); int ll = pow(l, 2);
    Tensor<double, 4> T(l, l, l, l); T = TensorCast(elec2int, l, l, l, l);  Eigen::array<int, 4> index({ 1,3,0,2 });
    MatrixXd elec2int_x(ll, ll); elec2int_x = MatrixCast(T.shuffle(index), ll, ll);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x); double E;
    if (func == "Muller") {
        gamma.opti(Muller_func, disp, epsi, sqrt(epsi), 1e-1, Maxiter);
        E = Muller_func.E(&gamma);
    }
    else {
        if (func == "BBC2") {
            gamma.opti(BBC2_func, disp, epsi, sqrt(epsi), 1e-1, Maxiter);
            E = BBC2_func.E(&gamma);
        }
        else {
            if (func == "PNOF7") {
                gamma.subspace();
                gamma.opti(PNOF7_func, disp, epsi, sqrt(epsi), 1e-1, Maxiter);
                E = PNOF7_func.E(&gamma);
            }
            //To add a new functional 
            //else {
            //  if (func=='my_new_func'){//included above
            //      gamma.opti(my_new_func); 
            //      E = my_new_func.E(&gamma);
            //  }
            else{
                gamma.opti(HF_func, disp, epsi, sqrt(epsi), 1e-1, Maxiter);
                E = HF_func.E(&gamma);
            }
            
        }
    }
    cout<<"E="<<E<<endl;
    return make_tuple(gamma.n, gamma.no); 
     
}

/*
\param arg    func: functional to use (modify this function to add new functinals)
              other arguments are provided by the Python part of Interface and used to build the 1RDM
\param result the energy of the corresponding 1RDm for the funtionla func.
*/
double E (string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int){
    int l = overlap.rows(); int ll = pow(l, 2);
    Tensor<double, 4> T(l, l, l, l); T = TensorCast(elec2int, l, l, l, l);  Eigen::array<int, 4> index({ 1,3,0,2 });
    MatrixXd elec2int_x(ll, ll); elec2int_x = MatrixCast(T.shuffle(index), ll, ll);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x); double E;
    if (func == "Muller") {
        E = Muller_func.E(&gamma);
    }
    else {
        if (func == "BBC2") {
            E = BBC2_func.E(&gamma);
        }
        else {
            if (func == "PNOF7") {
                gamma.subspace();
                E = PNOF7_func.E(&gamma);
            }
            else{
                E = HF_func.E(&gamma);
            }
            
        }
    }
    return E; 
}

/*  //        USED TO TEST C++ PART 

#include <chrono>  

int main(){
    int l =2; int ll = pow(l,2);
    VectorXd n (l); MatrixXd no (l,l); MatrixXd ovlp (l,l); MatrixXd I1 (l,l); MatrixXd I2 (ll,ll); double E_nuc; E_nuc = 0.17639240364;
    // H2 in STO-3G basis, CISD guess
    n << 0.96182731, 1.03676817;
    no<< 0.71365341, -0.70073708, -0.71365341, -0.70073708;
    ovlp<<1., 0.01826264, 0.01826264, 1. ;
    I1<<-0.64297414, -0.02083273,-0.02083273, -0.64297414;
    I2 << 7.74605944e-01, 6.54393797e-03,
        6.54393797e-03, 1.76382446e-01,
        6.54393797e-03, 1.53991591e-04,
        1.53991591e-04, 6.54393797e-03,
        6.54393797e-03, 1.53991591e-04,
        1.53991591e-04, 6.54393797e-03,
        1.76382446e-01, 6.54393797e-03,
        6.54393797e-03, 7.74605944e-01;
    Tensor<double, 4> T(l, l, l, l); T = TensorCast(I2,l,l,l,l);  Eigen::array<int, 4> index({ 1,3,0,2 });
    MatrixXd I2_x(ll, ll); I2_x = MatrixCast(T.shuffle(index), ll, ll);
    RDM1 gamma = RDM1(n,no,2,E_nuc,ovlp,I1,I2,I2_x);

    cout.precision(18);

    
    cout<<"E_Muller="<<Muller_func.E(&gamma) <<endl;
    gamma.subspace();
    cout<<"E_PNOF7 ="<<PNOF7_func.E(&gamma)<<endl;

    cout<<"grad E:"<<PNOF7_func.grad_E(&gamma,true,false).transpose()<<endl<<endl;
    cout<<"test dE:"<<(dE1(&gamma,true,false)+PNOF7_dEK(&gamma)).transpose()<<endl<<endl;
    cout<<"num grad:"<<grad_func(test_E,PNOF7_func,&gamma).transpose()<<endl<<endl;
        
    return 0;
}
*/

//Used to test numerical gradient of a functional
double test_E(Functional func,RDM1*gamma){
    MatrixXd W_J = func.compute_WJ(gamma); MatrixXd W_K = func.compute_WK(gamma);
    //return 1./2.*(W_J-W_K).trace();
    return func.E(gamma);
}

//code numerical gradient
VectorXd grad(double (*f)(Functional, RDM1*), Functional func, RDM1* gamma, double epsi){
    int l = gamma->n.size(); int ll = l*(l+1)/2; VectorXd res (ll);
    for (int i=0;i<ll;i++){
        RDM1 gamma_p = RDM1(gamma); RDM1 gamma_m = RDM1(gamma);
        if ( i<l ){
            VectorXd dn_p (l); dn_p = gamma->n; dn_p(i) += epsi;
            VectorXd dn_m (l); dn_m = gamma->n; dn_m(i) -= epsi;
            gamma_p.set_n(dn_p); gamma_m.set_n(dn_m);
            
        }
        else{
            VectorXd dtheta (ll-l); dtheta = VectorXd::Zero(ll-l); dtheta(i-l) = epsi;
            gamma_p.set_no( gamma->no*exp_unit(dtheta)); gamma_m.set_no( gamma->no*exp_unit(-dtheta));
        }
        res(i) = (f(func, &gamma_p) - f(func, &gamma_m))/(2.*epsi);
    }
    return res;
}

VectorXd grad_func(double (*f)(Functional, RDM1*), Functional func, RDM1* gamma){
    int l = gamma->n.size(); int ll = l*(l+1)/2;
    VectorXd x0 (ll); x0 = VectorXd::Zero(ll);
    for (int i=0;i<l;i++){
        x0(i) = gamma->n(i);
    }
    
    return grad(f, func, gamma,1e-8);
}