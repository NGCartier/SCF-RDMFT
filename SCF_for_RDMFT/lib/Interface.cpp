#include <tuple>
#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
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

// Wrapper to Python library
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
PYBIND11_MODULE(Compute_1RDM, m){
    m.def("Optimize_1RDM", &Optimize_1RDM, " Given tensors of integrals and a functional returns the 1RDM and prints its energy");
    m.def("test",&test);
    m.doc() = "C++ extension to compute 1RDMs from PySCF";
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
tuple<VectorXd,MatrixXd> Optimize_1RDM (string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int,
                                    int disp, double epsi, int Maxiter){
    
    int l = overlap.rows(); int ll = pow(l,2);
    Tensor<double,4> T(l,l,l,l); T = TensorCast(elec2int,l,l,l,l);  Eigen::array<int,4> index({ 1,3,0,2 });
    MatrixXd elec2int_x(ll, ll); elec2int_x = MatrixCast(T.shuffle(index), ll, ll);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x); double E;
    if (func=="Muller"){
        gamma.opti(Muller_func, disp, epsi, sqrt(epsi), sqrt(epsi) , Maxiter);
        E = Muller_func.E(&gamma);
    }
    //To add a new functional 
    //else {
    //  if (func=='my_new_func'){//included above
    //      gamma.opti(my_new_func); 
    //      E = my_new_func.E(&gamma);
    //  }
    else{ // Hartree Fock
        gamma.opti(HF_func,disp, epsi, sqrt(epsi), sqrt(epsi) , Maxiter);
        E = HF_func.E(&gamma);
        }
    //}    
    cout<<"E="<<E<<endl;
    return make_tuple(gamma.n, gamma.no); 
     
}
