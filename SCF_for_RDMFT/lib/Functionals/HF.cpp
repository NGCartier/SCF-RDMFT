#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;


#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "HF.hpp"
/* Defines the auxiliary functions needed to compute the energy of a functional in ../classes/Functional.cpp */

VectorXd HF_fJ(RDM1* gamma){
    return pow(gamma->n,2);
}

VectorXd HF_gJ(RDM1* gamma){
    return pow(gamma->n,2);
}

VectorXd HF_fK(RDM1* gamma){
    return pow(gamma->n,2);
}

VectorXd HF_gK(RDM1* gamma){ 
    return 1./2.*pow(gamma->n,2);
}

VectorXd HF_dfJ(RDM1* gamma){
    return 2.*gamma->n;
}

VectorXd HF_dgJ(RDM1* gamma){
    return 2.*gamma->n;
}

VectorXd HF_dfK(RDM1* gamma){
    return 2.*gamma->n;
}

VectorXd HF_dgK(RDM1* gamma){
    return gamma->n;
}
