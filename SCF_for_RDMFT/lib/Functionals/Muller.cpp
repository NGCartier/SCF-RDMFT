#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;


#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "Muller.hpp"
/* Defines the auxiliary functions needed to compute the energy of a functional in ../classes/Functional.cpp */

VectorXd Muller_fJ(RDM1* gamma){
    return pow(gamma->n,2);
}

VectorXd Muller_gJ(RDM1* gamma){
    return pow(gamma->n,2);
}

VectorXd Muller_fK(RDM1* gamma){
    return gamma->n;
}

VectorXd Muller_gK(RDM1* gamma){
    return gamma->n;
}

VectorXd Muller_dfJ(RDM1* gamma){
    return 2*gamma->n;
}

VectorXd Muller_dgJ(RDM1* gamma){
    return 2*gamma->n;
}

VectorXd Muller_dfK(RDM1* gamma){
    int l = gamma->n.size(); VectorXd res(l);
    for (int i=0;i<l;i++){ res(i) = 1; }
    return res;
}

VectorXd Muller_dgK(RDM1* gamma){
    int l = gamma->n.size(); VectorXd res(l);
    for (int i=0;i<l;i++){ res(i) = 1; }
    return res;
}
