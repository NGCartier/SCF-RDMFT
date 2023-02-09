#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "HF.hpp"

MatrixXd HF_WK(RDM1* gamma){
    int l = gamma->size(); MatrixXd W (l,l);
    VectorXd N = gamma->n();
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = gamma->n(i)* v(i,j);
        }
    }
    
    return 1./2.*W;
}

VectorXd HF_dWK(RDM1* gamma){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    VectorXd N = gamma->n();
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for(int j = 0; j<l; j++){
            dW(j) += gamma->dn(i,j)*v(i,i);
        }
    }
    return 1./2.*dW;
}

MatrixXd H_WK(RDM1* gamma){
    int l = gamma->size();
    return MatrixXd::Zero(l,l);
}

VectorXd H_dWK(RDM1* gamma){
    int l = gamma->size();
    return VectorXd::Zero(l);
}


MatrixXd E1_WK(RDM1* gamma){
    int l = gamma->size();
    return MatrixXd::Zero(l,l);
}

VectorXd E1_dWK(RDM1* gamma){
    int l = gamma->size();
    return VectorXd::Zero(l);
}