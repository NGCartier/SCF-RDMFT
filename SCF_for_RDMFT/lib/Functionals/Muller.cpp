#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "Muller.hpp"


MatrixXd Muller_WK(RDM1* gamma){
    int l = gamma->size(); MatrixXd W (l,l);
    VectorXd N = gamma->n().cwiseSqrt();
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = sqrt(gamma->n(i))* v(i,j);
        }
        
    }
    return W;
}

VectorXd Muller_dWK(RDM1* gamma){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    VectorXd N = gamma->n().cwiseSqrt();
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            dW(j) += gamma->dsqrt_n(i,j)*v(i,i);
        }
            
    }
    return dW;
}


