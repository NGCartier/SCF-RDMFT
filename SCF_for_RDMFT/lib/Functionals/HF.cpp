#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;


#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "HF.hpp"

MatrixXd HF_WK(RDM1* gamma){
    int l = gamma->n.size(); VectorXd N = pow(gamma->n,2); MatrixXd W (l,l);
    MatrixXd v = v_K(gamma,N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = N(i)* v(i,j);
        }
    }
    
    return 1./2.*W;
}

VectorXd HF_dWK(RDM1* gamma){
    int l = gamma->n.size(); VectorXd N = pow(gamma->n,2); VectorXd dN = 2.*gamma->n; VectorXd dW (l);
    MatrixXd v = v_K(gamma,N);
    for (int i = 0; i<l; i++){
        dW(i) = dN(i)* v(i,i);
    }
    return 1./2.*dW;
}

