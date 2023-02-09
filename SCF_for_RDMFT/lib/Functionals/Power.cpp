#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "Power.hpp"

double const ALPHA = 1.113;

MatrixXd Power_WK(RDM1* gamma){
    int l = gamma->n.size(); VectorXd N = pow(&gamma->n,ALPHA); MatrixXd W (l,l);
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = N(i)* v(i,j);
        }
    }
    return W;
}


VectorXd Power_dWK(RDM1* gamma){
    int l = gamma->n.size(); VectorXd N = pow(&gamma->n,ALPHA); VectorXd dN = ALPHA*pow(&gamma->n,ALPHA-1.); VectorXd dW (l);
    MatrixXd v = v_K(gamma,&N);
    for (int i = 0; i<l; i++){
        dW(i) = dN(i)* v(i,i);
    }
    return dW;
}
