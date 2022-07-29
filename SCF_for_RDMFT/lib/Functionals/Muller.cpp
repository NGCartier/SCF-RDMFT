#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;


#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "Muller.hpp"


MatrixXd Muller_WK(RDM1* gamma){
    int l = gamma->n.size(); MatrixXd W (l,l);
    MatrixXd v = v_K(gamma,gamma->n);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = gamma->n(i)* v(i,j);
        }
        
    }
    return W;
}

VectorXd Muller_dWK(RDM1* gamma){
    return v_K(gamma,gamma->n).diagonal();
}

