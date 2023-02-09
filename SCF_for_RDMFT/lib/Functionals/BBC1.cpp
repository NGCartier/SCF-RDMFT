#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "BBC1.hpp"
#include <iostream>

MatrixXd W_diag_C1(RDM1* gamma){
    int l = gamma->n.size(); MatrixXd W = MatrixXd::Zero(l,l);
    for (int i=0;i<l;i++){
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        for (int j=0;j<l;j++){
            if(gamma->n(i)<1.){
                W(i,j) = 2.* pow(gamma->n(i),2)*v_K(gamma,&n_i)(i,j);
            }
        }
    }
    return W;
}

MatrixXd BBC1_WK(RDM1* gamma){
    int l = gamma->n.size(); MatrixXd W (l,l);
    VectorXd n_virt = VectorXd::Zero(l); VectorXd n_occ = VectorXd::Zero(l); 
    for (int i=0;i<l;i++){
        if(gamma->n(i)<1.){
            n_virt(i) = gamma->n(i);
        }
        else{ 
            n_occ(i) = gamma->n(i);
        }
    }
    MatrixXd v_virt = v_K(gamma,&n_virt); MatrixXd v_occ = v_K(gamma,&n_occ); 
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){  
            W(i,j) = - n_virt(i)* v_virt(i,j) + n_virt(i)* v_occ(i,j) + n_occ(i)* v_virt(i,j) + n_occ(i)* v_occ(i,j); 
        }    
    }
    return W + W_diag_C1(gamma);
}

// Derivative respec to the occupations

VectorXd dW_diag_C1(RDM1* gamma){
    int l = gamma->n.size(); VectorXd dW = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        if(gamma->n(i)<1.){
            dW(i)= 2.* gamma->n(i) *v_K(gamma,&n_i)(i,i);
        }
    }
    return dW;
}

VectorXd BBC1_dWK(RDM1* gamma){
    int l = gamma->n.size(); VectorXd dW = VectorXd::Zero(l);
    VectorXd  n_virt = VectorXd::Zero(l); VectorXd  n_occ = VectorXd::Zero(l);
    VectorXd dn_virt = VectorXd::Zero(l); VectorXd dn_occ = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        if(gamma->n(i)<1.){
            n_virt(i) = gamma->n(i);
            dn_virt(i)=1.;
        }
        else{
            n_occ(i) = gamma->n(i);
            dn_occ(i)=1.;
        }
    }
    MatrixXd v_virt = v_K(gamma,&n_virt); MatrixXd v_occ = v_K(gamma,&n_occ);  
    for (int i=0; i<l; i++){   
        if (gamma->n(i)<1.){
            dW(i) += - dn_virt(i) * v_virt(i,i) + dn_virt(i) * v_occ(i,i);
        }
        else{
            dW(i) += dn_occ(i) * v_virt(i,i) + dn_occ(i) * v_occ(i,i);
        }   
    }
    return dW + dW_diag_C1(gamma);
}
