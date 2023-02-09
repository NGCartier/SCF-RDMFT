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
    int l = gamma->size(); MatrixXd W = MatrixXd::Zero(l,l);
    for (int i=0;i<l;i++){
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        for (int j=0;j<l;j++){
            if(gamma->x(i)<-gamma->mu(0)){
                W(i,j) = 2.* gamma->n(i)*v_K(gamma,&n_i)(i,j);
            }
        }
    }
    return W;
}

MatrixXd BBC1_WK(RDM1* gamma){
    int l = gamma->size(); MatrixXd W (l,l);
    VectorXd n_virt = VectorXd::Zero(l); VectorXd n_occ = VectorXd::Zero(l); 
    for (int i=0;i<l;i++){
        if(gamma->x(i)<-gamma->mu(0)){
            n_virt(i) = sqrt(gamma->n(i));
        }
        else{ 
            n_occ(i) = sqrt(gamma->n(i));
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
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        for (int j=0;j<l;j++){
            if(gamma->x(i)<-gamma->mu(0)){
                dW(j) += gamma->dn(i,j) *v_K(gamma,&n_i)(i,i);
            }
        }
    }
    return dW;
}

VectorXd BBC1_dWK(RDM1* gamma){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    VectorXd  n_virt = VectorXd::Zero(l); VectorXd  n_occ = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        if(gamma->x(i)<-gamma->mu(0)){
            n_virt(i) = sqrt(gamma->n(i));
        }
        else{
            n_occ(i) = sqrt(gamma->n(i));
        }
    }
    MatrixXd v_virt = v_K(gamma,&n_virt); MatrixXd v_occ = v_K(gamma,&n_occ);  
    for (int i=0; i<l; i++){   
        for (int j=0;j<l;j++){
            if (gamma->x(i)<-gamma->mu(0)){
                double dn_virt = gamma->dsqrt_n(i,j); 
                dW(j) += - dn_virt * v_virt(i,i) + dn_virt * v_occ(i,i);
            }
            else{
                double dn_occ = gamma->dsqrt_n(i,j); 
                dW(j) += dn_occ * v_virt(i,i) + dn_occ * v_occ(i,i);
            }
        } 
                     
    }
    return dW + dW_diag_C1(gamma);
}
