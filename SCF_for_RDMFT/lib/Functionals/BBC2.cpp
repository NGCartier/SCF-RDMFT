#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "BBC2.hpp"
#include "Muller.hpp"
#include <iostream>

MatrixXd W_diag(RDM1* gamma){
    int l = gamma->size(); MatrixXd W (l,l);
    for (int i=0;i<l;i++){
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        for (int j=0;j<l;j++){
            if(gamma->x(i)>=-gamma->mu(0)){
                W(i,j) = (gamma->n(i) - 1./2.*pow(gamma->n(i),2))*v_K(gamma,&n_i)(i,j);
            }
            else{
                W(i,j) = 2.* gamma->n(i)*v_K(gamma,&n_i)(i,j);
            }
        }
        
    }
    return W;
}

MatrixXd BBC2_WK(RDM1* gamma){
    int l = gamma->size(); MatrixXd W (l,l);
    VectorXd n_virt = VectorXd::Zero(l); VectorXd n_occ = VectorXd::Zero(l); VectorXd n_occ2 = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        if(gamma->x(i)<-gamma->mu(0)){
            n_virt(i) = sqrt(gamma->n(i));
        }
        else{ 
            n_occ(i) = sqrt(gamma->n(i));
            n_occ2(i)= gamma->n(i);
        }
    }
    MatrixXd v_virt = v_K(gamma,&n_virt); MatrixXd v_occ = v_K(gamma,&n_occ); MatrixXd v_occ2 = v_K(gamma,&n_occ2); 
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){  
            W(i,j) = - n_virt(i)* v_virt(i,j) + 1./sqrt(2)*n_virt(i)* v_occ(i,j) + 1./sqrt(2)*n_occ(i)* v_virt(i,j) + 1./2.*n_occ2(i)* v_occ2(i,j); 
        }    
    }
    return W + W_diag(gamma);
}

// Derivative respec to the occupations

VectorXd dW_diag(RDM1* gamma){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        for (int j=0;j<l;j++){
            if(gamma->x(i)>=-gamma->mu(0)){
                dW(j) += (1. - gamma->n(i))* gamma->dn(i,j) *v_K(gamma,&n_i)(i,i);
            }
            else{
                dW(j) += 2.*gamma->dn(i,j) *v_K(gamma,&n_i)(i,i);
            }
        }
    }
    return 1./2.*dW;
}

VectorXd BBC2_dWK(RDM1* gamma){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    VectorXd  n_virt = VectorXd::Zero(l); VectorXd  n_occ = VectorXd::Zero(l); VectorXd  n_occ2 = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        if(gamma->x(i)<-gamma->mu(0)){
            n_virt(i) = sqrt(gamma->n(i));
        }
        else{
            n_occ(i) = sqrt(gamma->n(i));
            n_occ2(i) = gamma->n(i);
        }
    }
    MatrixXd v_virt = v_K(gamma,&n_virt); MatrixXd v_occ = v_K(gamma,&n_occ); MatrixXd v_occ2 = v_K(gamma,&n_occ2); 
    for (int i=0; i<l; i++){   
        for (int j=0;j<l;j++){
            if (gamma->x(i)<-gamma->mu(0)){
                double dn_virt = gamma->dsqrt_n(i,j); 
                dW(j) += - dn_virt * v_virt(i,i) + 1./sqrt(2)*dn_virt * v_occ(i,i);
            }
            else{
                double dn_occ = gamma->dsqrt_n(i,j); double dn_occ2 = gamma->dn(i,j);
                dW(j) += 1./sqrt(2)*dn_occ * v_virt(i,i) + 1./2.*dn_occ2 * v_occ2(i,i);
            }
        } 
                     
    }
    return dW + dW_diag(gamma);
}
