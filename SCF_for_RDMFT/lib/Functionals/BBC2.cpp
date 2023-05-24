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
    int l = gamma->n.size(); MatrixXd W (l,l);
    for (int i=0;i<l;i++){
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        for (int j=0;j<l;j++){
            if(gamma->n(i)>=1){
                W(i,j) = (pow(gamma->n(i),2) - 1./2.*pow(gamma->n(i),4))*v_K(gamma,&n_i)(i,j);
            }
            else{
                W(i,j) = 2.* pow(gamma->n(i),2)*v_K(gamma,&n_i)(i,j);
            }
        }
        
    }
    return W;
}

MatrixXd BBC2_WK(RDM1* gamma){
    int l = gamma->n.size(); MatrixXd W (l,l);
    VectorXd n_virt = VectorXd::Zero(l); VectorXd n_occ = VectorXd::Zero(l); VectorXd n_occ2 = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        if(gamma->n(i)<1){
            n_virt(i) = gamma->n(i);
        }
        else{ 
            n_occ(i) = gamma->n(i);
            n_occ2(i)= pow(gamma->n(i),2);
        }
    }
    MatrixXd v_virt = v_K(gamma,&n_virt); MatrixXd v_occ = v_K(gamma,&n_occ); MatrixXd v_occ2 = v_K(gamma,&n_occ2); 
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){  
            W(i,j) = - n_virt(i)* v_virt(i,j) + n_virt(i)* v_occ(i,j) + n_occ(i)* v_virt(i,j) + 1./2.*n_occ2(i)* v_occ2(i,j); 
        }    
    }
    return W + W_diag(gamma);
}

// Derivative respec to the occupations

VectorXd dW_diag(RDM1* gamma){
    int l = gamma->n.size(); VectorXd dW (l);
    for (int i=0;i<l;i++){
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1;
        if(gamma->n(i)>=1){
            dW(i)= (gamma->n(i) - pow(gamma->n(i),3)) *v_K(gamma,&n_i)(i,i);
        }
        else{
            dW(i)= 2.* gamma->n(i) *v_K(gamma,&n_i)(i,i);
        }
    }
    return dW;
}

VectorXd BBC2_dWK(RDM1* gamma){
    int l = gamma->n.size(); VectorXd dW (l);
    VectorXd  n_virt = VectorXd::Zero(l); VectorXd  n_occ = VectorXd::Zero(l); VectorXd  n_occ2 = VectorXd::Zero(l);
    VectorXd dn_virt = VectorXd::Zero(l); VectorXd dn_occ = VectorXd::Zero(l); VectorXd dn_occ2 = VectorXd::Zero(l);
    for (int i=0;i<l;i++){
        if(gamma->n(i)<1){
            n_virt(i) = gamma->n(i);
            dn_virt(i)= 1;
        }
        else{ 
            n_occ(i) = gamma->n(i);
            dn_occ(i)= 1;
            n_occ2(i) = pow(gamma->n(i),2);
            dn_occ2(i)= 2*gamma->n(i);
        }
    }
    MatrixXd v_virt = v_K(gamma,&n_virt); MatrixXd v_occ = v_K(gamma,&n_occ); MatrixXd v_occ2 = v_K(gamma,&n_occ2); 
    for (int i = 0; i<l; i++){    
        dW(i) = - dn_virt(i)* v_virt(i,i) + dn_virt(i)* v_occ(i,i) + dn_occ(i)* v_virt(i,i) + 1./2.*dn_occ2(i)* v_occ2(i,i);             
    }
    return dW + dW_diag(gamma);
}
