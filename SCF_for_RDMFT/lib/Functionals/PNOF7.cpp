#include <math.h>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <iostream>
using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "../classes/EBI_add.hpp" //get defintion of sign without conflict
#include "PNOF7.hpp"

MatrixXd Wg(RDM1* gamma, int f) {
    int l = gamma->size();
    vector<int> omega = gamma->omega[f];
    MatrixXd res = MatrixXd::Zero(l,l); VectorXd n_occ = VectorXd::Zero(l); VectorXd n_virt = VectorXd::Zero(l); 
    
    for (int p: omega) {
        if (gamma->x(p)>-gamma->mu(f)){
            n_occ(p) = sqrt(gamma->n(p));
        }
        else{
            n_virt(p) = sqrt(gamma->n(p));
        }
         
    }
    VectorXd n = n_virt + n_occ;
    MatrixXd v_occ = v_K(gamma,&n_occ); MatrixXd v_virt = v_K(gamma,&n_virt); 
    for (int i: omega) {
        VectorXd n_i = VectorXd::Zero(l); if(n(i)>0){n_i(i) = 1.;}
        VectorXd n_i_virt = VectorXd::Zero(l); if(n_virt(i)>0){n_i_virt(i) = n(i);}
         VectorXd n_i_occ  = VectorXd::Zero(l); if(n_occ (i)>0){n_i_occ (i) = gamma->n(i);}
        for (int j=0;j<l;j++) {
            res(i,j) += - n_virt(i)*v_occ(i,j) - n_occ(i)*v_virt(i,j) + n_virt(i)*v_virt(i,j);

                        //Diagonal part                                 and Diagonal correction
            res(i,j) += pow(n(i),2)*v_J(gamma,&n_i)(i,j) - n_virt(i)*v_K(gamma,&n_i_virt)(i,j);
        }
    }
    return res;
}

MatrixXd Wfg(RDM1* gamma, int f, int g) {
    int l = gamma->size();
    vector<int> omega_f = gamma->omega[f]; vector<int> omega_g = gamma->omega[g];
    vector<int> omega_fg (omega_f); omega_fg.insert(omega_fg.end(),omega_g.begin(),omega_g.end());
    MatrixXd res = MatrixXd::Zero(l, l);
    VectorXd n_focc = VectorXd::Zero(l);  VectorXd n_gocc = VectorXd::Zero(l);  VectorXd nh_focc = VectorXd::Zero(l);  VectorXd nh_gocc = VectorXd::Zero(l);
    VectorXd n_fvirt = VectorXd::Zero(l); VectorXd n_gvirt = VectorXd::Zero(l); VectorXd nh_fvirt = VectorXd::Zero(l); VectorXd nh_gvirt = VectorXd::Zero(l);
    for (int p: omega_f) {
        if(gamma->x(p)> -gamma->mu(f)){
            double np = gamma->n(p);
            n_focc(p) = sqrt(np); nh_focc(p) = sqrt(np* abs(2.-np)); //abs to avoid numerical issues 
        }
        else{
            double np = gamma->n(p);
            n_fvirt(p) = sqrt(np); nh_fvirt(p) = sqrt(np* abs(2.-np));
        }  
    }
    for (int q: omega_g) {
        if(gamma->x(q)>= -gamma->mu(g)){
            double nq = gamma->n(q);
            n_gocc(q) = sqrt(nq); nh_gocc(q) = sqrt(nq*abs(2.-nq));
        }
        else{
            double nq = gamma->n(q);
            n_gvirt(q) = sqrt(nq); nh_gvirt(q) = sqrt(nq*abs(2.-nq));
        }
    }
    VectorXd n_f = n_focc+n_fvirt; VectorXd n_g = n_gocc+n_gvirt;
    n_f = pow(&n_f,2); n_g = pow(&n_g,2);
    MatrixXd v_f = v_J(gamma,&n_f) -1./2.*v_K(gamma,&n_f); MatrixXd v_g = v_J(gamma,&n_g) -1./2.*v_K(gamma,&n_g); 
    MatrixXd vh_focc = v_K(gamma,&nh_focc); MatrixXd vh_gocc = v_K(gamma,&nh_gocc); MatrixXd vh_fvirt = v_K(gamma,&nh_fvirt); MatrixXd vh_gvirt = v_K(gamma,&nh_gvirt);
    for (int i: omega_fg) {
        for (int j=0;j<l;j++) {
            res(i,j) = n_g(i)*v_f(i,j) + n_f(i)*v_g(i,j)
                      -nh_gocc(i)*vh_focc(i,j) - nh_focc(i)*vh_gocc(i,j) 
                      -nh_gocc(i)*vh_fvirt(i,j)- nh_focc(i)*vh_gvirt(i,j)
                      -nh_gvirt(i)*vh_focc(i,j)- nh_fvirt(i)*vh_gocc(i,j)
                      -nh_gvirt(i)*vh_fvirt(i,j)-nh_fvirt(i)*vh_gvirt(i,j);
        }
    }
    return res; 
}

MatrixXd PNOF7_WK(RDM1* gamma) {
    int l = gamma->size(); MatrixXd W = MatrixXd::Zero(l,l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        W += Wg(gamma, f);
        for (int g = 0; g < f; g++) {
            W += Wfg(gamma, f, g);
        }
    }
    return - W; //Factor -1 convention of the Functional_class
}


VectorXd dWg(RDM1* gamma, int f) {
    int l = gamma->size();
    vector<int> omega = gamma->omega[f];
    VectorXd res = VectorXd::Zero(l); VectorXd n_occ = VectorXd::Zero(l); VectorXd n_virt = VectorXd::Zero(l); 
    for (int p: omega) {
        if (gamma->x(p)>-gamma->mu(f)){
            n_occ(p) = sqrt(gamma->n(p));
        }
        else{
            n_virt(p) = sqrt(gamma->n(p));
        }
    }
    VectorXd n = n_virt + n_occ;
    MatrixXd v_occ = v_K(gamma,&n_occ); MatrixXd v_virt = v_K(gamma,&n_virt); 
    for (int i: omega) {
        VectorXd n_i = VectorXd::Zero(l); if(n(i)>0){n_i(i) = 1.;}
        VectorXd n_i_virt = VectorXd::Zero(l); if(n_virt(i)>0){n_i_virt(i) = n_virt(i);}
        double v_J_i = v_J(gamma,&n_i)(i,i); double v_K_i = v_K(gamma,&n_i_virt)(i,i);
        for (int j=0;j<l;j++){ 
            if (gamma->x(i)>-gamma->mu(f)){
                double dn_occ = gamma->dsqrt_n(i,j);
                res(j) += - dn_occ*v_virt(i,i);
            }
            else {
                double dn_virt = gamma->dsqrt_n(i,j); 
                //                                                     Diagonal correction
                res(j) += - dn_virt*v_occ(i,i) + dn_virt*v_virt(i,i) - dn_virt*v_K_i;
            }
            //Diagonal part                                 
            res(j) += 1./2.*gamma->dn(i,j)*v_J_i; 
        }
    }
    return res;
}

VectorXd dWfg(RDM1* gamma, int f, int g) {
    int l = gamma->size();
    vector<int> omega_f = gamma->omega[f]; vector<int> omega_g = gamma->omega[g];
    VectorXd res = VectorXd::Zero(l);
    VectorXd n_focc = VectorXd::Zero(l);  VectorXd n_gocc = VectorXd::Zero(l);  VectorXd nh_focc = VectorXd::Zero(l);  VectorXd nh_gocc = VectorXd::Zero(l);
    VectorXd n_fvirt = VectorXd::Zero(l); VectorXd n_gvirt = VectorXd::Zero(l); VectorXd nh_fvirt = VectorXd::Zero(l); VectorXd nh_gvirt = VectorXd::Zero(l);
    VectorXd np = gamma->n(); VectorXd sqrtn_p = np.cwiseSqrt(); VectorXd h = (VectorXd::Constant(l,2.)-np).cwiseSqrt();
    for (int p: omega_f) {
        if(gamma->x(p)>= -gamma->mu(f)){
            n_focc(p) = sqrtn_p(p); nh_focc(p) = sqrtn_p(p)* h(p);
        }
        else{
            n_fvirt(p) = sqrtn_p(p); nh_fvirt(p) = sqrtn_p(p)* h(p);
        }
    }
    for (int p: omega_g) {
        if(gamma->x(p)> -gamma->mu(g)){
            n_gocc(p) = sqrtn_p(p); nh_gocc(p) = sqrtn_p(p)* h(p);
        }
        else{
            n_gvirt(p) = sqrtn_p(p); nh_gvirt(p) = sqrtn_p(p)* h(p);
        }
    }
    VectorXd n_f = n_focc+n_fvirt; VectorXd n_g = n_gocc+n_gvirt;
    n_f = pow(&n_f,2); n_g = pow(&n_g,2);
    MatrixXd v_f = v_J(gamma,&n_f) -1./2.*v_K(gamma,&n_f); MatrixXd v_g = v_J(gamma,&n_g) -1./2.*v_K(gamma,&n_g); 
    MatrixXd vh_focc = v_K(gamma,&nh_focc); MatrixXd vh_gocc = v_K(gamma,&nh_gocc); MatrixXd vh_fvirt = v_K(gamma,&nh_fvirt); MatrixXd vh_gvirt = v_K(gamma,&nh_gvirt);
    for (int p: omega_f) {
        for (int q=0;q<l;q++){
            if(gamma->x(p)>= -gamma->mu(f)){
                double dnh_focc = (h(p) - min(np(p)/h(p),1e3) )* gamma->dsqrt_n(p,q);
                res(q) += - dnh_focc*vh_gocc(p,p) - dnh_focc*vh_gvirt(p,p);
            }
            else{
                double dnh_fvirt = (h(p) - min(np(p)/h(p),1e3) )* gamma->dsqrt_n(p,q); 
                res(q) += - dnh_fvirt*vh_gocc(p,p)- dnh_fvirt*vh_gvirt(p,p);
            }
            res(q) += gamma->dn(p,q)*v_g(p,p);
        }
    }
    for (int p: omega_g){
        for (int q=0;q<l;q++){
            if(gamma->x(p)>= -gamma->mu(g)){
                double dnh_gocc = (h(p) - min(np(p)/h(p),1e3) )* gamma->dsqrt_n(p,q); 
                res(q) += - dnh_gocc*vh_focc(p,p) - dnh_gocc*vh_fvirt(p,p);
            }
            else{
                double dnh_gvirt = (h(p) - min(np(p)/h(p),1e3) )* gamma->dsqrt_n(p,q); 
                res(q) += - dnh_gvirt*vh_focc(p,p)- dnh_gvirt*vh_fvirt(p,p);
            }
            res(q) += gamma->dn(p,q)*v_f(p,p);
        }
    }
    return res; 
}
/* get W_J -W_K for PNOF7 functional */
VectorXd PNOF7_dWK(RDM1* gamma) {
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        dW += dWg(gamma, f);
        for (int g = 0; g < f; g++) {
            dW += dWfg(gamma, f, g);
        }
    }
    
    return - dW;
}
/* get W_J -W_K for gth subspace of the PNOF7 functional */
VectorXd PNOF7_dWK_subspace(RDM1* gamma, int g){
    int l = gamma->size(); VectorXd dW = VectorXd::Zero(l);
    dW += dWg(gamma, g);
    for (int f = 0; f < gamma->omega.size(); f++) {
        if (f!=g){
            dW += dWfg(gamma, g, f);
        }
    }
    return - dW;
}
