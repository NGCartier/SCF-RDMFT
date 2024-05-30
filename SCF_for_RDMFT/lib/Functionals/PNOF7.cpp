#include <math.h>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include <iostream>
using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "PNOF7.hpp"

int max_index(RDM1* gamma, vector<int> omega){
    int index =omega[0];
    for (int i: omega){
        if( gamma->n(i)>gamma->n(index)){index =i;}
    }
    return index;
}

MatrixXd Wg(RDM1* gamma, vector<int> omega) {
    int l = gamma->n.size();
    MatrixXd res = MatrixXd::Zero(l,l); VectorXd n_occ = VectorXd::Zero(l); VectorXd n_virt = VectorXd::Zero(l); 
    int g = max_index(gamma, omega);
    for (int p: omega) {
        if (p==g){
            n_occ(p) = gamma->n(p);
        }
        else{
            n_virt(p) = gamma->n(p);
        }
         
    }
    VectorXd n = n_virt + n_occ;
    MatrixXd v_occ = v_K(gamma,&n_occ); MatrixXd v_virt = v_K(gamma,&n_virt); 
    for (int i:omega) {
        
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        VectorXd n_i_virt = VectorXd::Zero(l); if(n_virt(i)>0){n_i_virt(i) = gamma->n(i);}
        for (int j=0;j<l;j++) {
            res(i,j) += - n_virt(i)*v_occ(i,j) - n_occ(i)*v_virt(i,j) + n_virt(i)*v_virt(i,j);

                        //Diagonal part                                 and Diagonal correction
            res(i,j) += pow(n(i),2)*v_J(gamma, &n_i)(i,j) - n_virt(i)*v_K(gamma,&n_i_virt)(i,j);
        }
    }
    return res;
}

MatrixXd Wfg(RDM1* gamma, vector<int> omega_f, vector<int> omega_g, bool old) {
    int l = gamma->n.size();
    vector<int> omega_fg (omega_f); omega_fg.insert(omega_fg.end(),omega_g.begin(),omega_g.end());
    MatrixXd res = MatrixXd::Zero(l, l);
    VectorXd n_focc = VectorXd::Zero(l);  VectorXd n_gocc = VectorXd::Zero(l);  VectorXd nh_focc = VectorXd::Zero(l);  VectorXd nh_gocc = VectorXd::Zero(l);
    VectorXd n_fvirt = VectorXd::Zero(l); VectorXd n_gvirt = VectorXd::Zero(l); VectorXd nh_fvirt = VectorXd::Zero(l); VectorXd nh_gvirt = VectorXd::Zero(l);
    int f = max_index(gamma, omega_f);
    int g = max_index(gamma, omega_g);
    for (int p: omega_f) {
        if(p==f){
            n_focc(p) = gamma->n(p); nh_focc(p) = gamma->n(p)* sqrt(abs(2.-pow(gamma->n(p),2))); //abs to avoid numerical issues 
        }
        else{
            n_fvirt(p) = gamma->n(p); nh_fvirt(p) = gamma->n(p)* sqrt(abs(2.-pow(gamma->n(p),2)));
        }  
    }
    for (int q: omega_g) {
        if(q==g){
            n_gocc(q) = gamma->n(q); nh_gocc(q) = gamma->n(q)* sqrt(abs(2.-pow(gamma->n(q),2)));
        }
        else{
            n_gvirt(q) = gamma->n(q); nh_gvirt(q) = gamma->n(q)* sqrt(abs(2.-pow(gamma->n(q),2)));
        }
    }
    VectorXd n_f = n_focc+n_fvirt; VectorXd n_g = n_gocc+n_gvirt;
    n_f = pow(&n_f,2); n_g = pow(&n_g,2);
    MatrixXd v_f = v_J(gamma,&n_f) -1./2.*v_K(gamma,&n_f); MatrixXd v_g = v_J(gamma,&n_g) -1./2.*v_K(gamma,&n_g); 
    MatrixXd vh_focc = v_K(gamma, &nh_focc); MatrixXd vh_gocc = v_K(gamma, &nh_gocc); MatrixXd vh_fvirt = v_K(gamma,&nh_fvirt); MatrixXd vh_gvirt = v_K(gamma,&nh_gvirt);
    for (int i:omega_fg) {
        for (int j = 0; j < l; j++) {
            res(i,j) = n_g(i)*v_f(i,j) + n_f(i)*v_g(i,j)
                      -nh_gocc(i)*vh_focc(i,j) - nh_focc(i)*vh_gocc(i,j) 
                      -nh_gocc(i)*vh_fvirt(i,j)- nh_focc(i)*vh_gvirt(i,j)
                      -nh_gvirt(i)*vh_focc(i,j)- nh_fvirt(i)*vh_gocc(i,j)
                      +(2.*old-1.)*(nh_gvirt(i)*vh_fvirt(i,j)+nh_fvirt(i)*vh_gvirt(i,j) );
        }
    }
    return res; 
}

MatrixXd PNOF7_WK(RDM1* gamma) {
    int l = gamma->n.size(); MatrixXd W = MatrixXd::Zero(l, l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        W += Wg(gamma, gamma->omega[f]);
        for (int g = 0; g < f; g++) {
            W += Wfg(gamma, gamma->omega[f], gamma->omega[g],false);
        }
    }
    return - W; //Factor -1 convention of the Functional_class
}

MatrixXd PNOF7_old_WK(RDM1* gamma) {
    int l = gamma->n.size(); MatrixXd W = MatrixXd::Zero(l, l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        W += Wg(gamma, gamma->omega[f]);
        for (int g = 0; g < f; g++) {
            W += Wfg(gamma, gamma->omega[f], gamma->omega[g],true);
        }
    }
    return - W; //Factor -1 convention of the Functional_class
}


VectorXd dWg(RDM1* gamma, vector<int> omega) {
    int l = gamma->n.size();
    VectorXd res = VectorXd::Zero(l); VectorXd n_occ = VectorXd::Zero(l); VectorXd n_virt = VectorXd::Zero(l); 
    VectorXd dn_occ = VectorXd::Zero(l); VectorXd dn_virt = VectorXd::Zero(l);
    int g = max_index(gamma, omega);
    for (int p: omega) {
        if (p==g){
            n_occ(p) = gamma->n(p); dn_occ(p) = 1.;
        }
        else{
            n_virt(p) = gamma->n(p); dn_virt(p) = 1.;
        }
         
    }
    VectorXd n = n_virt + n_occ;
    MatrixXd v_occ = v_K(gamma,&n_occ); MatrixXd v_virt = v_K(gamma,&n_virt); 
    for (int i:omega) {
        VectorXd n_i = VectorXd::Zero(l); n_i(i) = 1.;
        VectorXd n_i_virt = VectorXd::Zero(l); if(n_virt(i)>0){n_i_virt(i) = gamma->n(i);}
        res(i) += - dn_virt(i)*v_occ(i,i) - dn_occ(i)*v_virt(i,i) + dn_virt(i)*v_virt(i,i);

        //Diagonal part                                 and Diagonal correction
        res(i) += n(i)*v_J(gamma,&n_i)(i,i) - dn_virt(i)*v_K(gamma,&n_i_virt)(i,i);
    }
    return res;
}
double sign(double x){
    if (x>0){return 1.;}
    else{ return -1.;}
}

VectorXd dWfg(RDM1* gamma, vector<int> omega_f, vector<int> omega_g, bool old) {
    int l = gamma->n.size();
    VectorXd res = VectorXd::Zero(l);
    VectorXd n_focc = VectorXd::Zero(l);  VectorXd n_gocc = VectorXd::Zero(l);  VectorXd nh_focc = VectorXd::Zero(l);  VectorXd nh_gocc = VectorXd::Zero(l);
    VectorXd n_fvirt = VectorXd::Zero(l); VectorXd n_gvirt = VectorXd::Zero(l); VectorXd nh_fvirt = VectorXd::Zero(l); VectorXd nh_gvirt = VectorXd::Zero(l);
    VectorXd dnh_focc = VectorXd::Zero(l);  VectorXd dnh_gocc = VectorXd::Zero(l); VectorXd dnh_fvirt = VectorXd::Zero(l); VectorXd dnh_gvirt = VectorXd::Zero(l);
    VectorXd h = (VectorXd::Constant(l,2.)-pow(&gamma->n,2) ).cwiseAbs().cwiseSqrt();
    int f = max_index(gamma, omega_f);
    int g = max_index(gamma, omega_g);
    for (int p: omega_f) {
        if(p==f){
            n_focc(p) = gamma->n(p); nh_focc(p) = gamma->n(p)* h(p);
            dnh_focc(p) = h(p) - pow(gamma->n(p),2)/max(h(p), 1e-5); 
        }
        else{
            n_fvirt(p) = gamma->n(p); nh_fvirt(p) = gamma->n(p)* h(p);
            dnh_fvirt(p) = h(p) - pow(gamma->n(p),2)/max(h(p), 1e-5); 
        }  
    }
    for (int q: omega_g) {
        if(q==g){
            n_gocc(q) = gamma->n(q); nh_gocc(q) = gamma->n(q)* h(q);
            dnh_gocc(q) = h(q) - pow(gamma->n(q),2)/max(h(q), 1e-5); 
        }
        else{
            n_gvirt(q) = gamma->n(q); nh_gvirt(q) = gamma->n(q)* h(q);
            dnh_gvirt(q) = h(q) - pow(gamma->n(q),2)/max(h(q), 1e-5); 
        }
    }
    VectorXd n_f_0 = n_focc+n_fvirt; VectorXd n_g_0 = n_gocc+n_gvirt;
    VectorXd n_f = pow(&n_f_0,2); VectorXd n_g = pow(&n_g_0,2);
    VectorXd dn_f = 2.*(n_f_0); VectorXd dn_g = 2.*(n_g_0);
    MatrixXd v_f = v_J(gamma,&n_f) -1./2.*v_K(gamma,&n_f); MatrixXd v_g = v_J(gamma,&n_g) -1./2.*v_K(gamma,&n_g); 
    MatrixXd vh_focc = v_K(gamma,&nh_focc); MatrixXd vh_gocc = v_K(gamma,&nh_gocc); MatrixXd vh_fvirt = v_K(gamma,&nh_fvirt); MatrixXd vh_gvirt = v_K(gamma,&nh_gvirt);
    for (int i = 0; i < l; i++) {
        res(i) = dn_g(i)*v_f(i,i) + dn_f(i)*v_g(i,i)
                -dnh_gocc(i)*vh_focc(i,i) - dnh_focc(i)*vh_gocc(i,i) 
                -dnh_gocc(i)*vh_fvirt(i,i)- dnh_focc(i)*vh_gvirt(i,i)
                -dnh_gvirt(i)*vh_focc(i,i)- dnh_fvirt(i)*vh_gocc(i,i)
                +(2.*old-1.)*(dnh_gvirt(i)*vh_fvirt(i,i)+dnh_fvirt(i)*vh_gvirt(i,i));
    }
    return res; 
}

/* get W_J -W_K for PNOF7 functional */
VectorXd PNOF7_dWK(RDM1* gamma) {
    int l = gamma->n.size(); VectorXd dW = VectorXd::Zero(l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        dW += dWg(gamma, gamma->omega[f]);
        for (int g = 0; g < f; g++) {
            dW += dWfg(gamma, gamma->omega[f], gamma->omega[g],false);
        }
    }
    return - dW;
}
/* get W_J -W_K for gth subspace of the PNOF7 functional */
VectorXd PNOF7_dWK_subspace(RDM1* gamma, int g){
    int l = gamma->n.size(); VectorXd dW = VectorXd::Zero(l);
    dW += dWg(gamma, gamma->omega[g]);
    for (int f = 0; f < gamma->omega.size(); f++) {
        if (f!=g){
            dW += dWfg(gamma, gamma->omega[g], gamma->omega[f],false);
        }
    }
    return - dW;
}

/* get W_J -W_K for PNOF7 functional */
VectorXd PNOF7_old_dWK(RDM1* gamma) {
    int l = gamma->n.size(); VectorXd dW = VectorXd::Zero(l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        dW += dWg(gamma, gamma->omega[f]);
        for (int g = 0; g < f; g++) {
            dW += dWfg(gamma, gamma->omega[f], gamma->omega[g],true);
        }
    }
    return - dW;
}
/* get W_J -W_K for gth subspace of the PNOF7 functional */
VectorXd PNOF7_old_dWK_subspace(RDM1* gamma, int g){
    int l = gamma->n.size(); VectorXd dW = VectorXd::Zero(l);
    dW += dWg(gamma, gamma->omega[g]);
    for (int f = 0; f < gamma->omega.size(); f++) {
        if (f!=g){
            dW += dWfg(gamma, gamma->omega[g], gamma->omega[f],true);
        }
    }
    return - dW;
}
