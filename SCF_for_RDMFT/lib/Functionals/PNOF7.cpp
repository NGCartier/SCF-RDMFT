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

MatrixXd Wg(RDM1* gamma, vector<int> omega) {
    int l = gamma->n.size();
    MatrixXd res = MatrixXd::Zero(l,l); VectorXd n_occ = VectorXd::Zero(l); VectorXd n_virt = VectorXd::Zero(l); 
    
    for (int p: omega) {
        if (gamma->n(p)>1.){
            n_occ(p) = gamma->n(p);
        }
        else{
            n_virt(p) = gamma->n(p);
        }
         
    }
    VectorXd n = n_virt + n_occ;
    MatrixXd v_occ = v_K(gamma,n_occ); MatrixXd v_virt = v_K(gamma,n_virt); 
    for (int i = 0; i < l; i++) {
        VectorXd n_i = VectorXd::Zero(l); if(n(i)>0){n_i(i) = 1.;}
        VectorXd n_i_virt = VectorXd::Zero(l); if(n_virt(i)>0){n_i_virt(i) = gamma->n(i);}
        for (int j = 0; j < l; j++) {
            res(i,j) += - n_virt(i)*v_occ(i,j) - n_occ(i)*v_virt(i,j) + n_virt(i)*v_virt(i,j);

            //Diagonal part                                 and Diagonal correction
            res(i,j) += pow(n(i),2)*v_J(gamma, n_i)(i,j) - n_virt(i)*v_K(gamma, n_i_virt)(i,j);
        }
    }
    return res;
}

MatrixXd Wfg(RDM1* gamma, vector<int> omega_f, vector<int> omega_g) {
    int l = gamma->n.size();
    MatrixXd res = MatrixXd::Zero(l, l);
    VectorXd n_focc = VectorXd::Zero(l);  VectorXd n_gocc = VectorXd::Zero(l);  VectorXd nh_focc = VectorXd::Zero(l);  VectorXd nh_gocc = VectorXd::Zero(l);
    VectorXd n_fvirt = VectorXd::Zero(l); VectorXd n_gvirt = VectorXd::Zero(l); VectorXd nh_fvirt = VectorXd::Zero(l); VectorXd nh_gvirt = VectorXd::Zero(l);
    for (int p: omega_f) {
        if(gamma->n(p)>1.){
            n_focc(p) = gamma->n(p); nh_focc(p) = gamma->n(p)* sqrt(abs(2.-pow(gamma->n(p),2))); //abs to avoid numerical issues 
        }
        else{
            n_fvirt(p) = gamma->n(p); nh_fvirt(p) = gamma->n(p)* sqrt(abs(2.-pow(gamma->n(p),2)));
        }  
    }
    for (int q: omega_g) {
        if(gamma->n(q)>=1.){
            n_gocc(q) = gamma->n(q); nh_gocc(q) = gamma->n(q)* sqrt(abs(2.-pow(gamma->n(q),2)));
        }
        else{
            n_gvirt(q) = gamma->n(q); nh_gvirt(q) = gamma->n(q)* sqrt(abs(2.-pow(gamma->n(q),2)));
        }
    }
    VectorXd n_f = pow(n_focc+n_fvirt,2); VectorXd n_g = pow(n_gocc+n_gvirt,2);
    MatrixXd v_f = v_J(gamma,n_f) -1./2.*v_K(gamma, n_f); MatrixXd v_g = v_J(gamma,n_g) -1./2.*v_K(gamma, n_g); 
    MatrixXd vh_focc = v_K(gamma, nh_focc); MatrixXd vh_gocc = v_K(gamma, nh_gocc); MatrixXd vh_fvirt = v_K(gamma,nh_fvirt); MatrixXd vh_gvirt = v_K(gamma,nh_gvirt);
    for (int i = 0; i < l; i++) {
        VectorXd n_i_fvirt = VectorXd::Zero(l); if(n_fvirt(i)>0){n_i_fvirt(i) = gamma->n(i);}
        VectorXd n_i_gvirt = VectorXd::Zero(l); if(n_gvirt(i)>0){n_i_gvirt(i) = gamma->n(i);}
        VectorXd n_i_focc  = VectorXd::Zero(l); if(n_focc (i)>0){n_i_focc (i) = gamma->n(i);}
        VectorXd n_i_gocc  = VectorXd::Zero(l); if(n_gocc (i)>0){n_i_gocc(i) = gamma->n(i);}
        for (int j = 0; j < l; j++) {
            res(i,j) = n_g(i)*v_f(i,j) + n_f(i)*v_g(i,j)
                      -nh_gocc(i)*vh_focc(i,j) - nh_focc(i)*vh_gocc(i,j) 
                      -nh_gocc(i)*vh_fvirt(i,j)- nh_focc(i)*vh_gvirt(i,j)
                      -nh_gvirt(i)*vh_focc(i,j)- nh_fvirt(i)*vh_gocc(i,j)
                      +nh_gvirt(i)*vh_fvirt(i,j)+nh_fvirt(i)*vh_gvirt(i,j);
        }
    }
    return res; 
}

MatrixXd PNOF7_WK(RDM1* gamma) {
    int l = gamma->n.size(); MatrixXd W = MatrixXd::Zero(l, l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        W += Wg(gamma, gamma->omega[f]);
        for (int g = 0; g < f; g++) {
            W += Wfg(gamma, gamma->omega[f], gamma->omega[g]);
        }
    }
    return - W; //Factor -1 convention of the Functional_class
}


VectorXd dWg(RDM1* gamma, vector<int> omega) {
    int l = gamma->n.size();
    VectorXd res = VectorXd::Zero(l); VectorXd n_occ = VectorXd::Zero(l); VectorXd n_virt = VectorXd::Zero(l); 
    VectorXd dn_occ = VectorXd::Zero(l); VectorXd dn_virt = VectorXd::Zero(l);
    for (int p: omega) {
        if (gamma->n(p)>1.){
            n_occ(p) = gamma->n(p); dn_occ(p) = 1.;
        }
        else{
            n_virt(p) = gamma->n(p); dn_virt(p) = 1.;
        }
         
    }
    VectorXd n = n_virt + n_occ;
    MatrixXd v_occ = v_K(gamma,n_occ); MatrixXd v_virt = v_K(gamma,n_virt); 
    for (int i = 0; i < l; i++) {
        VectorXd n_i = VectorXd::Zero(l); if(n(i)>0){n_i(i) = 1.;}
        VectorXd n_i_virt = VectorXd::Zero(l); if(n_virt(i)>0){n_i_virt(i) = gamma->n(i);}
        res(i) += - dn_virt(i)*v_occ(i,i) - dn_occ(i)*v_virt(i,i) + dn_virt(i)*v_virt(i,i);

        //Diagonal part                                 and Diagonal correction
        res(i) += n(i)*v_J(gamma, n_i)(i,i) - dn_virt(i)*v_K(gamma, n_i_virt)(i,i);
    }
    return res;
}
double sign(double x){
    if (x>0){return 1.;}
    else{ return -1.;}
}

VectorXd dWfg(RDM1* gamma, vector<int> omega_f, vector<int> omega_g) {
    int l = gamma->n.size();
    VectorXd res = VectorXd::Zero(l);
    VectorXd n_focc = VectorXd::Zero(l);  VectorXd n_gocc = VectorXd::Zero(l);  VectorXd nh_focc = VectorXd::Zero(l);  VectorXd nh_gocc = VectorXd::Zero(l);
    VectorXd n_fvirt = VectorXd::Zero(l); VectorXd n_gvirt = VectorXd::Zero(l); VectorXd nh_fvirt = VectorXd::Zero(l); VectorXd nh_gvirt = VectorXd::Zero(l);
    VectorXd dnh_focc = VectorXd::Zero(l);  VectorXd dnh_gocc = VectorXd::Zero(l); VectorXd dnh_fvirt = VectorXd::Zero(l); VectorXd dnh_gvirt = VectorXd::Zero(l);
    for (int p: omega_f) {
        if(gamma->n(p)>=1.){
            double h = sqrt(abs(2.-pow(gamma->n(p),2)));
            n_focc(p) = gamma->n(p); nh_focc(p) = gamma->n(p)* h;
            dnh_focc(p) = h - min(pow(gamma->n(p),2)/h,1e3); 
        }
        else{
            double h = sqrt(abs(2.-pow(gamma->n(p),2)));
            n_fvirt(p) = gamma->n(p); nh_fvirt(p) = gamma->n(p)* h;
            dnh_fvirt(p) = h - min(pow(gamma->n(p),2)/h,1e3); 
        }  
    }
    for (int q: omega_g) {
        if(gamma->n(q)>1.){
            double h = sqrt(abs(2.-pow(gamma->n(q),2)));
            n_gocc(q) = gamma->n(q); nh_gocc(q) = gamma->n(q)* h;
            dnh_gocc(q) = h - min(pow(gamma->n(q),2)/h,1e3); 
        }
        else{
            double h = sqrt(abs(2.-pow(gamma->n(q),2)));
            n_gvirt(q) = gamma->n(q); nh_gvirt(q) = gamma->n(q)* h;
            dnh_gvirt(q) = h - min(pow(gamma->n(q),2)/h,1e3); 
        }
    }
    VectorXd n_f = pow(n_focc+n_fvirt,2); VectorXd n_g = pow(n_gocc+n_gvirt,2);
    VectorXd dn_f = 2.*(n_focc+n_fvirt ); VectorXd dn_g = 2.*(n_gocc+n_gvirt);
    MatrixXd v_f = v_J(gamma,n_f) -1./2.*v_K(gamma, n_f); MatrixXd v_g = v_J(gamma,n_g) -1./2.*v_K(gamma, n_g); 
    MatrixXd vh_focc = v_K(gamma, nh_focc); MatrixXd vh_gocc = v_K(gamma, nh_gocc); MatrixXd vh_fvirt = v_K(gamma,nh_fvirt); MatrixXd vh_gvirt = v_K(gamma,nh_gvirt);
    for (int i = 0; i < l; i++) {
        VectorXd n_i_fvirt = VectorXd::Zero(l); if(n_fvirt(i)>0){n_i_fvirt(i) = gamma->n(i);}
        VectorXd n_i_gvirt = VectorXd::Zero(l); if(n_gvirt(i)>0){n_i_gvirt(i) = gamma->n(i);}
        VectorXd n_i_focc  = VectorXd::Zero(l); if(n_focc (i)>0){n_i_focc (i) = gamma->n(i);}
        VectorXd n_i_gocc  = VectorXd::Zero(l); if(n_gocc (i)>0){n_i_gocc(i) = gamma->n(i);}
        
        res(i) = dn_g(i)*v_f(i,i) + dn_f(i)*v_g(i,i)
                -dnh_gocc(i)*vh_focc(i,i) - dnh_focc(i)*vh_gocc(i,i) 
                -dnh_gocc(i)*vh_fvirt(i,i)- dnh_focc(i)*vh_gvirt(i,i)
                -dnh_gvirt(i)*vh_focc(i,i)- dnh_fvirt(i)*vh_gocc(i,i)
                +dnh_gvirt(i)*vh_fvirt(i,i)+dnh_fvirt(i)*vh_gvirt(i,i);
        
    }
    return res; 
}


VectorXd PNOF7_dWK(RDM1* gamma) {
    int l = gamma->n.size(); VectorXd dW = VectorXd::Zero(l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        dW += dWg(gamma, gamma->omega[f]);
        for (int g = 0; g < f; g++) {
            dW += dWfg(gamma, gamma->omega[f], gamma->omega[g]);
        }
    }
    return - dW;
}

VectorXd PNOF7_dWK_subspace(RDM1* gamma, int g){
    int l = gamma->n.size(); VectorXd dW = VectorXd::Zero(l);
    dW += dWg(gamma, gamma->omega[g]);
    for (int f = 0; f < gamma->omega.size(); f++) {
        if (f!=g){
            dW += dWfg(gamma, gamma->omega[g], gamma->omega[f]);
        }
    }
    return - dW;
}

/* // SLOW IMPLEMENTATION - USED TO TEST
#include "../classes/Matrix_Tensor_converter.cpp"
double contract(RDM1* gamma, int p, int q, double np, double nq) {
    int l = gamma->n.size(); double res = 0;
    Tensor<double,4> T(l,l,l,l); T = TensorCast(gamma->int2e,l,l,l,l);
    for (int i1 = 0;i1 < l;i1++){
        for (int j1 = 0;j1<l;j1++){
            for (int i2 = 0;i2<l;i2++){
                for (int j2=0;j2<l;j2++){
                    res += gamma->no(i1,p)*np*gamma->no(j1,p)*
                           gamma->no(i2,q)*nq*gamma->no(j2,q)*
                           T(i1,j1,i2,j2);
                }
            }
        }
    }
    return res;
}

double contract_x(RDM1* gamma, int p, int q, double np, double nq) {
    int l = gamma->n.size(); double res = 0;
    Tensor<double,4> T(l,l,l,l); T = TensorCast(gamma->int2e_x,l,l,l,l);
    for (int i1 = 0;i1 < l;i1++){
        for (int j1 = 0;j1<l;j1++){
            for (int i2 = 0;i2<l;i2++){
                for (int j2=0;j2<l;j2++){
                    res += gamma->no(i1,p)*np*gamma->no(j1,p)*
                           gamma->no(i2,q)*nq*gamma->no(j2,q)*
                           T(i1,j1,i2,j2);
                }
            }
        }
    }
    return res;
}

double contract(RDM1* gamma, int p, double np) {
    int l = gamma->n.size(); double res = 0;
    Tensor<double,4> T(l,l,l,l); T = TensorCast(gamma->int2e,l,l,l,l);
    for (int i1 = 0;i1 < l;i1++){
        for (int j1 = 0;j1<l;j1++){
            for (int i2 = 0;i2<l;i2++){
                for (int j2=0;j2<l;j2++){
                    res += gamma->no(i1,p)*np*gamma->no(j1,p)*
                           gamma->no(i2,p)*1 *gamma->no(j2,p)*
                           T(i1,j1,i2,j2);
                }
            }
        }
    }
    return res;
}

double Eg(RDM1* gamma, vector<int> omega_g){
    int l = omega_g.size();  double res = 0;
    
    for(int p :omega_g){
        res += contract(gamma, p, pow(gamma->n(p),2));
        for(int q : omega_g){
            if (p!=q){ 
                if(gamma->n(p)>1. || gamma->n(q)>1.){
                    res -= contract_x(gamma, p, q, gamma->n(p), gamma->n(q));
                }
                else{
                    res += contract_x(gamma, p, q, gamma->n(p), gamma->n(q));
                }
            }
        }
    }
    return res;

}

double Efg(RDM1* gamma, vector<int> omega_f, vector<int> omega_g){
    double res = 0;
    for(int p : omega_f){
        for(int q : omega_g){
            res -=1./2.*contract_x(gamma, p, q, pow(gamma->n(p),2), pow(gamma->n(q),2));
            res +=contract(gamma, p, q, pow(gamma->n(p),2), pow(gamma->n(q),2));
            double hp = sqrt(2.0-pow(gamma->n(p),2)); double hq = sqrt(2.0-pow(gamma->n(q),2)); 
            if(gamma->n(p)<1. && gamma->n(q)<1.){
                res += contract_x(gamma, p, q, gamma->n(p)*hp, gamma->n(q)*hq);
            }
            else{
                res -= contract_x(gamma, p, q, gamma->n(p)*hp, gamma->n(q)*hq);
            }
        }
    }
    return res;
}

MatrixXd PNOF7_EK(RDM1* gamma) {
    int l = gamma->n.size(); MatrixXd W = MatrixXd::Zero(l, l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        W(0,0) += Eg(gamma, gamma->omega[f]); 
        for (int g = 0; g < f; g++) {
            W(0,0) += 2*Efg(gamma, gamma->omega[f], gamma->omega[g]); 
        }
    }
    return - W; 
}

VectorXd dEg(RDM1* gamma, vector<int> omega_g){
    int l = gamma->n.size();
    VectorXd res = VectorXd::Zero(l);
    for(int p :omega_g){
        res(p) += contract(gamma, p, 2*gamma->n(p));
        for(int q : omega_g){
            if (p!=q){ 
                if(gamma->n(p)>1. || gamma->n(q)>1.){
                    res(p) -= contract_x(gamma, p, q, 1., gamma->n(q)) + contract_x(gamma, q, p, gamma->n(q), 1.);
                }
                else{
                    res(p) += contract_x(gamma, p, q, 1., gamma->n(q)) + contract_x(gamma, q, p, gamma->n(q), 1.);
                }
            }
        }
    }
    return res;

}

VectorXd dEfg(RDM1* gamma, vector<int> omega_f, vector<int> omega_g){
    int l = gamma->n.size();
    VectorXd res = VectorXd::Zero(l);
    for(int p : omega_f){
        for(int q : omega_g){
            res(p) -=1./2.*contract_x(gamma, p, q, 2*gamma->n(p), pow(gamma->n(q),2)) + 1./2.*contract_x(gamma, q, p, pow(gamma->n(q),2), 2*gamma->n(p));
            res(p) += contract(gamma, p, q, 2*gamma->n(p), pow(gamma->n(q),2)) + contract(gamma, q, p, pow(gamma->n(q),2), 2*gamma->n(p));
            double hp = sqrt(2.0-pow(gamma->n(p),2)); double hq = sqrt(2.0-pow(gamma->n(q),2)); 
            if(gamma->n(p)<1. && gamma->n(q)<1.){
                res(p) += contract_x(gamma, p, q, hp - pow(gamma->n(p),2)/hp, gamma->n(q)*hq) + contract_x(gamma, q, p, gamma->n(q)*hq, hp - pow(gamma->n(p),2)/hp);
            }
            else{
                res(p) -= contract_x(gamma, p, q, hp - pow(gamma->n(p),2)/hp, gamma->n(q)*hq) + contract_x(gamma, q, p, gamma->n(q)*hq, hp - pow(gamma->n(p),2)/hp);
            }
        }
    }
    return res;
}

VectorXd PNOF7_dEK(RDM1* gamma) {
    int l = gamma->n.size(); VectorXd W = VectorXd::Zero(l);
    for (int f = 0; f < gamma->omega.size(); f++) {
        W += dEg(gamma, gamma->omega[f]); 
        for (int g = 0; g < gamma->omega.size(); g++) {
            if (f!=g){
                W += dEfg(gamma, gamma->omega[f], gamma->omega[g]); 
            }
            
        }
    }
    return 1./2.*W; 
}
*/