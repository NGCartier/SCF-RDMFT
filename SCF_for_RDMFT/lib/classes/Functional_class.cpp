#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <float.h>
#include <iostream>
#include <time.h>

using namespace std;
using namespace Eigen;

#include "1RDM_class.hpp"
#include "Functional_class.hpp"
#include "Matrix_Tensor_converter.cpp"

/*Constructor for the Functional class
\param args 8 functions defining the functional F, assumed to be of the form:
        F = WJ(fJ,gJ) -WK(fK,gK) where WJ(fJ,gJ) = sum_{mu nu kappa lambda}fJ(n)_{mu nu}gJ(n)_{lambda kappa}[mu nu|kappa lambda]
                                       WK(fK,gK) = sum-{mu nu kappa lambda}fK(n)_{lambda mu}gK(n)_{nu kappa}[mu nu|kappa lambda]
                                       (n the occupations, J Coulomb, K exchange)
        see K.J.H. "Giesbertz, Avoiding the 4-index transformation in one-body reduced density matrix functional calculations for 
            separable functionals", Phys. Chem. Chem. Phys. 18, 21024-21031 (2016)
*/

Functional::Functional(VectorXd (*f_J0)(RDM1*), VectorXd (*g_J0)(RDM1*), VectorXd (*f_K0)(RDM1*), VectorXd (*g_K0)(RDM1*), 
                    VectorXd (*df_J0)(RDM1*), VectorXd (*dg_J0)(RDM1*), VectorXd (*df_K0)(RDM1*), VectorXd (*dg_K0)(RDM1*)){
    f_J=f_J0;g_J=g_J0;f_K=f_K0;g_K=g_K0;df_J=df_J0;dg_J=dg_J0;df_K=df_K0;dg_K=dg_K0;
}

//Computes the energy of gamma for the functional
double Functional::E(RDM1* gamma) const {
    int l = gamma->n.size();
    MatrixXd W_J(l,l); W_J = this->compute_WJ(gamma); MatrixXd W_K(l,l); W_K = this->compute_WK(gamma);
    return E1(gamma) + this->E_Hxc(&W_J,&W_K);
}
double Functional::E(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K) const {
    return E1(gamma) + this->E_Hxc(W_J,W_K);
}

//Computes the gradient of the energy of gamma for the functional
//if only_n = False return the derivatives respect to the NOs
//if only_no = False return the derivatives respect to the occupations
VectorXd Functional::grad_E(RDM1* gamma, bool only_n, bool only_no) const {
    return dE1(gamma, only_n, only_no)+this->dE_Hxc(gamma, only_n, only_no);
}
VectorXd Functional::grad_E(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K, 
                        MatrixXd* v_Jf, MatrixXd* v_Jg, MatrixXd* v_Kf, MatrixXd* v_Kg,
                        bool only_n, bool only_no) const {
    return dE1(gamma, only_n, only_no)+this->dE_Hxc(gamma, W_J, W_K, v_Jf, v_Jg, v_Kf, v_Kg, only_n, only_no);
}

//Computes the functional independant part of the energy (1 electron and nuclei)
double E1(RDM1* gamma){
    return gamma->E_nuc + compute_E1(&gamma->int1e,gamma->mat());
}

//Computes the derivative of the 1 electron part of the energy
VectorXd dE1(RDM1* gamma, bool only_n, bool only_no){
    int l = gamma->n.size(); int ll = l*(l+1)/2;
    MatrixXd* H1 = &gamma->int1e ; MatrixXd N (l,l); N = pow(gamma->n,2).asDiagonal(); 
    VectorXd dE1 (ll); MatrixXd g (l,l); g = gamma->mat(); MatrixXd* C = &gamma->no; 
    MatrixXd Ct (l,l); Ct = C->transpose(); MatrixXd NC (l,l); NC = N*Ct;
    if (not only_no){
        for (int i =0; i<l;i++){
            MatrixXd dg (l,l) ; dg = 2.* (*C)* dN(&gamma->n,&Ct,i);
            dE1(i) = compute_E1(H1,dg);
        } 
    }
    if (not only_n){
        int index = l;
        for (int a =0; a<l;a++){
            for (int b =0; b<a;b++){
                MatrixXd dC (l,l); dC = dU(C,a,b); MatrixXd dCNC (l,l); dCNC = dC*NC;
                MatrixXd dg (l,l); dg = dCNC+dCNC.transpose();
                dE1(index) =  compute_E1(H1,dg);
                index++;
            }
        }
    }
    if (only_n) {return dE1.segment(0,l);}
    else{ if(only_no){return dE1.segment(l, l*(l-1)/2);}
    else{ return dE1;}
    }
    
}

// Compute the 1 electron part of the energy
double compute_E1(MatrixXd* H, MatrixXd g){
    
    int l = g.rows(); int ll = pow(l,2);
    MatrixXd res (1,1); res =  H->reshaped(1,ll) * g.reshaped(ll,1);
    return res(0,0);

}

// Compute the potential in NO basis for J and K (using f and g, see Constructor operator)
MatrixXd Functional::compute_vJ_f(RDM1* gamma) const{
    int l = gamma->n.size(); int ll = pow(l,2);
    MatrixXd f (l,l) ; f = gamma->no * this->f_J(gamma).asDiagonal() *gamma->no.transpose();
    MatrixXd res (l,l); res = ( gamma->int2e *f.reshaped(ll,1) ).reshaped(l,l);
    return 1./2.* gamma->no.transpose() *res *gamma->no;
}
MatrixXd Functional::compute_vJ_g(RDM1* gamma) const{
    int l = gamma->n.size(); int ll = pow(l,2);
    MatrixXd g (l,l) ; g = gamma->no * this->g_J(gamma).asDiagonal() *gamma->no.transpose();
    MatrixXd res (l,l); res = ( gamma->int2e *g.reshaped(ll,1) ).reshaped(l,l);
    return 1./2.* gamma->no.transpose() *res *gamma->no;
}
MatrixXd Functional::compute_vK_f(RDM1* gamma) const{ 
    int l = gamma->n.size(); int ll = pow(l, 2);
    MatrixXd f(l, l); f = gamma->no * this->f_K(gamma).asDiagonal() * gamma->no.transpose();
    MatrixXd res(l, l); res = (gamma->int2e_x * f.reshaped(ll, 1)).reshaped(l, l);
    return 1. / 2. * gamma->no.transpose() * res * gamma->no;
}
MatrixXd Functional::compute_vK_g(RDM1* gamma) const{
    int l = gamma->n.size(); int ll = pow(l, 2);
    MatrixXd g(l, l); g = gamma->no * this->g_K(gamma).asDiagonal() * gamma->no.transpose();
    MatrixXd res(l, l); res = (gamma->int2e_x * g.reshaped(ll, 1)).reshaped(l, l);
    return 1. / 2. * gamma->no.transpose() * res * gamma->no;
}

// Compute the W_J W_K matrices in NO basis
MatrixXd Functional::compute_WJ(RDM1* gamma) const{
    int l = gamma->n.size(); MatrixXd W (l,l);
    VectorXd f (l); f = this->f_J(gamma);
    VectorXd g (l); g = this->g_J(gamma);
    MatrixXd v_f (l,l); v_f = compute_vJ_f(gamma);
    MatrixXd v_g (l,l); v_g = compute_vJ_g(gamma);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = f(i)* v_g(i,j) + g(i)* v_f(i,j);
        }
    }
    return 1./2. *W;
}
MatrixXd Functional::compute_WJ(RDM1* gamma, MatrixXd* v_Jf, MatrixXd* v_Jg) const{
    int l = gamma->n.size(); MatrixXd W (l,l);
    VectorXd f (l); f = this->f_J(gamma);
    VectorXd g (l); g = this->g_J(gamma);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = f(i)* v_Jg->coeff(i,j) + g(i)* v_Jf->coeff(i,j);
        }
    }
    return 1./2. *W;
}
MatrixXd Functional::compute_WK(RDM1* gamma) const{
    int l = gamma->n.size(); MatrixXd W (l,l);
    VectorXd f (l); f = this->f_K(gamma);
    VectorXd g (l); g = this->g_K(gamma);
    MatrixXd v_f (l,l); v_f = compute_vK_f(gamma);
    MatrixXd v_g (l,l); v_g = compute_vK_g(gamma);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = f(i)* v_g(i,j) + g(i)* v_f(i,j);
        }
    }
    return 1./2. *W;
}
MatrixXd Functional::compute_WK(RDM1* gamma, MatrixXd* v_Kf, MatrixXd* v_Kg) const{
    int l = gamma->n.size(); MatrixXd W (l,l);
    VectorXd f (l); f = this->f_K(gamma);
    VectorXd g (l); g = this->g_K(gamma);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = f(i)* v_Kg->coeff(i,j) + g(i)* v_Kf->coeff(i,j);
        }
    }
    return 1./2. *W;
}

// Compute the Hatree exchange correlation energy 
double Functional::E_Hxc(MatrixXd* W_J, MatrixXd* W_K) const{
    return (*W_J).trace()-(*W_K).trace();
}
double Functional::E_Hxc(RDM1* gamma) const{
    int l = gamma->n.size();
    MatrixXd W_J (l,l); W_J = this->compute_WJ(gamma);
    MatrixXd W_K (l,l); W_K = this->compute_WK(gamma);
    return (W_J).trace()-(W_K).trace();
}

// Compute the gradiant of the Hatree exchange correlation energy 
VectorXd Functional::dE_Hxc(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->n.rows(); int ll = l*(l+1)/2; VectorXd dE2 (ll);
    MatrixXd v_Jf (l,l); v_Jf = this->compute_vJ_f(gamma); MatrixXd v_Jg (l,l); v_Jg = this->compute_vJ_g(gamma);
    MatrixXd v_Kf (l,l); v_Kf = this->compute_vK_f(gamma); MatrixXd v_Kg (l,l); v_Kg = this->compute_vK_g(gamma);

    MatrixXd W_J (l,l); W_J = this->compute_WJ (gamma, &v_Jf, &v_Jg);
    MatrixXd W_K (l,l); W_K = this->compute_WK (gamma, &v_Kf, &v_Kg);
    VectorXd df_J (l); df_J= this->df_J(gamma); VectorXd dg_J (l); dg_J= this->dg_J(gamma);
    VectorXd df_K (l); df_K= this->df_K(gamma); VectorXd dg_K (l); dg_K= this->dg_K(gamma);
    if (not only_no){
        for (int i=0;i<l;i++){
            dE2(i) = (df_J(i)*v_Jg(i,i) + dg_J(i)*v_Jf(i,i) - df_K(i)*v_Kg(i,i) - dg_K(i)*v_Kf(i,i)); //issue factor 2
        } 
    }
    if (not only_n){
        int index = l;
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                dE2(index) = 4*(W_J(j,i) - W_J(i,j) - W_K(j,i) + W_K(i,j) ) ; //issue factor 4
                index ++;
            }
        }
    }
    if(only_n){ return dE2.segment(0,l);}
    else{if(only_no){return dE2.segment(l,l*(l-1)/2);}
    else{return dE2;}}
}
VectorXd Functional::dE_Hxc(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K, 
                        MatrixXd* v_Jf, MatrixXd* v_Jg, MatrixXd* v_Kf, MatrixXd* v_Kg,
                        bool only_n, bool only_no) const{
    int l = gamma->n.rows(); int ll = l*(l+1)/2; VectorXd dE2 (ll);
    
    if (not only_no){
        VectorXd df_J (l); df_J= this->df_J(gamma); VectorXd dg_J (l); dg_J= this->dg_J(gamma);
        VectorXd df_K (l); df_K= this->df_K(gamma); VectorXd dg_K (l); dg_K= this->dg_K(gamma);
        for (int i=0;i<l;i++){
            dE2(i) = (df_J(i)* v_Jg->coeff(i,i) + dg_J(i)* v_Jf->coeff(i,i) - df_K(i)* v_Kg->coeff(i,i) - dg_K(i)* v_Kf->coeff(i,i)); //issue factor 2
        } 
    }
    if (not only_n){
        int index = l;
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                dE2(index) = 4*(W_J->coeff(j,i) - W_J->coeff(i,j) - W_K->coeff(j,i) + W_K->coeff(i,j) ); //issue factor 4
                index ++;
            }
        }
    }
    if(only_n){ return dE2.segment(0,l);}
    else{if(only_no){return dE2.segment(l,l*(l-1)/2);}
    else{return dE2;}}
}
// Auxiliary functions for the 1 electron part
MatrixXd dU(MatrixXd* C,int i,int j){
    int l = C->rows();
    MatrixXd res (l,l); res = MatrixXd::Zero(l,l);
    res.col(i) = -C->col(j); res.col(j) = C->col(i);
    return res;
}
MatrixXd dN(VectorXd* N, MatrixXd* C, int i){
    int l = C->rows();
    MatrixXd res (l,l); res = MatrixXd::Zero(l,l);
    res.row(i) = C->row(i); 
    return N->asDiagonal() *res;
}
MatrixXd outer(VectorXd v1, VectorXd v2){
    int l = v1.size();
    MatrixXd res; res = MatrixXd::Zero(l,l);
    for(int i=0;i<l;i++){
        for(int j=0;j<l;j++){
            res(i,j)= v1(i)* v2(j);
        }
    }
    return res;
}

VectorXd pow(VectorXd v, double p){
    int l = v.size(); VectorXd res (l);
    for (int i=0; i<l;i++){
        res(i) = pow(v(i),2);
    }
    return res;
}


