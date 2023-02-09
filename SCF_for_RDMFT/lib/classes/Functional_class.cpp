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

Functional::Functional(MatrixXd(*W_K)(RDM1*), VectorXd(*dW_K)(RDM1*), VectorXd(*dW_K_subspace)(RDM1*,int), bool is_J_func){
    W_K_ = W_K; dW_K_ = dW_K; dW_K_subspace_ = dW_K_subspace; is_J_func_ = is_J_func;
}

//Check wether functional needs to build the subspaces (i.e. if functionals is PNOF7 for now)
bool Functional::needs_subspace() const{
    return dW_K_subspace_ != None;
}
//Check if two Functionals are identical;
bool Functional::operator==(const Functional& func){
    return 
    W_K_  == func.W_K_ &&
    dW_K_ == func.dW_K_ &&
    dW_K_subspace_ == func.dW_K_subspace_;
}

//Computes the energy of gamma for the functional
double Functional::E(RDM1* gamma) const {
    int l = gamma->n.size();
    MatrixXd W_J(l,l); W_J = compute_WJ(gamma); MatrixXd W_K(l,l); W_K = compute_WK(gamma);
    return E1(gamma) + E_Hxc(&W_J,&W_K);
}

double Functional::E(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K) const {
    return E1(gamma) + E_Hxc(W_J,W_K);
}

//Computes the gradient of the energy of gamma for the functional
//if only_n = False return the derivatives respect to the NOs
//if only_no = False return the derivatives respect to the occupations
VectorXd Functional::grad_E(RDM1* gamma, bool only_n, bool only_no) const {
    MatrixXd W_J = compute_WJ(gamma); MatrixXd W_K = compute_WK(gamma);
    return dE1(gamma, only_n, only_no)+dE_Hxc(gamma, &W_J,&W_K, only_n, only_no);
}

VectorXd Functional::grad_E(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K, bool only_n, bool only_no) const {
    return dE1(gamma, only_n, only_no)+dE_Hxc(gamma, W_J, W_K, only_n, only_no);
}

VectorXd Functional::grad_E_subspace(RDM1* gamma, int g) const{
    //Derivative of the energy resp to the occ, for only one subspace of PNOF Omega, assuming J fuctional.
    MatrixXd W_K = compute_WK(gamma); int l = gamma->n.size();
    VectorXd dE1_bis = VectorXd::Zero(l); VectorXd temp = dE1(gamma, true, false);
    for (int i: gamma->omega[g]){
        dE1_bis(i) = temp(i);
    }
    return dE1_bis+dE_Hxc_subspace(gamma, g);
}

//Computes the functional independant part of the energy (1 electron and nuclei)
double E1(RDM1* gamma){
    MatrixXd g = gamma->mat();
    return gamma->E_nuc + compute_E1(&gamma->int1e,&g);
}

//Computes the derivative of the 1 electron part of the energy
VectorXd dE1(RDM1* gamma, bool only_n, bool only_no){
    int l = gamma->n.size(); int ll = l*(l+1)/2;
    MatrixXd* H1 = &gamma->int1e ; MatrixXd N = pow(&gamma->n,2).asDiagonal(); 
    VectorXd dE1 (ll); MatrixXd g = gamma->mat(); MatrixXd* C = &gamma->no; 
    MatrixXd Ct = C->transpose(); MatrixXd NC = N*Ct;
    if (not only_no){
        for (int i =0; i<l;i++){
            MatrixXd dg (l,l) ; dg = 2.* (*C)* dN(&gamma->n,&Ct,i);
            dE1(i) = compute_E1(H1,&dg);
        } 
    }
    if (not only_n){
        int index = l;
        for (int a =0; a<l;a++){
            for (int b =0; b<a;b++){
                MatrixXd dC = dU(C,a,b); MatrixXd dCNC = dC*NC;
                MatrixXd dg = dCNC+dCNC.transpose();
                dE1(index) =  compute_E1(H1,&dg);
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
double compute_E1(MatrixXd* H, MatrixXd* g){
    
    int l = g->rows(); int ll = pow(l,2);
    MatrixXd res (1,1); res =  H->reshaped(1,ll) * g->reshaped(ll,1);
    return res(0,0);

}
// Compute the potential in NO basis for J and K (using f and g, see Constructor operator)
MatrixXd v_J(RDM1* gamma, VectorXd* n) {
    int l = gamma->n.size(); int ll = pow(l,2);
    MatrixXd f (l,l) ; f = gamma->no * n->asDiagonal() *gamma->no.transpose();
    MatrixXd res (l,l); res = ( gamma->int2e *f.reshaped(ll,1) ).reshaped(l,l);
    return gamma->no.transpose() *res *gamma->no;
}

MatrixXd v_K(RDM1* gamma, VectorXd* n) { 
    int l = gamma->n.size(); int ll = pow(l,2);
    MatrixXd f(l,l); f = gamma->no * n->asDiagonal() * gamma->no.transpose();
    MatrixXd res(l,l); res = (gamma->int2e_x * f.reshaped(ll,1)).reshaped(l,l);
    return gamma->no.transpose() * res * gamma->no;
}

// Compute the W_J W_K matrices in NO basis
MatrixXd Functional::compute_WJ(RDM1* gamma) const{
    if (is_J_func_){ 
        int l = gamma->n.size();
        return MatrixXd::Zero(l,l);
    }
    int l = gamma->n.size(); MatrixXd W (l,l);
    VectorXd N = pow(&gamma->n,2); MatrixXd v = v_J(gamma,&N);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = N(i)* v(i,j);
        }
    }
    return W;
}

MatrixXd Functional::compute_WK(RDM1* gamma) const{
    return W_K_(gamma);
}
// Compute the derivative of W_J W_K respect to the occupations
VectorXd Functional::compute_dW_J(RDM1* gamma) const{
    if (is_J_func_){
        int l = gamma->n.size();
        return VectorXd::Zero(l);
    }
    else{
        int l = gamma->n.rows(); VectorXd dW_J (l);
        VectorXd n = pow(&gamma->n,2);
        MatrixXd v = v_J(gamma,&n);
        for (int i=0;i<l;i++){
                dW_J(i) = 2. *gamma->n(i)*v(i,i); 
        } 
        return dW_J;

    }
}

VectorXd Functional::compute_dW_K(RDM1* gamma) const{
    return dW_K_(gamma);
}

// Compute the Hatree exchange correlation energy  
double Functional::E_Hxc(MatrixXd* W_J, MatrixXd* W_K) const{
    return 1./2.*( (*W_J).trace()-(*W_K).trace() );
}

// Compute the gradiant of the Hatree exchange correlation energy gradient
VectorXd Functional::dE_Hxc(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->n.rows(); int ll = l*(l+1)/2; VectorXd dE2 (ll);  
    if (not only_no){
        dE2.segment(0,l) = compute_dW_J(gamma) - compute_dW_K(gamma);
    }
    if (not only_n){
        MatrixXd W_J = compute_WJ(gamma); MatrixXd W_K = compute_WK(gamma);
        int index = l;
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                dE2(index) = 2*(W_J(j,i) - W_J(i,j) - W_K(j,i) + W_K(i,j) ); 
                index ++;
            }
        }
    }
    if(only_n){ return dE2.segment(0,l);}
    else{if(only_no){return dE2.segment(l,l*(l-1)/2);}
    else{return dE2;}}
}

VectorXd Functional::dE_Hxc(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K, bool only_n, bool only_no) const{
    int l = gamma->n.rows(); int ll = l*(l+1)/2; VectorXd dE2 (ll);  
    if (not only_no){
        dE2.segment(0,l) = compute_dW_J(gamma) - compute_dW_K(gamma);
    }
    if (not only_n){
        int index = l;
        for (int i=0;i<l;i++){
            for (int j=0;j<i;j++){
                dE2(index) = 2.*(W_J->coeff(j,i) - W_J->coeff(i,j) - W_K->coeff(j,i) + W_K->coeff(i,j) ); 
                index ++;
            }
        }
    }
    if(only_n){ return dE2.segment(0,l);}
    else{if(only_no){return dE2.segment(l,l*(l-1)/2);}
    else{return dE2;}}
}

VectorXd Functional::dE_Hxc_subspace(RDM1* gamma, int g) const{
    return - dW_K_subspace_(gamma,g);
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

VectorXd pow(const VectorXd* v, double p){
    int l = v->size(); VectorXd res (l);
    for (int i=0; i<l;i++){
        res(i) = pow(v->coeff(i),p);
    }
    return res;
}

VectorXd None (RDM1* gamma, int g){
    return VectorXd::Zero(0);
}

