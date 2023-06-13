#include <math.h>
#include <eigen3/Eigen/Core>
#include <iostream>

#include "numerical_deriv.hpp"
#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"

#include "../Functionals/PNOF7.hpp"
using namespace std;
using namespace Eigen;

#include<iomanip>
//Compute a numerical approximation of the gradiaent of the functional func at 1RDM gamma
//(used to test things)
VectorXd grad_func(Functional* func, RDM1* gamma, bool do_n, bool do_no, double epsi){
    int l = gamma->size(); int ll = l*(l+1)/2; VectorXd res (ll);
    if (do_n){
        for (int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.x(i, gamma->x(i)+epsi) ; gamma_m.x(i, gamma->x(i)-epsi);
            gamma_p.solve_mu(); gamma_m.solve_mu();
            res(i) = (func->E(&gamma_p) - func->E(&gamma_m))/(2*epsi);
        }
    }
    if (do_no){
        for (int i=l;i<ll;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m(gamma);
            VectorXd dtheta_p = VectorXd::Zero(ll-l); dtheta_p(i-l) =  epsi;
            VectorXd dtheta_m = VectorXd::Zero(ll-l); dtheta_m(i-l) = -epsi;
            gamma_p.set_no( gamma->no*exp_unit(&dtheta_p)); gamma_m.set_no( gamma->no*exp_unit(&dtheta_m));
            res(i) = (func->E(&gamma_p) - func->E(&gamma_m))/(2.*epsi);
        }
    }
    if (do_n &&(not do_no)){return res.segment(0,l);}
    if (do_no &&(not do_n)){return res.segment(l,ll-l);}
    else{return res;}
}
//Compute a numerical approximation of the Hessian of the functional func at 1RDM gamma
//(used to test things, requires analytical gradient)
MatrixXd hess_func(Functional* func, RDM1* gamma, bool do_n, bool do_no, double epsi){
    int l = gamma->size(); int ll = l*(l+1)/2; MatrixXd res =MatrixXd::Zero(ll,ll);
    if (do_n){
        for(int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.x(i, gamma->x(i)+epsi) ; gamma_m.x(i, gamma->x(i)-epsi);
            gamma_p.solve_mu(); gamma_m.solve_mu();
            VectorXd grad_p = func->grad_E(&gamma_p,true,false); VectorXd grad_m = func->grad_E(&gamma_m,true,false);
            res.block(i,0,1,l) = (grad_p-grad_m).transpose()/(2.*epsi);
        }
    }
    if (do_no){
        for(int i=l;i<ll;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            VectorXd dtheta = VectorXd::Zero(ll-l); dtheta(i-l) = epsi;
            VectorXd dtheta_m = VectorXd::Zero(ll-l); dtheta_m(i-l) = epsi;
            gamma_p.set_no( gamma->no*exp_unit(&dtheta)); gamma_m.set_no( gamma->no*exp_unit(&dtheta_m));
            VectorXd grad_p = func->grad_E(&gamma_p,false,true); VectorXd grad_m = func->grad_E(&gamma_m,false,true);
            res.block(i,l,1,ll-l) = (grad_p-grad_m).transpose()/(2.*epsi);
        }
    }
    if(do_n && do_no){
        for(int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.x(i, gamma->x(i)+epsi) ; gamma_m.x(i, gamma->x(i)-epsi);
            gamma_p.solve_mu(); gamma_m.solve_mu();
            VectorXd grad_p = func->grad_E(&gamma_p,false,true); VectorXd grad_m = func->grad_E(&gamma_m,false,true);
            res.block(i,l,1,ll-l) = (grad_p-grad_m).transpose()/(2.*epsi);
            res.block(l,i,ll-l,1) = (grad_p-grad_m)/(2.*epsi);
        }
    }
    if (do_n &&(not do_no)){return res.block(0,0,l,l);}
    if (do_no &&(not do_n)){return res.block(l,l,ll-l,ll-l);}   
    else{return res;}
}
