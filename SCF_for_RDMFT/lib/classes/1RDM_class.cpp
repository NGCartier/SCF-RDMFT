#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <float.h>
#include <iostream>
#include <iomanip>      
#include <chrono>
#include <tuple>
#include <nlopt.hpp>
#include <vector>
#include <dlib/optimization.h>
#include <dlib/global_optimization.h>

#include "1RDM_class.hpp"
#include "Functional_class.hpp"

using namespace std;
using namespace Eigen;

/*Constructor for the 1RDM
\param args ne: number of electron
            Enuc: nuclei energy 
            overlap: overlap matrix
            elec1int: 1electron integrals matrix
            elec2int: 2electrons intergrals matrix
            exch2int: 2electrons intergrals matrix permutated to give the exchange energy
\param result the corresponding 1RDM
*/
RDM1::RDM1(int ne, double Enuc,MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int, MatrixXd exch2int){
    n_elec = ne;
    E_nuc = Enuc;
    ovlp = overlap;
    int1e = elec1int;
    int2e = elec2int;
    int2e_x = exch2int;

    int l= overlap.rows();
    no = overlap.inverse().sqrt();
    if (ne>l){
        for (int i= 0;i<l; i++){
            if (i>ne){ n(i) = sqrt(2); }
            else {n(i) = 1;} 
        }
    }
    else {
        for (int i= 0;i<ne; i++){
            n(i) =1;
        } 
    } 
}
/*Constructor for the 1RDM
\param args n: vector of the sqrt of the occupations
            no: matrix of the NOs
            ne: number of electron
            Enuc: nuclei energy 
            overlap: overlap matrix
            elec1int: 1electron integrals matrix
            elec2int: 2electrons intergrals matrix
            exch2int: 2electrons intergrals matrix permutated to give the exchange energy
\param result the corresponding 1RDM
*/
RDM1::RDM1(VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int, MatrixXd exch2int){
    n_elec = ne;
    E_nuc = Enuc;
    ovlp = overlap;
    int1e = elec1int;
    int2e = elec2int;
    int2e_x = exch2int;
    n = occ;
    no = orbital_mat;

}
/*Copy a instance of the RDM1 class*/
RDM1::RDM1(const RDM1* gamma){
    n_elec = gamma->n_elec;
    E_nuc  = gamma->E_nuc;
    ovlp   = gamma->ovlp;
    int1e  = gamma->int1e;
    int2e  = gamma->int2e;
    int2e_x = gamma->int2e_x;
    n  = gamma->n;
    no = gamma->no;
}
/*Computes the matrix form of the 1RDM*/
MatrixXd RDM1::mat() const {
    int l = n.size();
    MatrixXd N (l,l); N = pow(n,2).asDiagonal();
    return no*N* no.transpose();
}
/*Prints the time lapse between t0 and t1 divided by iter (default iter=1)*/
void print_t(chrono::high_resolution_clock::time_point t1, chrono::high_resolution_clock::time_point t0, int iter){
    auto t_nano = chrono::duration_cast<chrono::nanoseconds>(t1-t0).count();
    t_nano /= iter;
    if(t_nano<1000){
        cout<<t_nano<<"ns";
        return;
    }
    auto t_micro = chrono::duration_cast<chrono::microseconds>(t1-t0).count();
    t_micro /= iter;
    t_nano -= t_micro*1000;
    if(t_micro<1000){
        cout<<t_micro<<"µs "<<t_nano<<"ns";
        return;
    }
    auto t_milli = chrono::duration_cast<chrono::milliseconds>(t1-t0).count();
    t_milli /= iter;
    t_micro -= t_milli*1000;
    if(t_milli<1000){
        cout<<t_milli<<"ms "<<t_micro<<"µs";
        return;
    }
    auto t_sec = chrono::duration_cast<chrono::seconds>(t1-t0).count();
    t_sec /= iter;
    t_milli -= t_sec*1000;
    if(t_sec<60){
        cout<<t_sec<<"s "<<t_milli<<"ms";
        return;
    }
    auto t_min = chrono::duration_cast<chrono::minutes>(t1-t0).count();
    t_min /= iter;
    t_sec -= t_min*60;
    if(t_min<60){
        cout<<t_min<<"'"<<t_sec<<"s";
        return;
    }
    auto t_hour = chrono::duration_cast<chrono::hours>(t1-t0).count();
    t_hour /= iter;
    t_min -= t_hour*60;
    cout<<t_hour<<"h"<<t_min<<"'";
}
/*Optimises the occupations (n) and NOs (no) of the 1RDM with respect to the energy minimisation
\param args func: the functional to use
            disp: if >1 displais details about the computation
            epsi: relative precision required for the optimisation
            epsi_n: relative precision required for the optimisation of the occupations (default sqrt(epsi))
            epsi_no: relative precision required for the optimisation of the NOs (default sqrt(epsi))
            maxiter: maximum number of iterations for one optimisation of the NOs/occupations, and maximum number of 
                     calls to those optimisations
the occupations and NOs are optimised in-place
*/
void RDM1::opti(Functional func, int disp, double epsi, double epsi_n, double epsi_no, int maxiter){
    
    cout<<setprecision(10);
    auto t_init = chrono::high_resolution_clock::now();
    int k = 0; int l = n.size(); int nit= 0; int ll = l*(l-1)/2;
    double E = func.E(this); double E_bis = DBL_MAX;
    
    bool detailed_disp;
    if (disp>1){detailed_disp = true;}
    else {detailed_disp = false;}
   
    while( (E_bis-E)/E>epsi && k<maxiter){
        k++; E_bis = E;
        auto t0 = chrono::high_resolution_clock::now();
        auto res  = opti_no(this, func, epsi_no, detailed_disp, maxiter);
        int nit_no = get<1>(res);

        auto t1 = chrono::high_resolution_clock::now();
        
        res = opti_n(this, func, epsi_n, min(1e-9, epsi_n*1e-2), detailed_disp, maxiter);
        int nit_n = get<1>(res); E = get<0>(res);
        auto t2 = chrono::high_resolution_clock::now();
        nit += nit_n + nit_no;
        
        if (disp>0){
            cout<<"Iteration "<<k <<" E="<<E<<" |grad_E|="<< norm2(func.grad_E(this,false,false))<<endl;
            cout<<"NO opti time: "; print_t(t1,t0); cout<<" and # of iter "<< nit_no<<endl;
            cout<<"Occ opti time: "; print_t(t2,t1); cout<<" and # of iter "<< nit_n<<endl;
        }
        
    }
    if (k==maxiter){
        cout<<"Computation did not converge"<<endl;
    }
    auto t_fin = chrono::high_resolution_clock::now();
    if (disp>0){ 
        cout<<endl;
        cout<<"Computational time "; print_t(t_fin,t_init); cout<<" total # of iter "<<nit<<endl;
    }
}


double norm2(VectorXd x){
    int l = x.size();
    double res = 0;
    for (int i =0; i<l;i++){
        res += pow(x(i),2);
    }
    return sqrt(res);
}

double norm1(VectorXd x){
    int l = x.size();
    double res = 0;
    for (int i =0;i<l;i++){
        res += abs(x(i));
    }
    return res;
}
//Structure used to pass the functional and 1RDM to NLopt methods
typedef struct{
        RDM1* gamma; Functional* func; 
    }data_struct;

//Objective function for the occupations optimistion called by the NLopt minimizer
double f_n(const vector<double>& x, vector<double>& grad, void* f_data){
    int l = x.size(); 
    VectorXd n = VectorXd::Map(x.data(),l); data_struct *data = (data_struct*) f_data;
    data->gamma->set_n(n);
    MatrixXd v_Jf (l,l); v_Jf = data->func->compute_vJ_f(data->gamma);
    MatrixXd v_Jg (l,l); v_Jg = data->func->compute_vJ_g(data->gamma);
    MatrixXd v_Kf (l,l); v_Kf = data->func->compute_vK_f(data->gamma);
    MatrixXd v_Kg (l,l); v_Kg = data->func->compute_vK_g(data->gamma);
    MatrixXd W_J (l,l); W_J = data->func->compute_WJ(data->gamma, &v_Jf, &v_Jg);
    MatrixXd W_K (l,l); W_K = data->func->compute_WK(data->gamma, &v_Kf, &v_Kg);
    double E = data->func->E(data->gamma, &W_J, &W_K);
    VectorXd temp(l); temp = data->func->grad_E(data->gamma, &W_J, &W_K, &v_Jf, &v_Jg, &v_Kf, &v_Kg, true,false);
    grad.resize(temp.size());
    Map<VectorXd>(grad.data(),grad.size()) = temp;
    
    return E;
    };

//Condition Tr{RDM1} = ne called by the NLopt minimiser
double ne_const(const vector<double>& x, vector<double>& grad, void* ne){
    int l = x.size(); double* Ne = (double*) ne;
    double res = 0;
    for (int i = 0;i<l;i++){
        res += pow(x[i],2);
        grad[i] = 2*x[i] ;
    }
    return res - *Ne;
}
/* Optimises the occupations of the 1RDM with respect to the inimisation of the energy
\param args gamma: 1RDM
            func: functional
            epsilon: required precision
            eta: acceptance on the constraint violation
            disp: get detail on the optimisation /TO DO/
            maxiter: maximum number of iterations
the occupations are optimised in-place
\param results  corresponding energy, number of iterations
*/
tuple<double,int> opti_n(RDM1* gamma, Functional func, double epsilon, double eta, bool disp, int maxiter){
    int l = gamma->n.size(); double Ne = static_cast<double>(gamma->n_elec); double* ne = &Ne;
    vector<double> x(l); Map<VectorXd>(x.data(),x.size()) = gamma->n;
    vector<double> grad (l,0); double fx; 
    nlopt::opt opti = nlopt::opt(nlopt::LD_SLSQP, l);
    
    data_struct f_data;
    f_data.gamma = gamma; f_data.func = &func; 

    opti.set_min_objective(f_n, &f_data);
    opti.set_xtol_rel(epsilon); opti.set_maxeval(maxiter);
    vector<double> min_n (l,-eta); vector<double> max_n(l, sqrt(2)+eta);
    opti.set_lower_bounds(min_n) ; opti.set_upper_bounds(max_n);
    opti.add_equality_constraint(ne_const, ne);
    nlopt::result res = opti.optimize(x, fx);
    if (disp){
        cout<<opti.get_algorithm_name()<<endl;
    }
    return make_tuple(opti.last_optimum_value(),opti.get_numevals());
}

MatrixXd exp_unit(VectorXd l_theta){
    int l = ((sqrt(8*l_theta.size())+1)+1)/2; int index = 0;
    MatrixXd res (l,l); 
    for (int i=0;i<l;i++){
        for (int j=0;j<=i;j++){
            if (i==j){res(i,i) = 0;}
            else{
                res(i,j) = l_theta(index);
                res(j,i) = -l_theta(index);
                index++;
            }
        }
    }
    return res.exp();
}
//Objective function for the Nos optimistion called by the NLopt minimizer
double f_no(const vector<double>& x, vector<double>& grad, void* f_data){
        int ll = x.size(); int l = (sqrt(8*ll+1)+1)/2;
        VectorXd l_theta = VectorXd::Map (x.data(), ll); data_struct *data = (data_struct*) f_data;
        MatrixXd NO (l,l); NO = data->gamma->no;
        data->gamma->set_no(data->gamma->no*exp_unit(l_theta));
        
        MatrixXd v_Jf (l,l); v_Jf = data->func->compute_vJ_f(data->gamma);
        MatrixXd v_Jg (l,l); v_Jg = data->func->compute_vJ_g(data->gamma);
        MatrixXd v_Kf (l,l); v_Kf = data->func->compute_vK_f(data->gamma);
        MatrixXd v_Kg (l,l); v_Kg = data->func->compute_vK_g(data->gamma);
        MatrixXd W_J (l,l); W_J = data->func->compute_WJ(data->gamma, &v_Jf, &v_Jg);
        MatrixXd W_K (l,l); W_K = data->func->compute_WK(data->gamma, &v_Kf, &v_Kg);
        double E = data->func->E(data->gamma, &W_J, &W_K) ; 
        grad.resize(ll) ;
        Map<VectorXd>(grad.data(),grad.size()) = data->func->grad_E(data->gamma, &W_J, &W_K, &v_Jf, &v_Jg, &v_Kf, &v_Kg, false,true);
        data->gamma->set_no (NO);
        return E;
    };
/* Optimises the NOs of the 1RDM with respect to the inimisation of the energy
\param args gamma: 1RDM
            func: functional
            epsilon: required precision
            disp: get detail on the optimisation /TO DO/
            maxiter: maximum number of iterations
the occupations are NOs in-place
\param results  corresponding energy, number of iterations
*/
tuple<double,int> opti_no(RDM1* gamma, Functional func, double epsilon, bool disp, int maxiter){
    int l = gamma->n.size(); int ll = l*(l-1)/2;
    vector<double> x (ll,0); double fx; 
    nlopt::opt opti = nlopt::opt(nlopt::LD_LBFGS, ll);
    
    data_struct f_data;
    f_data.gamma = gamma; f_data.func = &func; 
    
    opti.set_min_objective(f_no, &f_data);
    opti.set_xtol_rel(epsilon); opti.set_maxeval(maxiter);
    opti.set_vector_storage(20);
    nlopt::result res = opti.optimize(x, fx);
    if (disp){
        cout<<opti.get_algorithm_name()<<endl;
    }
    VectorXd l_theta = VectorXd::Map (x.data(), ll);
    gamma->set_no(gamma->no*exp_unit(l_theta));
    return make_tuple(opti.last_optimum_value(),opti.get_numevals());
}

