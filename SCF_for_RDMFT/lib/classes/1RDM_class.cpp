#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <float.h>
#include <fstream>
#include <iostream>
#include <iomanip>      
#include <chrono>
#include <tuple>
#include <nlopt.hpp>
#include <vector>

#include "1RDM_class.hpp"
#include "Functional_class.hpp"
#include "../Functionals/PNOF7.hpp" 
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
    vector<int> v; 
    for (int i = 0; i < l; i++) {
        v.push_back(i);
    }
    omega.push_back(v);
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
    vector<int> v;
    for (int i = 0; i < occ.size(); i++) {
        v.push_back(i);
    }
    omega.push_back(v);
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
    omega = gamma->omega;
}
RDM1::~RDM1(){}; 

/*Computes the matrix form of the 1RDM*/
MatrixXd RDM1::mat() const {
    int l = n.size(); 
    MatrixXd N (l,l); N = pow(&n,2.).asDiagonal();
    return no*N* no.transpose();
}

/*Build the ensenble of subspaces used to compute the energy by some functionals (PNOF7 for now) :
Works such that a subspace is composed of 1 occupied and any number of unoccupied natural orbitals of 
expoentioanlly decreasing occupation.*/
void RDM1::subspace() {
    // Requires 2*Nocc = N_elec (usually the cas but not for [0.66, 0.66, 0.66] for ex.)
    omega.clear();
    int l = n.size(); int Nocc = 0;
    for (int i = 0; i < l; i++) {
        if (n(i) > 1.) { Nocc++; }
    }
    int N_omega = l/Nocc; int N_res = l%Nocc;
    
    if(l >= n_elec){
        for (int i = 0; i < N_res; i++) {
            vector<int> v; int p0 = i+l-Nocc;
            v.push_back(p0); double Z = 0;
            for (int j = 1; j < N_omega+1; j++) {
                Z += exp(-j);
            }
            for (int j = 1; j < N_omega; j++) {
                int p = l-j*Nocc -i-1;
                n(p)= sqrt( (2. - pow(n(p0),2)) * exp(-j)/Z );
                v.push_back(p);
            }
            int p = N_res-i-1;
            n(p) = sqrt( (2 - pow(n(p0),2)) * exp(-N_omega)/Z );
            v.push_back(p);
            omega.push_back(v);
            
        }
        for (int i= N_res; i < Nocc;i++){
            vector<int> v; int p0 = i+l-Nocc;
            v.push_back(p0); double Z = 0;
            for (int j = 1; j < N_omega; j++) {
                Z += exp(-j);
            }
            for (int j = 1; j < N_omega; j++) {
                int p = l-j*Nocc -i-1;
                n(p) = sqrt( (2. - pow(n(p0),2)) * exp(-j)/Z );
                v.push_back(p);
            }
            omega.push_back(v);
        }
    }
    else{
        for (int i=0; i<N_res; i++){
            vector<int> v; int p0 = N_res+i; int p = N_res-i-1;
            v.push_back(p0); v.push_back(p);
            n(p) = sqrt( 2. - pow(n(p0),2) );
            omega.push_back(v);
        }
        for (int i= N_res; i < Nocc; i++){
            vector<int> v; int p0 = i+N_res;
            v.push_back(p0); n(p0) = sqrt(2.); 
            omega.push_back(v);
        }
        
    }
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
void RDM1::opti(Functional* func, int disp, double epsi, double epsi_n, double epsi_no, int maxiter){
    cout<<setprecision(-log(epsi)+1);
    auto t_init = chrono::high_resolution_clock::now();
    int k = 0; int l = n.size(); int nit= 0; int ll = l*(l-1)/2;
    double E = func->E(this); double E_bis = DBL_MAX;
    
    bool detailed_disp;
    if (disp>1){detailed_disp = true;}
    else {detailed_disp = false;}
    double epsi_no_bis = epsi_no;
    while( (abs((E_bis-E)/E)>epsi  || epsi_no_bis!=epsi_no  ) && k<maxiter){ 
        

        k++; E_bis = E; epsi_no_bis = epsi_no;
        auto t0 = chrono::high_resolution_clock::now();
        tuple<double,int> res;
        try{
            res  = opti_no(this, func, epsi_no, detailed_disp, maxiter);
            if(E>E_bis){break;}
        }
        catch(...){
            cout<<"/!\\ NO iteration interrupted due to Nlopt failure."<<endl;   
        }
        int nit_no = get<1>(res); E = get<0>(res);
        

        auto t1 = chrono::high_resolution_clock::now();
        try{
            res = opti_n(this, func, epsi_n, min(1e-9, epsi_n*1e-2), detailed_disp, maxiter);
            if(E>E_bis){break;}
        }
        catch(...){
            cout<<"/!\\ Occ iteration interrupted due to Nlopt failure."<<endl;   
        }
        int nit_n = get<1>(res); E = get<0>(res);
        auto t2 = chrono::high_resolution_clock::now();
        nit += nit_n + nit_no;
        if (disp>0){
            cout<<"Iteration "<<k <<" E="<<E<<" |grad_E|="<< (func->grad_E(this,false,false)).norm()<<endl;
            cout<<"NO opti time: "; print_t(t1,t0); cout<<" and # of iter "<< nit_no<<endl;
            cout<<"Occ opti time: "; print_t(t2,t1); cout<<" and # of iter "<< nit_n<<endl;
        }
        if (nit_n <=15 && epsi_no>epsi_n){epsi_no /=6.9;}
        
    }
    if (k==maxiter){
        cout<<"Computation did not converge"<<endl;
    }
    auto t_fin = chrono::high_resolution_clock::now();
    //if (disp>0){ 
        cout<<endl;
        cout<<"Computational time "; print_t(t_fin,t_init); cout<<" total # of iter "<<nit<<endl;
    //}
}


double norm2(VectorXd* x){
    int l = x->size();
    double res = 0;
    for (int i =0; i<l;i++){
        res += pow(x->coeff(i),2);
    }
    return sqrt(res);
}

double norm1(VectorXd* x){
    int l = x->size();
    double res = 0;
    for (int i =0;i<l;i++){
        res += abs(x->coeff(i));
    }
    return res;
}

//Structure used to pass the functional and 1RDM to NLopt methods
typedef struct{
    RDM1* gamma; Functional* func; int g;
    }data_struct;

//Default objective function for the occupations optimistion called by the NLopt minimizer
double f_n(const vector<double>& x, vector<double>& grad, void* f_data){
    int l = x.size(); 
    VectorXd n = VectorXd::Map(x.data(),l); data_struct *data = (data_struct*) f_data;
    data->gamma->set_n(n);
    MatrixXd W_J (l,l); W_J = data->func->compute_WJ(data->gamma);
    MatrixXd W_K (l,l); W_K = data->func->compute_WK(data->gamma);
    double E = data->func->E(data->gamma, &W_J, &W_K);
    VectorXd temp(l); temp = data->func->grad_E(data->gamma, &W_J, &W_K, true,false);
    grad.resize(temp.size());
    Map<VectorXd>(grad.data(),grad.size()) = temp;
    
    return E;
    };

//Objective function for the occupations optimistion called by the NLopt minimizer if subspaces are needed (i.e. PNOF7)
double f_n_subspace(const vector<double>& x, vector<double>& grad, void* f_data) {
    data_struct* data = (data_struct*)f_data;
    int l = x.size(); 
    for (int i = 0; i < l; i++) {
        data->gamma->set_ni(x[i], data->gamma->omega[data->g][i]);
    }
    MatrixXd W_J = data->func->compute_WJ(data->gamma);
    MatrixXd W_K = data->func->compute_WK(data->gamma);
    double E = data->func->E(data->gamma, &W_J, &W_K);
    VectorXd temp = data->func->grad_E_subspace(data->gamma, data->g);
    grad.resize(l);
    for (int i = 0; i < l; i++) {
        grad[i] = temp(data->gamma->omega[data->g][i]);
    }
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
tuple<double,int> opti_n(RDM1* gamma, Functional* func, double epsilon, double eta, bool disp, int maxiter){
    tuple<double, int> optmum;


    if (gamma->omega.size() == 1) {
        int l = gamma->n.size();
        double Ne = static_cast<double>(gamma->n_elec); 
        vector<double> x(l); Map<VectorXd>(x.data(),x.size()) = gamma->n; double fx; 
        nlopt::opt opti = nlopt::opt(nlopt::LD_SLSQP, l);
        epsilon = epsilon/10.0; // sum_i n-i = ne constrained by NLopt method up to epsilon**2 -> can impact the
                               // energy if one does not reduces epsilon.
        data_struct f_data;
        f_data.gamma = gamma; f_data.func = func; 

        opti.set_min_objective(f_n, &f_data);
        opti.set_xtol_rel(epsilon); opti.set_maxeval(maxiter);
        vector<double> min_n (l,-eta); vector<double> max_n(l, sqrt(2)+eta);
        opti.set_lower_bounds(min_n) ; opti.set_upper_bounds(max_n);
        opti.add_equality_constraint(ne_const, &Ne);
        nlopt::result res = opti.optimize(x, fx);
        if (disp){
            cout<<opti.get_algorithm_name()<<endl;
        }
        optmum = make_tuple(opti.last_optimum_value(), opti.get_numevals());
    }
    else { //called if func need subspaces (i.e. PNOF7)
        for (int g = 0; g < gamma->omega.size(); g++) {
            int l = gamma->omega[g].size();
            if(l>1){
                double Ne = 2.; 
                vector<double> x(l,0) ; vector<double> grad(l,0) ;double fx;
                
                for (int i = 0; i < l; i++) {
                    x[i]= gamma->n(gamma->omega[g][i]);
                }
                nlopt::opt opti = nlopt::opt(nlopt::LD_SLSQP, l);
                epsilon = epsilon / 10.0;
                data_struct f_data;
                f_data.gamma = gamma; f_data.func = func; f_data.g = g;
                opti.set_min_objective(f_n_subspace, &f_data);
                opti.set_xtol_rel(epsilon); opti.set_maxeval(maxiter);
                vector<double> min_n(l, -eta); vector<double> max_n(l, sqrt(2) + eta);
                opti.set_lower_bounds(min_n); opti.set_upper_bounds(max_n);
                opti.add_equality_constraint(ne_const, &Ne);
                nlopt::result res = opti.optimize(x, fx);
                optmum = make_tuple(opti.last_optimum_value(), opti.get_numevals());  
            }
                                
        }
    }
    return optmum;
}

MatrixXd exp_unit(VectorXd* l_theta){
    int l = ((sqrt(8*l_theta->size())+1)+1)/2; int index = 0;
    MatrixXd res (l,l); 
    for (int i=0;i<l;i++){
        for (int j=0;j<=i;j++){
            if (i==j){res(i,i) = 0;}
            else{
                res(i,j) = l_theta->coeff(index);
                res(j,i) = -l_theta->coeff(index);
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
        data->gamma->set_no(data->gamma->no*exp_unit(&l_theta));
        
        MatrixXd W_J (l,l); W_J = data->func->compute_WJ(data->gamma);
        MatrixXd W_K (l,l); W_K = data->func->compute_WK(data->gamma);
        double E = data->func->E(data->gamma, &W_J, &W_K) ; 
        grad.resize(ll) ;
        Map<VectorXd>(grad.data(),grad.size()) = data->func->grad_E(data->gamma, &W_J, &W_K, false,true);
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
tuple<double,int> opti_no(RDM1* gamma, Functional* func, double epsilon, bool disp, int maxiter){
    int l = gamma->n.size(); int ll = l*(l-1)/2;
    vector<double> x (ll,0); double fx; 
    nlopt::opt opti = nlopt::opt(nlopt::LD_LBFGS, ll);
    if (*func == PNOF7_func){ // Convergence issue for PNOF7 with BFGS (quick fix).
        opti = nlopt::opt(nlopt::LD_SLSQP, ll);
    }
    
    data_struct f_data;
    f_data.gamma = gamma; f_data.func = func; 
    
    opti.set_min_objective(f_no, &f_data);
    opti.set_xtol_rel(epsilon); opti.set_maxeval(maxiter);
    opti.set_vector_storage(20);
    nlopt::result res = opti.optimize(x, fx);
    if (disp){
        cout<<opti.get_algorithm_name()<<endl;
    }
    VectorXd l_theta = VectorXd::Map (x.data(), ll);
    gamma->set_no(gamma->no*exp_unit(&l_theta));
    return make_tuple(opti.last_optimum_value(),opti.get_numevals());
}

