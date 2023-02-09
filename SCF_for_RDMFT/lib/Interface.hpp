#ifndef _INTERFACE_1RDM_
#define _INTERFACE_1RDM_
//see Interface.cpp
#include <map>

using namespace Eigen;
class RDM1; class Functional;
tuple<VectorXd,MatrixXd> Optimize_1RDM (string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int,
                                    int disp, double epsi, int Maxiter);

void test(VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int);
double E(string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc, MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int);

//Dictionary of functionals 
extern std::map<string, Functional> Funcs;

#endif