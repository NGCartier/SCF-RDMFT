#ifndef _INTERFACE_1RDM_
#define _INTERFACE_1RDM_
//see Interface.cpp
tuple<VectorXd,MatrixXd> Optimize_1RDM (string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int,
                                    int disp, double epsi, int Maxiter);

void test(VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int);
double E(string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc, MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int);
VectorXd grad(double (*)(Functional, RDM1*), Functional, RDM1*, double epsi=1e-8); VectorXd grad_func (double (*)(Functional, RDM1*), Functional, RDM1*);
#endif