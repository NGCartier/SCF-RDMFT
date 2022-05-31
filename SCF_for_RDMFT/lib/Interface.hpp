#ifndef _INTERFACE_1RDM_
#define _INTERFACE_1RDM_
//see Interface.cpp
tuple<VectorXd,MatrixXd> Optimize_1RDM (string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int,
                                    int disp, double epsi, int Maxiter);
#endif