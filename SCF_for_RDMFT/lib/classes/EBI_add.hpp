#ifndef _EBI_ADD_hpp_
#define _EBI_ADD_hpp_
#include "Functional_class.hpp"
#include "1RDM_class.hpp"
#include<tuple>
using namespace std;
class Functional; class RDM1;

double erfinv (const double);
double arcerf (const double);
double derf (const double);
double dderf(const double);
void solve_mu_aux(RDM1*);
void solve_mu_subs_aux(RDM1*);
double sign(const double x);

#endif