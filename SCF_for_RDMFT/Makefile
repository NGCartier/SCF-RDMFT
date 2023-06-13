PYEXT = $(shell python3-config --extension-suffix)
PYFILE = -I/Users/nicolascartier/Documents/Python/anaconda3/include/python3.8
EIGFILE = -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3
FLAGS = -shared -Ofast -fsigned-zeros -std=c++17 -undefined dynamic_lookup $(python3 -m pybind11 --includes) $(PYFILE) $(EIGFILE)
FILES = ./lib/Interface.cpp ./lib/classes/1RDM_class.cpp ./lib/classes/Functional_class.cpp ./lib/numerical_deriv/numerical_deriv.cpp ./lib/classes/EBI_add.cpp 
FUNCS:= $(shell ls ./lib/Functionals/*.cpp)
FLAGS2 = -lnlopt -lm
Compute_1RDM.$(PYEXT): 
	c++  $(FLAGS) $(FILES)  $(FUNCS) $(FLAGS2) -o Compute_1RDM$(PYEXT)
clean:
	rm Compute_1RDM.$(PYEXT)