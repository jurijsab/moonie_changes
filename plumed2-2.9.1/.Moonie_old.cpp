/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2015-2023 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "Bias.h"
#include "ActionRegister.h"
#include "tools/Random.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
// changes for MOONIE 31.07.2024
#include "core/ActionAtomistic.h"
#include <cmath>
#include <random>
#include <map>

/// file created for MOONIE 24.07.2024

namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS MOONIE
/*
Add Moonie.

This action can be used to create fictitious collective variables coupled to the real ones.
The coupling by rigorous imposition of the holonomic constraints between fictitious collective 
variables and the value of the collective variable, which depends on the real coordinates.
The imposition of the holonomic constraint is achieved by SHAKE and RATTLE constraining algorithms. 

The main goal of this action is to output the quantities to compute the free-energy profile. 
These quantities are Lagrangian multipliers (lambda and mu) for each collective variable. We obtain
lambda from the SHAKE iteration procedure, while the mu multiplier is computed through the RATTLE
iteration. In addition the programme also outputs the value of the det(Z)^(-0.5). The free-energy 
profile can later be recovered through post-processing.

In addition, this subroutine also performs the time propagation of the fictitious collective variables
and their corresponding velocities through the OVRVO algorithm, described in several sources, e.g.
D. A. Sivak, J. D. Chodera, and G. E. Crooks, The Journal of Physical Chemistry B, vol. 118, no. 24, pp. 6466–6474, 2014.
Furthermore, the fictitious collective variables can be thermostated at their own temperature (which should be higher than
the physical one), similarly to the TAMD algorithm, published in:
L. Maragliano and E. Vanden-Eijnden, Chemical Physics Letters, vol. 426, no. 1, pp. 168–175, 2006.  

TO DO: replace this reference with the one of the paper on Moonie, once it is published.
The whole procedure is an extension of Blue Moon formalism, presented in:
G. Ciccotti and M. Ferrario, Computation, vol. 6, no. 1, 2018.
Further details on the definition of the Z matrix can be found there.  

This action takes three compulsory parameters: "MASS", "TEMPERATURE" and "FRICTION".
"MASS" is a vectorial keyword that takes the masses of all fictitious collective variables.
"TEMPERATURE" is a scalar keyword, taking the fictitious temperature, at which the fictitious collective variables are thermostated.   
"FRICTION" is a vectorial keyword that takes the frictions of all fictitious collective variables for their thermostating.

In addition, the subroutine takes three optional parameters: "DO_ONLY_BLUE_MOON", "VALUES_COLVARS" and "PERIOD_COLVARS".
"DO_ONLY_BLUE_MOON" is a boolean keyword that introduces the option of running the original Blue Moon. If DO_ONLY_BLUE_MOON=1 the time propagation 
and thermostating of the fictitious collective variables is disabled. Instead the user MUST define the values, to which the collective variables
are to relax. To this end, "VALUES_COLVARS" is a vectorial keyword, which takes the user-defined values of all collective variables.
"PERIOD_COLVARS" is a vectorial keyword that takes the period of the collective variables, in case they are periodic. If this keyword is 0 for 
any collective variable, the program will treat this collective variable as non-periodic.

\warning
If DO_ONLY_BLUE_MOON=1 and the keyword "VALUES_COLVARS" is not defined, the program will assign the target values of collective variables to 0.         

\par Exemples
This is an example of the input file for alanine dipeptide with periodic torsional angles:
\plumedfile 
TORSION ATOMS=5,7,9,15 LABEL=phi
TORSION ATOMS=7,9,15,17 LABEL=psi

MOONIE ARG=phi,psi MASS=100,100 TEMPERATURE=3500 FRICTION=1.0,1.0 PERIOD_COLVARS=2*3.14159265,2*3.14159265

FLUSH STRIDE=100

PRINT ARG=phi,psi STRIDE=1 FILE=COLVAR

PRINT ...
  ARG=*.*
  STRIDE=10
  FILE=COLVAR_ALL
... PRINT

ENDPLUMED
\endplumedfile

This is an example of the input file for alanine dipeptide with periodic torsional angles for performing a pure Blue Moon simulation:
\plumedfile 
TORSION ATOMS=5,7,9,15 LABEL=phi
TORSION ATOMS=7,9,15,17 LABEL=psi

MOONIE ARG=phi,psi MASS=100,100 TEMPERATURE=3500 FRICTION=1.0,1.0 PERIOD_COLVARS=2*3.14159265,2*3.14159265 DO_ONLY_BLUE_MOON=1 VALUES_COLVARS=3.0,3.0

FLUSH STRIDE=100

PRINT ARG=phi,psi STRIDE=1 FILE=COLVAR

PRINT ...
  ARG=*.*
  STRIDE=10
  FILE=COLVAR_ALL
... PRINT

ENDPLUMED
\endplumedfile

*/
//+ENDPLUMEDOC

class Moonie : public Bias {
  bool firsttime;
  
  /* changes for MOONIE 02.09.2024
  declaration of auxiliary varibles and their velocities (s_aux, vs_aux)
  declaration of Lagrangian multipliers for SHAKE (lambda_multipliers) and
  for RATTLE (mu_multipliers)*/ 
  std::vector<double> s_aux;
  std::vector<double> vs_aux;
  std::vector<double> lambda_multipliers;
  std::vector<double> mu_multipliers;
  
  //declaration of accuracy at which the iterative algorithms stop
  double stop_iteration_accuracy;
  
  //declaration of auxiliary masses and Langevin thermostat related quantities
  std::vector<double> mass;
  std::vector<double> friction;
  std::vector<double> a_thermo;
  
  // changes for MOONIE 10.12.2024 
  std::vector< std::vector<double> > Z_matrix;
  double det_Z;
  double det_Z_to_minus_half;

  double kbt;
  Random rand;
  int shake_rattle_switch; // changes for MOONIE 29.08.2024
  int do_only_blue_moon;
  
  std::vector<double> period_s_aux; // changes for MOONIE 13.05.2025
  
  //declaration of output arrays
  std::vector<Value*> s_values;
  std::vector<Value*> vs_values;
  std::vector<Value*> lambda_values;
  std::vector<Value*> mu_values;
  
  // changes for MOONIE 10.12.2024 
  Value * det_Z_value;
  Value * det_Z_value_to_minus_half;
public:
  explicit Moonie(const ActionOptions&);
  void calculate() override;
  void update() override;
  static void registerKeywords(Keywords& keys);

// changes for MOONIE 31.07.2024
private:
  
  // changes for MOONIE 30.08.2024
  //declaration of the vector od ActionAtomistic; each element corresponds 
  //to the respective collective variable
  std::vector<PLMD::ActionAtomistic*> actionAtomistic_vector;

  /*declaration of the vector of gradients of collective variables at the beginning of the time step
  each element corresponds to the gradient of respective reaction coordinates
  the vector is updated at the beginning of RATTLE subroutine, as the coordinates change no longer from there on*/ 
  std::vector< std::vector<double> > derivatives_old;
  std::vector<double> current_derivative_old; //auxiliary vector for derivatives_old
  
  std::normal_distribution<double> rand_num;
  std::default_random_engine generator;
  
  // changes for MOONIE 30.10.2024
  /* declaration of the repetition_vector which takes care of the possible artificial repetition of the same particle
     within the collective variable subroutines */
  std::vector< std::vector<int> > repetition_vector;
  
  // changes for MOONIE 14.05.2025
  /* declaration of the vector of maps that will aleviate the computation of the Z matrix */
  std::vector< std::vector< std::map<int, int> > > map_vector;
  
  // changes for MOONIE 10.12.2024 
  void update_derivatives_old();
  void compute_Z();
  void compute_detZ();
  
  void shake();
  void rattle();
  
  // changes for MOONIE 16.05.2025
  void fold_if_requested(double* value, double period); 
  
};

PLUMED_REGISTER_ACTION(Moonie,"MOONIE")

void Moonie::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","MASS","specifies the masses of all auxiliary variables");
  keys.add("compulsory","TEMPERATURE","specifies the temperature of the auxiliary variables");
  keys.add("compulsory","FRICTION","add a friction of the auxiliary variables");
  keys.add("optional","DO_ONLY_BLUE_MOON","disable moonie and do only blue moon");
  keys.add("optional","VALUES_COLVARS","fixed values of collective variables for do only blue moon");
  keys.add("optional","PERIOD_COLVARS","period of each collective variable"); //if collective varable is not periodic, it's period is 0
  //keys.add("optional","TEMP","the system temperature - needed when FRICTION is present. If not provided will be taken from MD code (if available)");
  componentsAreNotOptional(keys);
  keys.addOutputComponent("_s","default","Output of the auxiliary variable for every reaction coordinate");
  keys.addOutputComponent("_lambda","default","Output of the SHAKE lagrange multiplier for every reaction coordinate");
  keys.addOutputComponent("_mu","default","Output of the RATTLE lagrange multiplier for every reaction coordinate");
  
  // changes for MOONIE 10.12.2024
  keys.addOutputComponent("detZ","default","Output of the determinant of Z");
  keys.addOutputComponent("detZtoMinusHalf","default","Output of the determinant of Z to the power of -0.5");
  /*keys.addOutputComponent("_fict","default","one or multiple instances of this quantity can be referenced elsewhere in the input file. "
                          "These quantities will named with the arguments of the bias followed by "
                          "the character string _tilde. It is possible to add forces on these variable.");
  keys.addOutputComponent("_vfict","default","one or multiple instances of this quantity can be referenced elsewhere in the input file. "
                          "These quantities will named with the arguments of the bias followed by "
                          "the character string _tilde. It is NOT possible to add forces on these variable.");*/
}

Moonie::Moonie(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao),
  firsttime(true),
  
  // changes for MOONIE 02.09.2024
  // initialization of auxiliary variables, their velicity and both lagrangian multipliers
  s_aux(getNumberOfArguments(), 0.0),
  vs_aux(getNumberOfArguments(),0.0),
  lambda_multipliers(getNumberOfArguments(),0.0),
  mu_multipliers(getNumberOfArguments(),0.0),
  stop_iteration_accuracy(0.000001),
  
  // initialisation of auxiliary masses and thermostat related quantities
  mass(getNumberOfArguments(),0.0),
  friction(getNumberOfArguments(),0.0),
  a_thermo(getNumberOfArguments(),0.0),
  rand_num(0.0,1.0),
  kbt(0.0),
  do_only_blue_moon(0),
  
  // changes for MOONIE 13.05.2025
  period_s_aux(getNumberOfArguments(), 0.0),
  
  shake_rattle_switch(0), // changes for MOONIE 29.08.2024
  
  // changes for MOONIE 10.12.2024 
  // initialisation of Z_matrix, its determinant and its determinant to the power of -0.5
  Z_matrix(getNumberOfArguments(), std::vector<double>(getNumberOfArguments(), 0.0)),
  det_Z(0.0),
  det_Z_to_minus_half(0.0),
    
  //initialization of the arrays of output values
  s_values(getNumberOfArguments(),NULL),
  lambda_values(getNumberOfArguments(),NULL),
  mu_values(getNumberOfArguments(),NULL),
  
  // changes for MOONIE 10.12.2024
  det_Z_value(NULL),
  det_Z_value_to_minus_half(NULL)
{
  // read the quantities from the imput file
  parseVector("MASS",mass);
  //parseVector("TEMPERATURE",temperature);
  parseVector("FRICTION",friction);
  
  double temp=-1.0;
  parse("TEMPERATURE",temp);
  //bool do_only_blue_moon_bool = false;
  parse("DO_ONLY_BLUE_MOON", do_only_blue_moon);
  //if(do_only_blue_moon_bool == true) do_only_blue_moon = 1;
  // changes for MOONIE 12.05.2025
  parseVector("VALUES_COLVARS", s_aux);
  
  // changes for MOONIE 13.05.2025
  parseVector("PERIOD_COLVARS", period_s_aux);
  
  /*if(do_only_blue_moon) {
  	//std::cout<<"PLUMED::Moonie:: before :: keywords.exists('VALUES_COLVARS'): "<<parseVector("VALUES_COLVARS", s_aux)<<"\n";
  	std::vector<double> s_aux_fixed = s_aux;
  	parseVector("VALUES_COLVARS", s_aux);
  	std::cout<<"PLUMED::Moonie:: s_aux[0]: "<<s_aux[0]<<" s_aux[1]: "<<s_aux[1]<<" s_aux_fixed[0]: "<<s_aux_fixed[0]<<" s_aux_fixed[1]: "<<s_aux_fixed[1]<<"\n";
  	std::cout<<"PLUMED::Moonie:: after :: keywords.exists('VALUES_COLVARS'): "<<keywords.exists("VALUES_COLVARS")<<" keywords.reserved('VALUES_COLVARS'): "<<keywords.reserved("VALUES_COLVARS")<<"\n"; 
  }*/
  
  //std::cout<<"PLUMED::Moonie:: s_aux[0]: "<<s_aux[0]<<" s_aux[1]: "<<s_aux[1]<<"\n";
  
  if(temp>=0.0) kbt=plumed.getAtoms().getKBoltzmann()*temp;
  else kbt=plumed.getAtoms().getKbT();
  checkRead();

  log.printf("  with mass");
  for(unsigned i=0; i<mass.size(); i++) log.printf(" %f",mass[i]);
  log.printf("\n");

  log.printf("  with friction");
  for(unsigned i=0; i<friction.size(); i++) log.printf(" %f",friction[i]);
  log.printf("\n");

  /*bool hasFriction=false;
  for(unsigned i=0; i<getNumberOfArguments(); ++i) if(friction[i]>0.0) hasFriction=true;

  if(hasFriction) {
    log.printf("  with friction");
    for(unsigned i=0; i<friction.size(); i++) log.printf(" %f",friction[i]);
    log.printf("\n");
  }*/

  log.printf("  and kbt");
  log.printf(" %f",kbt);
  log.printf("\n");
  
  // changes for MOONIE 31.07.2024
  //actionAtomistic = dynamic_cast<PLMD::ActionAtomistic*>(getPntrToArgument(0)->getPntrToAction());
  
  // changes for MOONIE 30.08.2024
  // fill the vector with action atomistic for each collective variable and allocate the space for the vector of current_derivatives_old
  unsigned max_size_derivatives = 0;
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
   		actionAtomistic_vector.push_back(dynamic_cast<PLMD::ActionAtomistic*>(getPntrToArgument(i)->getPntrToAction()));
   		//pos_old_vector.push_back((dynamic_cast<PLMD::ActionAtomistic*>(getPntrToArgument(i)->getPntrToAction()))->getPositions());   	
   		max_size_derivatives = std::max(max_size_derivatives, getPntrToArgument(i)->getNumberOfDerivatives());
   }
   
   current_derivative_old.reserve(max_size_derivatives);
   
  //std::cout<<"PLUMED :: Moonie.cpp :: current_derivative_old.size(): "<<current_derivative_old.size()<<"\n";
  
  /*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  	for(unsigned j=0; j<getNumberOfArguments(); ++j) {
  		std::cout<<"PLUMED :: Moonie.cpp :: i: "<< i<<" j: "<<j<<" Z_matrix[i][j]: "<<Z_matrix[i][j]<<"\n";
  	}
  }*/

  // prepare the quantities that are going to be outputted
  for(unsigned i=0; i<getNumberOfArguments(); i++) {
    /*std::string comp=getPntrToArgument(i)->getName()+"_s";
    addComponentWithDerivatives(comp);
    if(getPntrToArgument(i)->isPeriodic()) {
      std::string a,b;
      getPntrToArgument(i)->getDomain(a,b);
      componentIsPeriodic(comp,a,b);
    } else componentIsNotPeriodic(comp);
    s_values[i]=getPntrToComponent(comp);*/
    std::string comp=getPntrToArgument(i)->getName()+"_s";
    addComponent(comp);
    componentIsNotPeriodic(comp);
    s_values[i]=getPntrToComponent(comp);
    
    comp=getPntrToArgument(i)->getName()+"_lambda";
    addComponent(comp);
    componentIsNotPeriodic(comp);
    lambda_values[i]=getPntrToComponent(comp);
    
    comp=getPntrToArgument(i)->getName()+"_mu";
    addComponent(comp);
    componentIsNotPeriodic(comp);
    mu_values[i]=getPntrToComponent(comp);
  }
  
  // changes for MOONIE 10.12.2024
  std::string comp="detZ";
  addComponent(comp);
  componentIsNotPeriodic(comp);
  det_Z_value=getPntrToComponent(comp);
  
  comp="detZtoMinusHalf";
  addComponent(comp);
  componentIsNotPeriodic(comp);
  det_Z_value_to_minus_half=getPntrToComponent(comp);

  /*log<<"  Bibliography "<<plumed.cite("Iannuzzi, Laio, and Parrinello, Phys. Rev. Lett. 90, 238302 (2003)");
  if(hasFriction) {
    log<<plumed.cite("Maragliano and Vanden-Eijnden, Chem. Phys. Lett. 426, 168 (2006)");
    log<<plumed.cite("Abrams and Tuckerman, J. Phys. Chem. B 112, 15742 (2008)");
  }
  log<<"\n";*/
}

// changes for MOONIE 10.12.2024
// computation of Z matrix
// TO DO with a map
void Moonie::compute_Z() {
	double element = 0.0;
	//double element_1 = 0.0; //just for checking
	double mass_real = 1.0;
	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=i; j<getNumberOfArguments(); ++j) {
			/*for(unsigned k=0; k < repetition_vector[i].size(); k++){
				for(unsigned l=0; l < repetition_vector[j].size(); l++){
					//exclude any repetitions
					if((repetition_vector[i][k] != -1) || (repetition_vector[j][l] != -1)) continue;
					
					//exclude all the indexes that are not the same
					if(actionAtomistic_vector[i]->getAbsoluteIndex(k).index() != actionAtomistic_vector[j]->getAbsoluteIndex(l).index()) continue;
					
					mass_real = actionAtomistic_vector[i]->getMass(k);
					element += (derivatives_old[i][3*k] * derivatives_old[j][3*l] + derivatives_old[i][3*k+1] * derivatives_old[j][3*l+1] + derivatives_old[i][3*k+2] * derivatives_old[j][3*l+2])/mass_real;		
				}
			
			}*/
			
			// changes for MOONIE 14.05.2025 - map 
			for(auto map_element : map_vector[i][j]){
				int k = map_element.first;
				int l = map_element.second;
				mass_real = actionAtomistic_vector[i]->getMass(k);
				element += (derivatives_old[i][3*k] * derivatives_old[j][3*l] + derivatives_old[i][3*k+1] * derivatives_old[j][3*l+1] + derivatives_old[i][3*k+2] * derivatives_old[j][3*l+2])/mass_real;		
			}
			
			if(i == j){
				// computation of diagonal elements
				//std::cout<<"Moonie :: compute_Z in if(i == j) :: i: "<<i<<" j: "<<j<<" element: "<<element<<" Z_matrix[i][i]: "<<Z_matrix[i][i]<<"\n";
				// changes for MOONIE 12.05.2025
				if(do_only_blue_moon == 0) element += 1/mass[i];
				Z_matrix[i][i] = element;
			} else {
				// computation of off-diagonal elements
				Z_matrix[i][j] = element;
				Z_matrix[j][i] = element;
			}
			element = 0.0;
			//element_1 = 0.0; //just for checking
			
			//std::cout<<"Moonie :: compute_Z :: i: "<<i<<" j: "<<j<<"\n";
		}
	}
	
	//std::cout<<"Moonie :: Z_matrix[0][0]: "<<Z_matrix[0][0]<<" Z_matrix[0][1]: "<<Z_matrix[0][1]<<" Z_matrix[1][0]: "<<Z_matrix[1][0]<<" Z_matrix[1][1]: "<<Z_matrix[1][1]<<"\n";
}

// changes for MOONIE 10.12.2024
//computation of determinant of Z and determinant of Z to the power of -0.5
void Moonie::compute_detZ() {
  	if(getNumberOfArguments() == 1){
  		det_Z = Z_matrix[0][0];
  		det_Z_to_minus_half = 1.0/sqrt(det_Z);
  	} else if(getNumberOfArguments() == 2){
  		det_Z = Z_matrix[0][0] * Z_matrix[1][1] - Z_matrix[0][1] * Z_matrix[1][0];
  		det_Z_to_minus_half = 1.0/sqrt(det_Z);
  	} else if(getNumberOfArguments() == 3){
  		det_Z = Z_matrix[0][0] * Z_matrix[1][1] * Z_matrix[2][2] + Z_matrix[0][1] * Z_matrix[1][2] * Z_matrix[2][0] + Z_matrix[0][2] * Z_matrix[1][0] * Z_matrix[2][1] - Z_matrix[1][1] * Z_matrix[0][2] * Z_matrix[2][0] - Z_matrix[0][0] * Z_matrix[1][2] * Z_matrix[2][1] - Z_matrix[2][2] * Z_matrix[0][1] * Z_matrix[1][0];
  		det_Z_to_minus_half = 1.0/sqrt(det_Z);
  	} else if(getNumberOfArguments() == 4){
  		det_Z = Z_matrix[0][0] * Z_matrix[1][1] * Z_matrix[2][2] * Z_matrix[3][3] + Z_matrix[0][0] * Z_matrix[1][2] * Z_matrix[2][3] * Z_matrix[3][1] + Z_matrix[0][0] * Z_matrix[1][3] * Z_matrix[2][1] * Z_matrix[3][2] - Z_matrix[0][0] * Z_matrix[1][3] * Z_matrix[2][2] * Z_matrix[3][1] - Z_matrix[0][0] * Z_matrix[1][2] * Z_matrix[2][1] * Z_matrix[3][3] - Z_matrix[0][0] * Z_matrix[1][1] * Z_matrix[2][3] * Z_matrix[3][2] - Z_matrix[0][1] * Z_matrix[1][0] * Z_matrix[2][2] * Z_matrix[3][3] - Z_matrix[0][2] * Z_matrix[1][0] * Z_matrix[2][3] * Z_matrix[3][1] - Z_matrix[0][3] * Z_matrix[1][0] * Z_matrix[2][1] * Z_matrix[3][2] + Z_matrix[0][3] * Z_matrix[1][0] * Z_matrix[2][2] * Z_matrix[3][1] + Z_matrix[0][2] * Z_matrix[1][0] * Z_matrix[2][1] * Z_matrix[3][3] + Z_matrix[0][1] * Z_matrix[1][0] * Z_matrix[2][3] * Z_matrix[3][2] + Z_matrix[0][1] * Z_matrix[1][2] * Z_matrix[2][0] * Z_matrix[3][3] + Z_matrix[0][2] * Z_matrix[1][3] * Z_matrix[2][0] * Z_matrix[3][1] + Z_matrix[0][3] * Z_matrix[1][1] * Z_matrix[2][0] * Z_matrix[3][2] - Z_matrix[0][3] * Z_matrix[1][2] * Z_matrix[2][0] * Z_matrix[3][1] - Z_matrix[0][2] * Z_matrix[1][1] * Z_matrix[2][0] * Z_matrix[3][3] - Z_matrix[0][1] * Z_matrix[1][3] * Z_matrix[2][0] * Z_matrix[3][2] - Z_matrix[0][1] * Z_matrix[1][2] * Z_matrix[2][3] * Z_matrix[3][0] - Z_matrix[0][2] * Z_matrix[1][3] * Z_matrix[2][1] * Z_matrix[3][0] - Z_matrix[0][3] * Z_matrix[1][1] * Z_matrix[2][2] * Z_matrix[3][0] + Z_matrix[0][3] * Z_matrix[1][2] * Z_matrix[2][1] * Z_matrix[3][0] + Z_matrix[0][2] * Z_matrix[1][1] * Z_matrix[2][3] * Z_matrix[3][0] + Z_matrix[0][1] * Z_matrix[1][3] * Z_matrix[2][2] * Z_matrix[3][0];
  		det_Z_to_minus_half = 1.0/sqrt(det_Z);
  	
  	} else {
  		//TO DO ERROR MESSAGE
  		std::cout<<"ERROR :: Moonie :: more than 4 collective variables are not suported :( Keep calm and lower the number of your collective variables :) \n";
  		exit(-1);
  	}
  	
  	//std::cout<<"Moonie :: det_Z: "<<det_Z<<" det_Z_to_minus_half: "<<det_Z_to_minus_half<<"\n";
}

// changes for MOONIE 16.05.2025
void Moonie::fold_if_requested(double* value, double period) {
	if(period != 0) *value -= std::round(*value/period) * period;
}

void Moonie::calculate() {

  /*std::cout<<"Moonie :: calculate() :: before :: do_only_blue_moon: "<<do_only_blue_moon<<"\n";
  // moved down for MOONIE 10.12.2024
  if(do_only_blue_moon != 0) {
  	
  	return;
  }
  std::cout<<"Moonie :: calculate() :: after :: do_only_blue_moon: "<<do_only_blue_moon<<"\n";*/
  
  if(firsttime) {
    // changes for MOONIE 10.09.2024
    // the initial computation of all the quantities, which takes place at the very beginning of the simulation
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    	// changes for MOONIE 12.05.2025
    	if(do_only_blue_moon == 0) s_aux[i] = getArgument(i);
    	a_thermo[i] = exp(-getTimeStep()*getStride()*friction[i]);
    	current_derivative_old.clear();
   		for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
  			//std::cout<<"PLUMED :: Moonie.cpp :: check derivative i: "<<i<<" j: "<<j<<" getDerivative(j): "<<getPntrToArgument(i)->getDerivative(j)<<"\n";
  			current_derivative_old.push_back(getPntrToArgument(i)->getDerivative(j));
  		}
  		derivatives_old.push_back(current_derivative_old);
  		
  		
  		// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  		// we create the vector, called repetition_vector_current, and we fill it with the values of -1 
  		std::vector<Vector> &pos = actionAtomistic_vector[i]->modifyPositions();
  		std::vector<int> repetition_vector_current;
  		for(unsigned j=0; j < pos.size(); j++){
  			repetition_vector_current.push_back(-1);
  		}
  		//std::cout<<"PLUMED :: Moonie :: i: "<<i<<" repetition_vector_current.size(): "<<repetition_vector_current.size()<<" components: 0th: "<<repetition_vector_current[0]<<" components: 1st: "<<repetition_vector_current[1]<<" components: 2nd: "<<repetition_vector_current[2]<<" components: 3rd: "<<repetition_vector_current[3]<<" components: 4th: "<<repetition_vector_current[4]<<" components: 5th: "<<repetition_vector_current[5]<<"\n";
  		int index_j;
  		int index_k;
  		// We check, whether the indexes, corresponding to the particles of the vector pos, repeat.
  		// If they do, we modify the component of vector, called repetition_vector_current, form -1 to the position of the first occurence of that particle in vector pos.
  		// For example, if the perticle is written on the 0th and 4th position, the repetition_vector_current = [-1, -1, -1, -1, 0, -1, ...].
  		// Another example: the 5th element is a copy of the 1st element: repetition_vector_current = [-1, -1, -1, -1, -1, 1, -1, ...].
  		// In the case of alanine dipeptide the repetition_vector_current = [-1, -1, 1, -1, 3, -1] for both collective variables.
  		for(unsigned j=0; j < pos.size(); j++){
  			index_j = actionAtomistic_vector[i]->getAbsoluteIndex(j).index();
  			for(unsigned k=j+1; k < pos.size(); k++){
  				index_k = actionAtomistic_vector[i]->getAbsoluteIndex(k).index();
  				if(index_j == index_k){ 
  					//repetition_vector_current[j] = k;
  					repetition_vector_current[k] = j;
  					//std::cout<<"PLUMED :: Moonie :: i: "<<i<<" j: "<<j<<" k: "<<k<<" index_j: "<<index_j<<" index_k: "<<index_k<<"\n";
  				}
  			}
  		}
  		// We fill the repetition_vector with the corresponding repetition_vector_current for each reaction coordinate.
  		repetition_vector.push_back(repetition_vector_current);
  		
  		//std::cout<<"PLUMED :: Moonie :: i: "<<i<<" repetition_vector_current.size(): "<<repetition_vector_current.size()<<" components: 0th: "<<repetition_vector_current[0]<<" components: 1st: "<<repetition_vector_current[1]<<" components: 2nd: "<<repetition_vector_current[2]<<" components: 3rd: "<<repetition_vector_current[3]<<" components: 4th: "<<repetition_vector_current[4]<<" components: 5th: "<<repetition_vector_current[5]<<"\n";
  		
    }
    firsttime=false;
    
    /*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
			std::cout<<"PLUMED :: Moonie :: very beginning 1 :: i: "<<i<<" j: "<<j<<" derivatives_old[i][j]: "<<derivatives_old[i][j]<<"\n";
		}	
	}*/
	
	 // changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
	 // We correct the vector of dervivatives of reaction coordinates, due to the repetition of indeces. We add the contributions of all the non-first occurences of the same particle to the derivative corresponding to the first occurence.
	 for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < repetition_vector[i].size(); j++){
			//std::cout<<"PLUMED :: Moonie :: very beginning :: i: "<<i<<" j: "<<j<<" repetition_vector[i][j]: "<<repetition_vector[i][j]<<"\n";
			if(repetition_vector[i][j] != -1){ //TO DO the trick with multiplication
				derivatives_old[i][3*repetition_vector[i][j]] += derivatives_old[i][3*j];
				derivatives_old[i][3*repetition_vector[i][j] + 1] += derivatives_old[i][3*j + 1];
				derivatives_old[i][3*repetition_vector[i][j] + 2] += derivatives_old[i][3*j + 2];
			}
		}	
	}
	
	// changes for MOONIE 15.05.2025 - building the map_vector for optimized computation of Z matrix
	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		std::vector< std::map<int, int> > map_vector_temporary;
		for(unsigned j=0; j<getNumberOfArguments(); ++j) {
			std::map<int, int> map;
			for(unsigned k=0; k < repetition_vector[i].size(); k++){
				for(unsigned l=0; l < repetition_vector[j].size(); l++){
					//exclude any repetitions
					if((repetition_vector[i][k] != -1) || (repetition_vector[j][l] != -1)) continue;
					
					//exclude all the indexes that are not the same
					if(actionAtomistic_vector[i]->getAbsoluteIndex(k).index() != actionAtomistic_vector[j]->getAbsoluteIndex(l).index()) continue;
					map[k] = l; 
				}
			}
			map_vector_temporary.push_back(map);
		}
		map_vector.push_back(map_vector_temporary);
	}
	
	//checking the map
	/*for(unsigned i=0; i<map_vector.size(); ++i) {
		for(unsigned j=0; j<map_vector[i].size(); ++j) {
			for(auto neki : map_vector[i][j]){
				std::cout<<"map check:: i: "<<i<<" j: "<<j<<" map_vector element: "<<neki.first<<" "<<neki.second<<"\n";
			}		
		}
	}*/
	
	
	/*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
			std::cout<<"PLUMED :: Moonie :: very beginning 2 :: i: "<<i<<" j: "<<j<<" derivatives_old[i][j]: "<<derivatives_old[i][j]<<"\n";
		}	
	}*/
	
  }
  
  //std::cout<<"Moonie :: calculate() :: before :: do_only_blue_moon: "<<do_only_blue_moon<<"\n";
  // changes for MOONIE 10.12.2024
  // If we do not do the iterations of Moonie, but compute just certain properties, corresponding to the ordinary MD simulation.
  // TO DO: change the tag and implement correctly the do_only_blue_moon_option
  // changes for MOONIE 12.05.2025 - TO DO: erase if below
  /*if(do_only_blue_moon) {
  	update_derivatives_old();
  	compute_Z();
  	compute_detZ();
  	det_Z_value->set(det_Z);
	det_Z_value_to_minus_half->set(det_Z_to_minus_half);
  	return;
  }*/
  //std::cout<<"Moonie :: calculate() :: after :: do_only_blue_moon: "<<do_only_blue_moon<<"\n";
  
  
  /*double ene=0.0;
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    const double cv=difference(i,fict[i],getArgument(i));
    const double k=kappa[i];
    const double f=-k*cv;
    ene+=0.5*k*cv*cv;
    setOutputForce(i,f);
    ffict[i]=-f;
  };
  setBias(ene);
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    fict[i]=fictValue[i]->bringBackInPbc(fict[i]);
    fictValue[i]->set(fict[i]);
    vfictValue[i]->set(vfict_laststep[i]);
  }*/
  
  //std::cout<<"PLUMED:: Moonie.cpp :: shake_rattle_switch: "<<shake_rattle_switch<<" getNumberOfArguments(): "<<getNumberOfArguments()<<"\n";
  
  /*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
	for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
		std::cout<<"PLUMED :: Moonie :: before SHAKE :: i: "<<i<<" j: "<<j<<" derivatives_old[i][j]: "<<derivatives_old[i][j]<<"\n";
	}	
  }*/
  
  
  // Value shake_rattle_switch augments upon the calling of Moonie from GROMACS. 
  // Moonie subroutine is called 3 times from GROMACS in one integration step. 
  // The first time to do shake: shake_rattle_switch == 0,
  // the second time to do nothing: shake_rattle_switch == 1,
  // and the third time to do rattle: shake_rattle_switch == 2.
  // Afterwords, shake_rattle_switch is set back to zero.
  if(shake_rattle_switch == 0) {
  	shake();
  	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    	lambda_values[i]->set(lambda_multipliers[i] * 2.0/pow(getTimeStep()*getStride(), 2));
  	}
  }
  
  /*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
	for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
		std::cout<<"PLUMED :: Moonie :: after SHAKE before RATTLE :: i: "<<i<<" j: "<<j<<" derivatives_old[i][j]: "<<derivatives_old[i][j]<<"\n";
	}	
  }*/
  
  if(shake_rattle_switch == 2) {
  	rattle();
  	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    	s_values[i]->set(s_aux[i]);
    	mu_values[i]->set(mu_multipliers[i]* 2.0/(getTimeStep()*getStride()));
  	}
  	//std::cout<<"PLUMED::Moonie:: shake_rattle_switch == 2 :: s_aux[0]: "<<s_aux[0]<<" s_aux[1]: "<<s_aux[1]<<" vs_aux[0]: "<<vs_aux[0]<<" vs_aux[1]: "<<vs_aux[1]<<"\n";
  	// changes for MOONIE 10.12.2024
  	det_Z_value->set(det_Z);
	det_Z_value_to_minus_half->set(det_Z_to_minus_half);
  }
  
  /*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
	for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
		std::cout<<"PLUMED :: Moonie :: after RATTLE :: i: "<<i<<" j: "<<j<<" derivatives_old[i][j]: "<<derivatives_old[i][j]<<"\n";
	}	
  }*/
  
  
  /*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  	std::vector<Vector> vel = actionAtomistic_vector[i]->getVelocities();
  	std::vector<Vector> pos = actionAtomistic_vector[i]->getPositions();
  	
  	//actionAtomistic->positions;
  	
  	//Vector vel0 = actionAtomistic->getVelocity(0);
  	
  	//Vector pos0 = actionAtomistic->getPosition(0);
  	//Vector pos1 = actionAtomistic->getPosition(1);
  
  	std::cout<<"PLUMED :: Moonie.cpp :: argument number: "<< i <<" getArgument(i): "<<getArgument(i)<<" size_pos: "<<pos.size()<<" size_vel: "<<vel.size()<<"\n";
  	
  	//std::cout<<"PLUMED :: Moonie.cpp :: argument number: "<< i <<" getArgument(i): "<<getArgument(i)<<" pos0: "<<pos0<<" pos1: "<<pos1<<"\n";
  
  	std::vector<Vector>::iterator iter_pos = pos.begin();
  
  	for(iter_pos; iter_pos < pos.end(); iter_pos++) {
  		std::cout<<"PLUMED :: Moonie.cpp :: pos_i: "<<*iter_pos<<"\n";
  	} 
  
  	std::vector<Vector>::iterator iter_vel = vel.begin();
  
  	for(iter_vel; iter_vel < vel.end(); iter_vel++) {
  		std::cout<<"PLUMED :: Moonie.cpp :: vel_i: "<<*iter_vel<<"\n";
  	}
  
  }*/
  
  
  // changes for MOONIE 29.08.2024
  //std::cout<<"PLUMED:: Moonie.cpp :: shake_rattle_switch: "<<shake_rattle_switch<<" getNumberOfArguments(): "<<getNumberOfArguments()<<"\n";
  if(shake_rattle_switch == 2) {
  	shake_rattle_switch = -1;
  }
  shake_rattle_switch++;
  
  
  
  //std::cout<<"Moonie::calculate() \n";
}

//This function is redundant, but is here due to the fact that this function is present in all the other bias subroutines in PLUMED.
void Moonie::update() {
  /*double dt=getTimeStep()*getStride();
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    double mass=kappa[i]*tau[i]*tau[i]/(4*pi*pi); // should be k/omega**2
    double c1=std::exp(-0.5*friction[i]*dt);
    double c2=std::sqrt(kbt*(1.0-c1*c1)/mass);
// consider additional forces on the fictitious particle
// (e.g. MetaD stuff)
    ffict[i]+=fictValue[i]->getForce();

// update velocity (half step)
    vfict[i]+=ffict[i]*0.5*dt/mass;
// thermostat (half step)
    vfict[i]=c1*vfict[i]+c2*rand.Gaussian();
// save full step velocity to be dumped at next step
    vfict_laststep[i]=vfict[i];
// thermostat (half step)
    vfict[i]=c1*vfict[i]+c2*rand.Gaussian();
// update velocity (half step)
    vfict[i]+=ffict[i]*0.5*dt/mass;
// update position (full step)
    fict[i]+=vfict[i]*dt;
  }*/
  
    /*std::cout<<"PLUMED::Moonie:: doing update() \n";
	std::vector<Vector> &pos = actionAtomistic_vector[0]->getPositions_changable();
	std::vector<Vector> &vel = actionAtomistic_vector[0]->getVelocities_changable();*/
	
	/*for(unsigned j=0; j < pos.size(); j++){
		pos[j] = {1.0, 2.0, 3.0};
		vel[j] = {-1.0, -2.0, -3.0};
	}*/
	
	//actionAtomistic_vector[0]->updatePositions();
  	//actionAtomistic_vector[0]->updateVelocities();
  
  //std::cout<<"Moonie::update() \n";
  
}

//This subroutine executes the "SHAKE" iteration (i.e. the iteration of position and correction of 
// the velocities at half timestep)
void Moonie::shake(){
	double dt=getTimeStep()*getStride();
	//std::cout<<"PLUMED::Moonie:: doing shake \n";
	
	// thermostating veocities of auxilliary variables (vs_aux) and performing
	// the first 2 steps of Velocity-Verlet integration for auxilliary veriables
	// changes for MOONIE 12.05.2025
	if(do_only_blue_moon == 0){
		for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  			actionAtomistic_vector[i]->share();
  			vs_aux[i] *= sqrt(a_thermo[i]);
  			vs_aux[i] += sqrt(kbt*(1-a_thermo[i])/mass[i]) * rand_num(generator);
  			s_aux[i] += dt*vs_aux[i];
  			//s_aux[i] = 0.12; //0.09572;
  			//s_aux[i] = 0.27;
  		}	
  	}
  	
  	/*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  		std::vector<Vector> &pos = actionAtomistic_vector[i]->modifyPositions();
  		
  		for(unsigned j=0; j < pos.size(); j++){
  			std::cout<<"PLUMED :: shake() beginning :: j: "<<j<<" pos[j]: "<<pos[j]<<" getPosition(j): "<<actionAtomistic_vector[i]->getPosition(j)<<"\n";
  			
  			pos[j][0] += 1.0;
  			pos[j][1] += 1.0;
  			pos[j][2] += 1.0;
  		}
  		
  		actionAtomistic_vector[i]->applyPositions();
  		actionAtomistic_vector[i]->updatePositions();
  		//actionAtomistic_vector[i]->retrieveAtoms();
  		
  		for(unsigned j=0; j < pos.size(); j++){
  			std::cout<<"PLUMED :: shake() end :: j: "<<j<<" pos[j]: "<<pos[j]<<" getPosition(j): "<<actionAtomistic_vector[i]->getPosition(j)<<"\n";	
  		}
  	}*/
  	
  	double mass_real = 1.0;
  	
  	// first step of SHAKE
  	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  		lambda_multipliers[i] = 0.0; //TO DO: comment out this line and take as the first approximatoion of lambda its value from the end of the previous time step
  		actionAtomistic_vector[i]->retrieveAtoms();
  		std::vector<Vector> &pos = actionAtomistic_vector[i]->modifyPositions();
  		std::vector<Vector> &vel = actionAtomistic_vector[i]->modifyVelocities();
  		
  		
  		//std::vector<Vector> &pos_changable = actionAtomistic_vector[i]->getPositions_changable();
  		//std::vector<AtomNumber> indeces = actionAtomistic_vector[i]->getAbsoluteIndexes();
  		
  		/*std::vector<PLMD::AtomNumber>::iterator iter_index = indeces.begin();
  		for(iter_index; iter_index < indeces.end(); iter_index++){
  			std::cout<<"PLUMED :: shake() beginning changable check :: index: "<<iter_index->index()<<" pos_changable: "<<pos_changable[iter_index->index()]<<"\n";
  		}*/
  		
  		// the first step of SHAKE algorithm
  		for(unsigned j=0; j < pos.size(); j++){
  			// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  			if(repetition_vector[i][j] != -1){ 
  				pos[j] = pos[repetition_vector[i][j]];
  				continue;
  			}
  			
  			mass_real = actionAtomistic_vector[i]->getMass(j);
  			
  			/*std::cout<<"PLUMED :: shake() beginning :: j: "<<j<<" pos[j]: "<<pos[j]<<" getPosition(j): "<<actionAtomistic_vector[i]->getPosition(j)<<" vel[j]: "<<vel[j]<<" index: "<<actionAtomistic_vector[i]->getAbsoluteIndex(j).index()<<"\n";*/
  			
  			AtomNumber index_current = actionAtomistic_vector[i]->getAbsoluteIndex(j);
  			
  			//std::cout<<"PLUMED :: shake() beginning :: j: "<<j<<" index_current: "<<index_current.index()<<" getGlobalPosition(index_current): "<<actionAtomistic_vector[i]->getGlobalPosition(index_current)<<"\n";
  			
  			//unsigned current_index = actionAtomistic_vector[i]->getAbsoluteIndex(j);
  			
  			pos[j][0] += derivatives_old[i][j*3] * lambda_multipliers[i]/mass_real;
  			pos[j][1] += derivatives_old[i][j*3 + 1] * lambda_multipliers[i]/mass_real;
  			pos[j][2] += derivatives_old[i][j*3 + 2] * lambda_multipliers[i]/mass_real;
  		}
  		
  		// changes for MOONIE 12.05.2025
  		if(do_only_blue_moon == 0) s_aux[i] -= lambda_multipliers[i]/mass[i];
  		
  		actionAtomistic_vector[i]->applyPositions();
  		actionAtomistic_vector[i]->updatePositions();  		
  		//actionAtomistic_vector[i]->retrieveAtoms();
  		getPntrToArgument(i)->clearDerivatives();
  		actionAtomistic_vector[i]->calculate();
  	}
  	
  	// SHAKE iteration
  	bool stop_iteration = false;
  	double stop_iteration_condition = 0.0;
  	double sum_denominator = 0.0;
  	double lambda_multiplier_increment = 0.0;
  	double numerator = 0.0;
  	int iteration_number = 0;
  	
  	
  	while(stop_iteration == false) {
  		stop_iteration_condition = 0.0;
  		iteration_number += 1;
  		/*if(iteration_number > 2000) {
  			std::cout<<"PLUMED :: shake() :: exiting here \n";
  			exit(3); //TO DO: comment out
  		}*/

  		for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  			actionAtomistic_vector[i]->retrieveAtoms();
  			std::vector<Vector> &pos = actionAtomistic_vector[i]->modifyPositions();
  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: shake() :: iter before :: i: "<<i<<" pos[0]: "<<pos[0]<<" pos[1]: "<<pos[1]<<"\n";
  			/*for(unsigned k=0; k < pos.size(); k++){
  				std::cout<<"PLUMED :: shake() before :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" getPosition(k): "<<actionAtomistic_vector[i]->getPosition(k)<<"\n";
  			}*/
  			
  			//compute new derivatives
  			std::vector<double> new_derivative;
   			for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
  				//std::cout<<"PLUMED :: Moonie.cpp :: check derivative i: "<<i<<" j: "<<j<<" getDerivative(j): "<<getPntrToArgument(i)->getDerivative(j)<<"\n";
  				new_derivative.push_back(getPntrToArgument(i)->getDerivative(j));
  				//std::cout<<"PLUMED :: Moonie.cpp :: check derivative i: "<<i<<" j: "<<j<<" getDerivative(j): "<<getPntrToArgument(i)->getDerivative(j)<<" new_derivative[j]: "<<new_derivative[j]<<" size_derivatives: "<<new_derivative.size()<<"\n";
  			}
  			
  			// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  			for(unsigned k=0; k < repetition_vector[i].size(); k++){
  				if(repetition_vector[i][k] != -1){
  					new_derivative[3*repetition_vector[i][k]] += new_derivative[3*k];
  					new_derivative[3*repetition_vector[i][k] + 1] += new_derivative[3*k + 1];
  					new_derivative[3*repetition_vector[i][k] + 2] += new_derivative[3*k + 2];
  				}
  			}
  			
  			/*for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
  				std::cout<<"PLUMED :: Moonie.cpp :: check derivative i: "<<i<<" j: "<<j<<" getDerivative(j): "<<getPntrToArgument(i)->getDerivative(j)<<" new_derivative[j]: "<<new_derivative[j]<<" size_derivatives: "<<new_derivative.size()<<"\n";
  			}*/
  			
  			sum_denominator = 0.0;
  			//ignoring the box derivatives
  			//compute the first part of the denominator of the multiplier increment
  			for(unsigned k=0; k < pos.size(); k++){
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1) continue;
  				mass_real = actionAtomistic_vector[i]->getMass(k);
  				sum_denominator += (derivatives_old[i][k*3] * new_derivative[k*3] + derivatives_old[i][k*3 +1] * new_derivative[k*3 +1] + derivatives_old[i][k*3 +2] * new_derivative[k*3 +2])/mass_real;
  				//std::cout<<"PLUMED :: Moonie :: sum_denominator :: i: "<<i<<" k: "<<k<<" deriv_old: "<<derivatives_old[i][k*3]<<", "<<derivatives_old[i][k*3 +1]<<", "<<derivatives_old[i][k*3 +2]<<" new_derivative: "<<new_derivative[k*3]<<", "<<new_derivative[k*3 +1]<<", "<<new_derivative[k*3 +2]<<"\n";
  			}
  			
  			//UNCOMMENT
  			//compute the second part of the denominator of the multiplier increment
  			sum_denominator += 1.0/mass[i];
  			
  			//std::cout<<"PLUMED :: shake() before increment :: i: "<<i<<" getArgument(i): "<<getArgument(i)<<" s_aux[i]: "<<s_aux[i]<<" sum_denominator: "<<sum_denominator<<"\n";
  			//std::cout<<"PLUMED :: shake() before increment :: i: "<<i<<" std::round((getArgument(i) - s_aux[i])/(2*M_PI)): "<<std::round((getArgument(i) - s_aux[i])/(2*M_PI))<<"\n";
  			
  			// compute the numerator of multiplier increment	
  			numerator = getArgument(i) - s_aux[i]; //- std::round((getArgument(i) - s_aux[i])/(2*M_PI)) * 2*M_PI; 		
  			//lambda_multiplier_increment = -(getArgument(i) - s_aux[i])/sum_denominator; //(getArgument(i) - s_aux[i])/sum_denominator;
  			// changes for MOONIE 13.05.2025
  			//if(period_s_aux[i] != 0) numerator -= std::round(numerator/period_s_aux[i]) * period_s_aux[i]; 
  			
  			// changes for MOONIE 16.05.2025
  			fold_if_requested(&numerator, period_s_aux[i]);
  			
  			// set the lambda increment
  			// update lambda
  			lambda_multiplier_increment = -numerator/sum_denominator;
  			lambda_multipliers[i] += lambda_multiplier_increment;
  			
  			// correct/update the physical degrees of freedom, i.e. the coordinates
  			for(unsigned k=0; k < pos.size(); k++){
  				//std::cout<<"PLUMED :: shake() before position correction :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" getPosition(k): "<<actionAtomistic_vector[i]->getPosition(k)<<"\n";
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1){ 
  					pos[k] = pos[repetition_vector[i][k]];
  					//std::cout<<"PLUMED :: shake() position hacking :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" getPosition(k): "<<actionAtomistic_vector[i]->getPosition(k)<<"\n";
  					continue;
  				}
  				mass_real = actionAtomistic_vector[i]->getMass(k);
  				pos[k][0] += derivatives_old[i][k*3] * lambda_multiplier_increment/mass_real;
  				pos[k][1] += derivatives_old[i][k*3 + 1] * lambda_multiplier_increment/mass_real;
  				pos[k][2] += derivatives_old[i][k*3 + 2] * lambda_multiplier_increment/mass_real;
  				
  				//std::cout<<"PLUMED :: shake() after position correction :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" getPosition(k): "<<actionAtomistic_vector[i]->getPosition(k)<<"\n";
  			}
  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: shake(): before s update iter :: i: "<<i<<" s_aux[i]: "<<s_aux[i]<<"\n";
  			
  			//correct/update the auxilliary varaibles
  			// changes for MOONIE 12.05.2025
  			if(do_only_blue_moon == 0) s_aux[i] -= lambda_multiplier_increment/mass[i];
  			
  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: shake(): after s update iter :: i: "<<i<<" s_aux[i]: "<<s_aux[i]<<"\n";
  			 
  			// TO DO: put this in a separate function
  			// PBC for the ausiliary varaibles - fold s_aux 			
  			// changes for MOONIE 12.05.2025 and 13.05.2025
  			//if(do_only_blue_moon == 0 && period_s_aux[i] != 0) s_aux[i] -= std::round(s_aux[i]/period_s_aux[i]) * period_s_aux[i]; 
  			
  			// changes for MOONIE 16.05.2025
  			if(do_only_blue_moon == 0) fold_if_requested(&s_aux[i], period_s_aux[i]);
  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: shake() :: iter after :: i: "<<i<<" pos[0]: "<<pos[0]<<" pos[1]: "<<pos[1]<<"\n";
  			
  			/*for(unsigned k=0; k < pos.size(); k++){
  				std::cout<<"PLUMED :: shake() after :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" getPosition(k): "<<actionAtomistic_vector[i]->getPosition(k)<<" s_aux[i]: "<<s_aux[i]<<" lambda_multipliers[i]: "<<lambda_multipliers[i]<<" lambda_multiplier_increment: "<<lambda_multiplier_increment<<"\n";
  			}*/
  			  			
  			
  			actionAtomistic_vector[i]->applyPositions();
  			actionAtomistic_vector[i]->updatePositions();
  			
  			//actionAtomistic_vector[i]->retrieveAtoms();
  			getPntrToArgument(i)->clearDerivatives();
  			actionAtomistic_vector[i]->calculate();
  			
  			/*for(unsigned k=0; k < pos.size(); k++){
  				std::cout<<"PLUMED :: shake() after after :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" getPosition(k): "<<actionAtomistic_vector[i]->getPosition(k)<<"\n";
  			}*/
  			
  		}
  		
  		// compute the end-of-iteration condition
  		for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  			//std::cout<<"PLUMED :: shake() :: i: "<<i<<" before stop_iteration_condition: "<<stop_iteration_condition<<" getArgument(i): "<<getArgument(i)<<" s_aux[i]: "<<s_aux[i]<<"\n";
  			stop_iteration_condition += (getArgument(i) - s_aux[i])*(getArgument(i) - s_aux[i]);
  			//std::cout<<"PLUMED :: shake() :: i: "<<i<<" after stop_iteration_condition: "<<stop_iteration_condition<<" getArgument(i): "<<getArgument(i)<<" s_aux[i]: "<<s_aux[i]<<"\n";
  		}
  		
  		
  	
  		//END ITERATION CONDITION - check
  		if(stop_iteration_condition < stop_iteration_accuracy) stop_iteration = true;
  		
  	}
  	
  	
  	//CORRECT VELOCITIES
  	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		actionAtomistic_vector[i]->retrieveAtoms();
		std::vector<Vector> &vel = actionAtomistic_vector[i]->modifyVelocities();
		std::vector<Vector> &pos = actionAtomistic_vector[i]->modifyPositions();
		
  		for(unsigned k=0; k < vel.size(); k++){
  			//std::cout<<"PLUMED :: shake() before velocity correction :: i: "<<i<<" k: "<<k<<" vel[k]: "<<vel[k]<<"\n";
  			// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  			if(repetition_vector[i][k] != -1){ 
  				vel[k] = vel[repetition_vector[i][k]];
  				//std::cout<<"PLUMED :: shake() velocity hacking :: i: "<<i<<" k: "<<k<<" vel[k]: "<<vel[k]<<"\n";
  				continue;
  			}
  				
  			//std::cout<<"PLUMED :: shake() end pos not velocities :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" getPosition(k): "<<actionAtomistic_vector[i]->getPosition(k)<<" vel[k]: "<<vel[k]<<" lambda_multipliers[i]: "<<lambda_multipliers[i]<<"\n";
  				
  			mass_real = actionAtomistic_vector[i]->getMass(k);
  			vel[k][0] += derivatives_old[i][k*3] * lambda_multipliers[i]/(mass_real * dt);
  			vel[k][1] += derivatives_old[i][k*3 + 1] * lambda_multipliers[i]/(mass_real * dt);
  			vel[k][2] += derivatives_old[i][k*3 + 2] * lambda_multipliers[i]/(mass_real * dt);
  			
  			//std::cout<<"PLUMED :: shake() after velocity correction :: i: "<<i<<" k: "<<k<<" vel[k]: "<<vel[k]<<"\n";
  				
  			//std::cout<<"PLUMED :: shake() end end :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" getPosition(k): "<<actionAtomistic_vector[i]->getPosition(k)<<" vel[k]: "<<vel[k]<<" lambda_multipliers[i]: "<<lambda_multipliers[i]<<"\n";
  		}
  		
  		// changes for MOONIE 12.05.2025
  		if(do_only_blue_moon == 0) vs_aux[i] -= lambda_multipliers[i]/(mass[i] * dt);
  		
  		actionAtomistic_vector[i]->applyVelocities();
  		actionAtomistic_vector[i]->updateVelocities();
  		//actionAtomistic_vector[i]->retrieveAtoms();
  		
  		//std::cout<<"PLUMED :: Moonie.cpp :: shake() end :: i: "<<i<<" s_aux[i]: "<<s_aux[i]<<"\n";
  	}
  	 	
}

// function that updates derivatives old array
// changes for MOONIE 10.12.2024
void Moonie::update_derivatives_old(){
	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		actionAtomistic_vector[i]->share();
		current_derivative_old.clear();
   		for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
  			//std::cout<<"PLUMED :: Moonie.cpp :: check derivative i: "<<i<<" j: "<<j<<" getDerivative(j): "<<getPntrToArgument(i)->getDerivative(j)<<"\n";
  			current_derivative_old.push_back(getPntrToArgument(i)->getDerivative(j));
  		}
  		derivatives_old[i] = current_derivative_old;
	}
	
	/*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
			std::cout<<"PLUMED :: Moonie.cpp :: check derivative i: "<<i<<" j: "<<j<<" derivatives_old[i][j]: "<<derivatives_old[i][j]<<"\n";
		}
	}*/
	
	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < repetition_vector[i].size(); j++){
			//std::cout<<"PLUMED :: Moonie :: very beginning :: i: "<<i<<" j: "<<j<<" repetition_vector[i][j]: "<<repetition_vector[i][j]<<"\n";
			if(repetition_vector[i][j] != -1){
				derivatives_old[i][3*repetition_vector[i][j]] += derivatives_old[i][3*j];
				derivatives_old[i][3*repetition_vector[i][j] + 1] += derivatives_old[i][3*j + 1];
				derivatives_old[i][3*repetition_vector[i][j] + 2] += derivatives_old[i][3*j + 2];
			}
		}	
	}
	
	/*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
			std::cout<<"PLUMED :: Moonie.cpp :: check derivative second instance i: "<<i<<" j: "<<j<<" derivatives_old[i][j]: "<<derivatives_old[i][j]<<"\n";
		}
	}*/
}

// This subroutine executes the "RATTLE" iteration (i.e. the iteration of velocities at the end of timestep)
void Moonie::rattle(){
	//std::cout<<"PLUMED::Moonie:: doing rattle \n";
	
	// thermostat auxilliary variables
  	// changes for MOONIE 10.09.2024
    // changes for MOONIE 12.05.2025
    if(do_only_blue_moon == 0) {
    	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
			vs_aux[i] *= sqrt(a_thermo[i]);
  			vs_aux[i] += sqrt(kbt*(1-a_thermo[i])/mass[i]) * rand_num(generator);
  			//std::cout<<"PLUMED::Moonie:: rattle() :: i: "<<i<<" a_thermo[i]: "<<a_thermo[i]<<" rand_num(generator): "<<rand_num(generator)<<"\n";
			
  			
  			
  			// moved to a separate function for MOONIE on 10.12.2024
  			/*actionAtomistic_vector[i]->share();
  			//std::cout<<"PLUMED :: Moonie.cpp :: RATTLE :: check positions: pos_new0: "<<actionAtomistic_vector[i]->getPosition(0)<<" pos_new1: "<<actionAtomistic_vector[i]->getPosition(1)<<"\n";
  			current_derivative_old.clear();
   			for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
  				//std::cout<<"PLUMED :: Moonie.cpp :: check derivative i: "<<i<<" j: "<<j<<" getDerivative(j): "<<getPntrToArgument(i)->getDerivative(j)<<"\n";
  				current_derivative_old.push_back(getPntrToArgument(i)->getDerivative(j));
  			}
  			derivatives_old[i] = current_derivative_old;*/
    	}
    }
    
    // moved to a separate function for MOONIE on 10.12.2024
    // changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
	/*for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < repetition_vector[i].size(); j++){
			//std::cout<<"PLUMED :: Moonie :: very beginning :: i: "<<i<<" j: "<<j<<" repetition_vector[i][j]: "<<repetition_vector[i][j]<<"\n";
			if(repetition_vector[i][j] != -1){
				derivatives_old[i][3*repetition_vector[i][j]] += derivatives_old[i][3*j];
				derivatives_old[i][3*repetition_vector[i][j] + 1] += derivatives_old[i][3*j + 1];
				derivatives_old[i][3*repetition_vector[i][j] + 2] += derivatives_old[i][3*j + 2];
			}
		}	
	}*/
	
	// update of the derivatives
	// compute Z matrix and is determinant
	// changes for MOONIE 10.12.2024
	update_derivatives_old();
	compute_Z();
  	compute_detZ();
	
   
   	double mass_real = 1.0;
   	
   	// first step of rattle
   	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		mu_multipliers[i] = 0.0; //TO DO: comment out this line and take as the first approximatoion of mu its value from the end of the previous time step
		actionAtomistic_vector[i]->retrieveAtoms();
  		std::vector<Vector> &vel = actionAtomistic_vector[i]->modifyVelocities();
  		
  		for(unsigned j=0; j < vel.size(); j++){
  			// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  			if(repetition_vector[i][j] != -1){ 
  				vel[j] = vel[repetition_vector[i][j]];
  				continue;
  			}
  			mass_real = actionAtomistic_vector[i]->getMass(j);
  			
  			vel[j][0] += derivatives_old[i][j*3] * mu_multipliers[i]/mass_real;
  			vel[j][1] += derivatives_old[i][j*3 + 1] * mu_multipliers[i]/mass_real;
  			vel[j][2] += derivatives_old[i][j*3 + 2] * mu_multipliers[i]/mass_real;
  		}
  		
  		// changes for MOONIE 12.05.2025
  		if(do_only_blue_moon == 0) vs_aux[i] -= mu_multipliers[i]/mass[i];
  				
  		actionAtomistic_vector[i]->applyVelocities();
  		actionAtomistic_vector[i]->updateVelocities();
  	}
  	
  	// rattle() iteration
  	bool stop_iteration = false;
  	double stop_iteration_condition = 0.0;
  	double stop_iteration_condition_each_cv = 0.0;
  	
  	double sum_denominator = 0.0;
  	double mu_multiplier_increment = 0.0;
  	
  	while(stop_iteration == false) {
  		//stop_iteration_condition = 0.0;
  		
  		for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  			actionAtomistic_vector[i]->retrieveAtoms();
  			std::vector<Vector> &vel = actionAtomistic_vector[i]->modifyVelocities();
  			//std::vector<Vector> &pos = actionAtomistic_vector[i]->modifyPositions();
  			
  			//compute the first part of the denominator of the multiplier increment
  			sum_denominator = 0.0;
  			for(unsigned k=0; k < vel.size(); k++){
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1) continue;
  				
  				mass_real = actionAtomistic_vector[i]->getMass(k);
  				sum_denominator += (derivatives_old[i][k*3] * derivatives_old[i][k*3] + derivatives_old[i][k*3 +1] * derivatives_old[i][k*3 +1] + derivatives_old[i][k*3 +2] * derivatives_old[i][k*3 +2])/mass_real;
  				//std::cout<<"PLUMED :: Moonie :: rattle() :: sum_denominator :: i: "<<i<<" k: "<<k<<" deriv_old: "<<derivatives_old[i][k*3]<<", "<<derivatives_old[i][k*3 +1]<<", "<<derivatives_old[i][k*3 +2]<<" sum_denominator: "<<sum_denominator<<"\n";
  			}
  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: rattle() :: i: "<<i<<" sum_denominator: "<<sum_denominator<<"\n";
  			
  			// compute the second part of the denominator of the multiplier increment
  			sum_denominator += 1.0/mass[i];
  			
  			// compute the first part of numerator of the multiplier increment
  			mu_multiplier_increment = 0.0;
  			for(unsigned k=0; k < vel.size(); k++){			
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1) continue;
  				mu_multiplier_increment -= (derivatives_old[i][k*3] * vel[k][0] + derivatives_old[i][k*3 + 1] * vel[k][1] + derivatives_old[i][k*3 + 2] * vel[k][2]);
  				
  				//std::cout<<"PLUMED :: Moonie.cpp :: rattle() :: i: "<<i<<" k: "<<k<<" scalar product: "<<derivatives_old[i][k*3] * vel[k][0] + derivatives_old[i][k*3 + 1] * vel[k][1] + derivatives_old[i][k*3 + 2] * vel[k][2]<<"\n";
  			
  			}
  			
  			// compute the second part of numerator of the multiplier increment
  			// changes for MOONIE 12.05.2025
  			if(do_only_blue_moon == 0) mu_multiplier_increment += vs_aux[i];
  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: rattle() :: i: "<<i<<" sum scalar products: "<<mu_multiplier_increment<<"\n";
  			
  			// compute the multiplier increment
  			// update/correct the multiplier
  			mu_multiplier_increment /= sum_denominator;
  			mu_multipliers[i] += mu_multiplier_increment;
  			
  			// update/correct the velocities of physical degrees of freedom
  			for(unsigned k=0; k < vel.size(); k++){
  				//std::cout<<"PLUMED :: rattle() before velocity correction :: i: "<<i<<" k: "<<k<<" vel[k]: "<<vel[k]<<"\n";
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1){ 
  					vel[k] = vel[repetition_vector[i][k]];
  					//std::cout<<"PLUMED :: rattle() velocity hacking :: i: "<<i<<" k: "<<k<<" vel[k]: "<<vel[k]<<"\n";
  					continue;
  				}
  				
  				//std::cout<<"PLUMED :: Moonie.cpp :: rattle() :: bf :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" vel[k]: "<<vel[k]<<" mu_multiplier_increment: "<<mu_multiplier_increment<<"\n";
  				
  				mass_real = actionAtomistic_vector[i]->getMass(k);
  				vel[k][0] += derivatives_old[i][k*3] * mu_multiplier_increment/mass_real;
  				vel[k][1] += derivatives_old[i][k*3 + 1] * mu_multiplier_increment/mass_real;
  				vel[k][2] += derivatives_old[i][k*3 + 2] * mu_multiplier_increment/mass_real;
  				
  				//std::cout<<"PLUMED :: rattle() after velocity correction :: i: "<<i<<" k: "<<k<<" vel[k]: "<<vel[k]<<"\n";
  				//std::cout<<"PLUMED :: Moonie.cpp :: rattle() :: af :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" vel[k]: "<<vel[k]<<" mu_multiplier_increment: "<<mu_multiplier_increment<<"\n";
  			}
  			
  			// update/correct the velocities of auxiliary degrees of freedom
  			// changes for MOONIE 12.05.2025
  			if(do_only_blue_moon == 0) vs_aux[i] -= mu_multiplier_increment/mass[i];
  			
  			actionAtomistic_vector[i]->applyVelocities();
  			actionAtomistic_vector[i]->updateVelocities();


  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: shake() :: iter after updatePositions() after :: i: "<<i<<" actionAtomistic_vector[i]->getPosition(0): "<<actionAtomistic_vector[i]->getPosition(0)<<" actionAtomistic_vector[i]->getPosition(1): "<<actionAtomistic_vector[i]->getPosition(1)<<"\n";
  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: shake() :: iter after updatePositions() after :: i: "<<i<<" pos[0]: "<<pos[0]<<" pos[1]: "<<pos[1]<<"\n";
  			
  			/*stop_iteration_condition_each_cv = 0.0;
  			
  			vs_aux[i] -= mu_multiplier_increment/mass[i];
  				
  			for(unsigned k=0; k < vel.size(); k++){
  				stop_iteration_condition_each_cv += derivatives_old[i][k*3] * vel[k][0] + derivatives_old[i][k*3 + 1] * vel[k][1] + derivatives_old[i][k*3 + 2] * vel[k][2];
  				
  				//std::cout<<"PLUMED :: Moonie.cpp :: rattle() stop_iter_check :: i: "<<i<<" k: "<<k<<" scalar product: "<<derivatives_old[i][k*3] * vel[k][0] + derivatives_old[i][k*3 + 1] * vel[k][1] + derivatives_old[i][k*3 + 2] * vel[k][2]<<" vel[k][0]: "<<vel[k][0]<<" vel[k][1]: "<<vel[k][1]<<" vel[k][2]: "<<vel[k][2]<<" derivatives_old[i][k*3]: "<<derivatives_old[i][k*3]<<" derivatives_old[i][k*3 + 1]: "<<derivatives_old[i][k*3 + 1]<<" derivatives_old[i][k*3 + 2]: "<<derivatives_old[i][k*3 + 2]<<"\n";
  			}
  			
  			stop_iteration_condition += stop_iteration_condition_each_cv*stop_iteration_condition_each_cv;
  			
  			//UNCOMMENT
  			stop_iteration_condition -= vs_aux[i];*/
  			
  			//std::cout<<"PLUMED :: Moonie.cpp :: rattle() :: nearly the end :: i: "<<i<<" s_aux[i]: "<<s_aux[i]<<" getArgument(i): "<<getArgument(i)<<" stop_iteration_condition: "<<stop_iteration_condition<<"\n";
  			
  		} 		
  		
  		//std::cout<<"PLUMED :: Moonie.cpp :: rattle() :: nearly the end :: stop_iteration_condition: "<<stop_iteration_condition<<"\n";
  	
  		//END ITERATION CONDITION
  		// compute the end-of-iteration condition
  		stop_iteration_condition = 0.0;
  		for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  			stop_iteration_condition_each_cv = 0.0;
  			//std::vector<Vector> &pos = actionAtomistic_vector[i]->modifyPositions();
  			std::vector<Vector> &vel = actionAtomistic_vector[i]->modifyVelocities();
  			for(unsigned k=0; k < vel.size(); k++){
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1) continue;
  				stop_iteration_condition_each_cv += derivatives_old[i][k*3] * vel[k][0] + derivatives_old[i][k*3 + 1] * vel[k][1] + derivatives_old[i][k*3 + 2] * vel[k][2];
  				//std::cout<<"PLUMED :: Moonie.cpp :: rattle() :: af :: i: "<<i<<" k: "<<k<<" pos[k]: "<<pos[k]<<" vel[k]: "<<vel[k]<<" s_aux[i]: "<<s_aux[i]<<" vs_aux[i]: "<<vs_aux[i]<<"\n";
  			}
  			// changes for MOONIE 12.05.2025
  			if(do_only_blue_moon == 0) stop_iteration_condition_each_cv -= vs_aux[i];
  			stop_iteration_condition += stop_iteration_condition_each_cv*stop_iteration_condition_each_cv;
  		}
  		
  		// end the iteration
  		if(stop_iteration_condition < stop_iteration_accuracy) stop_iteration = true;
  		
  	}
   
   
    
}


}

}
