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

TO DO add the whole description

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

  componentsAreNotOptional(keys);
  keys.addOutputComponent("_s","default","Output of the auxiliary variable for every reaction coordinate");
  keys.addOutputComponent("_lambda","default","Output of the SHAKE lagrange multiplier for every reaction coordinate");
  keys.addOutputComponent("_mu","default","Output of the RATTLE lagrange multiplier for every reaction coordinate");
  
  // changes for MOONIE 10.12.2024
  keys.addOutputComponent("detZ","default","Output of the determinant of Z");
  keys.addOutputComponent("detZtoMinusHalf","default","Output of the determinant of Z to the power of -0.5");
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
  
  
  
  if(temp>=0.0) kbt=plumed.getAtoms().getKBoltzmann()*temp;
  else kbt=plumed.getAtoms().getKbT();
  checkRead();

  log.printf("  with mass");
  for(unsigned i=0; i<mass.size(); i++) log.printf(" %f",mass[i]);
  log.printf("\n");

  log.printf("  with friction");
  for(unsigned i=0; i<friction.size(); i++) log.printf(" %f",friction[i]);
  log.printf("\n");


  log.printf("  and kbt");
  log.printf(" %f",kbt);
  log.printf("\n");
  
  
  // changes for MOONIE 30.08.2024
  // fill the vector with action atomistic for each collective variable and allocate the space for the vector of current_derivatives_old
  unsigned max_size_derivatives = 0;
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
   		actionAtomistic_vector.push_back(dynamic_cast<PLMD::ActionAtomistic*>(getPntrToArgument(i)->getPntrToAction())); 	
   		max_size_derivatives = std::max(max_size_derivatives, getPntrToArgument(i)->getNumberOfDerivatives());
   }
   
   current_derivative_old.reserve(max_size_derivatives);

  // prepare the quantities that are going to be outputted
  for(unsigned i=0; i<getNumberOfArguments(); i++) {
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

}

// changes for MOONIE 10.12.2024
// computation of Z matrix
void Moonie::compute_Z() {
	double element = 0.0;
	double mass_real = 1.0;
	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=i; j<getNumberOfArguments(); ++j) {
			
			// changes for MOONIE 14.05.2025 - map 
			for(auto map_element : map_vector[i][j]){
				int k = map_element.first;
				int l = map_element.second;
				mass_real = actionAtomistic_vector[i]->getMass(k);
				element += (derivatives_old[i][3*k] * derivatives_old[j][3*l] + derivatives_old[i][3*k+1] * derivatives_old[j][3*l+1] + derivatives_old[i][3*k+2] * derivatives_old[j][3*l+2])/mass_real;		
			}
			
			if(i == j){
				// computation of diagonal elements
				// changes for MOONIE 12.05.2025
				if(do_only_blue_moon == 0) element += 1/mass[i];
				Z_matrix[i][i] = element;
			} else {
				// computation of off-diagonal elements
				Z_matrix[i][j] = element;
				Z_matrix[j][i] = element;
			}
			element = 0.0;
		}
	}
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
}

// changes for MOONIE 16.05.2025
void Moonie::fold_if_requested(double* value, double period) {
	if(period != 0) *value -= std::round(*value/period) * period;
}

void Moonie::calculate() {
  
  if(firsttime) {
    // changes for MOONIE 10.09.2024
    // the initial computation of all the quantities, which takes place at the very beginning of the simulation
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    	// changes for MOONIE 12.05.2025
    	if(do_only_blue_moon == 0) s_aux[i] = getArgument(i);
    	a_thermo[i] = exp(-getTimeStep()*getStride()*friction[i]);
    	current_derivative_old.clear();
   		for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
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
  					repetition_vector_current[k] = j;
  				}
  			}
  		}
  		// We fill the repetition_vector with the corresponding repetition_vector_current for each reaction coordinate.
  		repetition_vector.push_back(repetition_vector_current);
    }
    firsttime=false;
	
	 // changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
	 // We correct the vector of dervivatives of reaction coordinates, due to the repetition of indeces. We add the contributions of all the non-first occurences of the same particle to the derivative corresponding to the first occurence.
	 for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < repetition_vector[i].size(); j++){
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
  }
  
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
  
  if(shake_rattle_switch == 2) {
  	rattle();
  	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    	s_values[i]->set(s_aux[i]);
    	mu_values[i]->set(mu_multipliers[i]* 2.0/(getTimeStep()*getStride()));
  	}
  	// changes for MOONIE 10.12.2024
  	det_Z_value->set(det_Z);
	det_Z_value_to_minus_half->set(det_Z_to_minus_half);
  }
  
  // changes for MOONIE 29.08.2024
  if(shake_rattle_switch == 2) {
  	shake_rattle_switch = -1;
  }
  shake_rattle_switch++;
}

//This function is redundant, but is here due to the fact that this function is present in all the other bias subroutines in PLUMED.
void Moonie::update() {
  
}

//This subroutine executes the "SHAKE" iteration (i.e. the iteration of position and correction of 
// the velocities at half timestep)
void Moonie::shake(){
	double dt=getTimeStep()*getStride();
	// thermostating veocities of auxilliary variables (vs_aux) and performing
	// the first 2 steps of Velocity-Verlet integration for auxilliary veriables
	// changes for MOONIE 12.05.2025
	if(do_only_blue_moon == 0){
		for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  			actionAtomistic_vector[i]->share();
  			vs_aux[i] *= sqrt(a_thermo[i]);
  			vs_aux[i] += sqrt(kbt*(1-a_thermo[i])/mass[i]) * rand_num(generator);
  			s_aux[i] += dt*vs_aux[i];
  		}	
  	}
  	
  	double mass_real = 1.0;
  	
  	// first step of SHAKE
  	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  		lambda_multipliers[i] = 0.0; //TO DO: comment out this line and take as the first approximatoion of lambda its value from the end of the previous time step
  		actionAtomistic_vector[i]->retrieveAtoms();
  		std::vector<Vector> &pos = actionAtomistic_vector[i]->modifyPositions();
  		std::vector<Vector> &vel = actionAtomistic_vector[i]->modifyVelocities();
  		
  		// the first step of SHAKE algorithm
  		for(unsigned j=0; j < pos.size(); j++){
  			// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  			if(repetition_vector[i][j] != -1){ 
  				pos[j] = pos[repetition_vector[i][j]];
  				continue;
  			}
  			
  			mass_real = actionAtomistic_vector[i]->getMass(j);
  			
  			AtomNumber index_current = actionAtomistic_vector[i]->getAbsoluteIndex(j);
  			
  			pos[j][0] += derivatives_old[i][j*3] * lambda_multipliers[i]/mass_real;
  			pos[j][1] += derivatives_old[i][j*3 + 1] * lambda_multipliers[i]/mass_real;
  			pos[j][2] += derivatives_old[i][j*3 + 2] * lambda_multipliers[i]/mass_real;
  		}
  		
  		// changes for MOONIE 12.05.2025
  		if(do_only_blue_moon == 0) s_aux[i] -= lambda_multipliers[i]/mass[i];
  		
  		actionAtomistic_vector[i]->applyPositions();
  		actionAtomistic_vector[i]->updatePositions();  		
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
  			
  			//compute new derivatives
  			std::vector<double> new_derivative;
   			for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
  				new_derivative.push_back(getPntrToArgument(i)->getDerivative(j));
  			}
  			
  			// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  			for(unsigned k=0; k < repetition_vector[i].size(); k++){
  				if(repetition_vector[i][k] != -1){
  					new_derivative[3*repetition_vector[i][k]] += new_derivative[3*k];
  					new_derivative[3*repetition_vector[i][k] + 1] += new_derivative[3*k + 1];
  					new_derivative[3*repetition_vector[i][k] + 2] += new_derivative[3*k + 2];
  				}
  			}
  			
  			sum_denominator = 0.0;
  			//ignoring the box derivatives
  			//compute the first part of the denominator of the multiplier increment
  			for(unsigned k=0; k < pos.size(); k++){
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1) continue;
  				mass_real = actionAtomistic_vector[i]->getMass(k);
  				sum_denominator += (derivatives_old[i][k*3] * new_derivative[k*3] + derivatives_old[i][k*3 +1] * new_derivative[k*3 +1] + derivatives_old[i][k*3 +2] * new_derivative[k*3 +2])/mass_real;
  			}
  			
  			//UNCOMMENT
  			//compute the second part of the denominator of the multiplier increment
  			sum_denominator += 1.0/mass[i];
  			
  			// compute the numerator of multiplier increment	
  			numerator = getArgument(i) - s_aux[i];		
  			
  			// changes for MOONIE 16.05.2025
  			fold_if_requested(&numerator, period_s_aux[i]);
  			
  			// set the lambda increment
  			// update lambda
  			lambda_multiplier_increment = -numerator/sum_denominator;
  			lambda_multipliers[i] += lambda_multiplier_increment;
  			
  			// correct/update the physical degrees of freedom, i.e. the coordinates
  			for(unsigned k=0; k < pos.size(); k++){
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1){ 
  					pos[k] = pos[repetition_vector[i][k]];
  					continue;
  				}
  				mass_real = actionAtomistic_vector[i]->getMass(k);
  				pos[k][0] += derivatives_old[i][k*3] * lambda_multiplier_increment/mass_real;
  				pos[k][1] += derivatives_old[i][k*3 + 1] * lambda_multiplier_increment/mass_real;
  				pos[k][2] += derivatives_old[i][k*3 + 2] * lambda_multiplier_increment/mass_real;
  			}
  			
  			//correct/update the auxilliary varaibles
  			// changes for MOONIE 12.05.2025
  			if(do_only_blue_moon == 0) s_aux[i] -= lambda_multiplier_increment/mass[i];
  			
  			// changes for MOONIE 16.05.2025
  			if(do_only_blue_moon == 0) fold_if_requested(&s_aux[i], period_s_aux[i]);
  			  					
  			actionAtomistic_vector[i]->applyPositions();
  			actionAtomistic_vector[i]->updatePositions();
  			
  			getPntrToArgument(i)->clearDerivatives();
  			actionAtomistic_vector[i]->calculate();
  			
  		}
  		
  		// compute the end-of-iteration condition
  		for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  			stop_iteration_condition += (getArgument(i) - s_aux[i])*(getArgument(i) - s_aux[i]);
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
  			// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  			if(repetition_vector[i][k] != -1){ 
  				vel[k] = vel[repetition_vector[i][k]];
  				continue;
  			}
  				
  			mass_real = actionAtomistic_vector[i]->getMass(k);
  			vel[k][0] += derivatives_old[i][k*3] * lambda_multipliers[i]/(mass_real * dt);
  			vel[k][1] += derivatives_old[i][k*3 + 1] * lambda_multipliers[i]/(mass_real * dt);
  			vel[k][2] += derivatives_old[i][k*3 + 2] * lambda_multipliers[i]/(mass_real * dt);
  		}
  		
  		// changes for MOONIE 12.05.2025
  		if(do_only_blue_moon == 0) vs_aux[i] -= lambda_multipliers[i]/(mass[i] * dt);
  		
  		actionAtomistic_vector[i]->applyVelocities();
  		actionAtomistic_vector[i]->updateVelocities();
  	}
  	 	
}

// function that updates derivatives old array
// changes for MOONIE 10.12.2024
void Moonie::update_derivatives_old(){
	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		actionAtomistic_vector[i]->share();
		current_derivative_old.clear();
   		for(unsigned j=0; j < getPntrToArgument(i)->getNumberOfDerivatives(); j++){
  			current_derivative_old.push_back(getPntrToArgument(i)->getDerivative(j));
  		}
  		derivatives_old[i] = current_derivative_old;
	}
	
	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
		for(unsigned j=0; j < repetition_vector[i].size(); j++){
			if(repetition_vector[i][j] != -1){
				derivatives_old[i][3*repetition_vector[i][j]] += derivatives_old[i][3*j];
				derivatives_old[i][3*repetition_vector[i][j] + 1] += derivatives_old[i][3*j + 1];
				derivatives_old[i][3*repetition_vector[i][j] + 2] += derivatives_old[i][3*j + 2];
			}
		}	
	}
}

// This subroutine executes the "RATTLE" iteration (i.e. the iteration of velocities at the end of timestep)
void Moonie::rattle(){
	// thermostat auxilliary variables
  	// changes for MOONIE 10.09.2024
    // changes for MOONIE 12.05.2025
    if(do_only_blue_moon == 0) {
    	for(unsigned i=0; i<getNumberOfArguments(); ++i) {
			vs_aux[i] *= sqrt(a_thermo[i]);
  			vs_aux[i] += sqrt(kbt*(1-a_thermo[i])/mass[i]) * rand_num(generator);
    	}
    }
    	
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
  			
  			//compute the first part of the denominator of the multiplier increment
  			sum_denominator = 0.0;
  			for(unsigned k=0; k < vel.size(); k++){
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1) continue;
  				
  				mass_real = actionAtomistic_vector[i]->getMass(k);
  				sum_denominator += (derivatives_old[i][k*3] * derivatives_old[i][k*3] + derivatives_old[i][k*3 +1] * derivatives_old[i][k*3 +1] + derivatives_old[i][k*3 +2] * derivatives_old[i][k*3 +2])/mass_real;
  			}
  			
  			// compute the second part of the denominator of the multiplier increment
  			sum_denominator += 1.0/mass[i];
  			
  			// compute the first part of numerator of the multiplier increment
  			mu_multiplier_increment = 0.0;
  			for(unsigned k=0; k < vel.size(); k++){			
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1) continue;
  				mu_multiplier_increment -= (derivatives_old[i][k*3] * vel[k][0] + derivatives_old[i][k*3 + 1] * vel[k][1] + derivatives_old[i][k*3 + 2] * vel[k][2]);
  			
  			}
  			
  			// compute the second part of numerator of the multiplier increment
  			// changes for MOONIE 12.05.2025
  			if(do_only_blue_moon == 0) mu_multiplier_increment += vs_aux[i];
  			
  			// compute the multiplier increment
  			// update/correct the multiplier
  			mu_multiplier_increment /= sum_denominator;
  			mu_multipliers[i] += mu_multiplier_increment;
  			
  			// update/correct the velocities of physical degrees of freedom
  			for(unsigned k=0; k < vel.size(); k++){
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1){ 
  					vel[k] = vel[repetition_vector[i][k]];
  					continue;
  				}
  				
  				mass_real = actionAtomistic_vector[i]->getMass(k);
  				vel[k][0] += derivatives_old[i][k*3] * mu_multiplier_increment/mass_real;
  				vel[k][1] += derivatives_old[i][k*3 + 1] * mu_multiplier_increment/mass_real;
  				vel[k][2] += derivatives_old[i][k*3 + 2] * mu_multiplier_increment/mass_real;
  			}
  			
  			// update/correct the velocities of auxiliary degrees of freedom
  			// changes for MOONIE 12.05.2025
  			if(do_only_blue_moon == 0) vs_aux[i] -= mu_multiplier_increment/mass[i];
  			
  			actionAtomistic_vector[i]->applyVelocities();
  			actionAtomistic_vector[i]->updateVelocities();
  			
  		}
  	
  		//END ITERATION CONDITION
  		// compute the end-of-iteration condition
  		stop_iteration_condition = 0.0;
  		for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  			stop_iteration_condition_each_cv = 0.0;
  			std::vector<Vector> &vel = actionAtomistic_vector[i]->modifyVelocities();
  			for(unsigned k=0; k < vel.size(); k++){
  				// changes for MOONIE 30.10.2024 - brutal hack to fix double indexing iteration
  				if(repetition_vector[i][k] != -1) continue;
  				stop_iteration_condition_each_cv += derivatives_old[i][k*3] * vel[k][0] + derivatives_old[i][k*3 + 1] * vel[k][1] + derivatives_old[i][k*3 + 2] * vel[k][2];
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

