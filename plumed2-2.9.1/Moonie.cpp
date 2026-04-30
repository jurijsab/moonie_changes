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
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "core/ActionAtomistic.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS MOONIE
/*
Add Moonie.

This action can be used to create fictitious collective variables coupled to the real ones.
The coupling is obtained by the rigorous imposition of holonomic constraints between the fictitious collective
variables and the value of the collective variable, which depends on the real coordinates.
The imposition of the holonomic constraint is achieved by SHAKE and RATTLE constraining algorithms.

The main goal of this action is to output the quantities to compute the associated free-energy profile.
These quantities are Lagrangian multipliers (lambda and mu) for each collective variable. We obtain
lambda from the SHAKE iteration procedure, while the mu multiplier is computed through the RATTLE
iteration. In addition this module outputs the value of the det(Z)^(-0.5) (see later for Z). The free-energy
profile can later be recovered through post-processing.

In addition, this class also performs the time propagation of the fictitious collective variables
and their corresponding velocities through the OVRVO algorithm, described in several sources (e.g.
D. A. Sivak, J. D. Chodera, and G. E. Crooks, The Journal of Physical Chemistry B, vol. 118, no. 24, pp. 6466–6474, 2014).
Furthermore, the fictitious collective variables can be thermostated at their own temperature (which should be higher than
the physical one), similarly to the TAMD algorithm, published in:
L. Maragliano and E. Vanden-Eijnden, Chemical Physics Letters, vol. 426, no. 1, pp. 168–175, 2006.

The whole procedure is a dynamical extension of the Blue Moon formalism.

This action takes three compulsory parameters: "MASS", "TEMPERATURE" and "FRICTION".
"MASS" is a vectorial keyword that takes the masses of all fictitious collective variables.
"TEMPERATURE" is a scalar keyword, taking the fictitious temperature, at which the fictitious collective variables are thermostated.
"FRICTION" is a vectorial keyword that takes the frictions of all fictitious collective variables for their thermostating.

In addition, this action (bias) takes three optional parameters: "DO_ONLY_BLUE_MOON", "VALUES_COLVARS" and "PERIOD_COLVARS".
"DO_ONLY_BLUE_MOON" is a boolean keyword that introduces the option of running the original Blue Moon algorithm only. If DO_ONLY_BLUE_MOON=1 the time propagation
and thermostating of the fictitious collective variables are disabled. Instead the user must define the constant collective variables values, to which the
system has to relax. To this end we use "VALUES_COLVARS", a vectorial keyword, which takes the user-defined values of all collective variables.
"PERIOD_COLVARS" is a vectorial keyword that takes the period of the collective variables, in case they are periodic. If this keyword is 0 for
any collective variable, the program will treat this collective variable as non-periodic.
*/
//+ENDPLUMEDOC

class Moonie : public Bias {
private:
  // =====================================================================
  // Small internal helper types.
  // =====================================================================

  // This structure replaces the old repetition-vector hack with a canonical
  // per-CV representation built on *unique physical atoms*.
  //
  // The ActionAtomistic interface may expose a local atom list in which the
  // same physical atom appears more than once. That is convenient for the CV
  // implementation, but it is awkward for Moonie because all constraint
  // mathematics should only see each physical atom once.
  //
  // For each CV we therefore precompute a compact representation:
  //   - local_to_unique[local] tells which unique atom a local entry belongs to
  //   - unique_to_local[unique] stores one representative local entry
  //   - unique_absolute_indices[unique] stores the physical atom id
  //
  // Derivatives are compressed onto this unique representation before they are
  // used in SHAKE, RATTLE, and Z-matrix algebra.
  struct CvTopology {
    std::vector<unsigned> local_to_unique;
    std::vector<unsigned> unique_to_local;
    std::vector<int> unique_absolute_indices;
  };

  // For each pair of CVs we store the overlap between their unique physical
  // atom lists. Each entry contains (unique-index-in-i, unique-index-in-j).
  // This is used to compute Z efficiently and cleanly.
  using PairOverlap = std::vector< std::pair<unsigned, unsigned> >;

  // =====================================================================
  // Constants controlling numerical behavior.
  // =====================================================================
  static constexpr double kStopIterationAccuracy = 1.0e-5;
  static constexpr int    kMaxShakeIterations    = 30000;
  static constexpr int    kMaxRattleIterations   = 30000;
  static constexpr int    kMaxSupportedColvars   = 4;

  // =====================================================================
  // User input and user-facing state.
  // =====================================================================
  bool firsttime_;
  bool restart_state_loaded_;
  bool have_values_colvars_;

  std::vector<double> s_aux_;
  std::vector<double> vs_aux_;
  std::vector<double> lambda_multipliers_;
  std::vector<double> mu_multipliers_;

  std::vector<double> mass_;
  std::vector<double> friction_;
  std::vector<double> a_thermo_;
  std::vector<double> period_s_aux_;

  double kbt_;
  int shake_rattle_switch_;
  int do_only_blue_moon_;

  std::vector< std::vector<double> > Z_matrix_;
  double det_Z_;
  double det_Z_to_minus_half_;

  // =====================================================================
  // Internal structural state.
  // =====================================================================
  std::vector<PLMD::ActionAtomistic*> actionAtomistic_vector_;
  std::vector<CvTopology> cv_topology_;
  std::vector<PairOverlap> flat_overlap_map_; // flattened [i*ncv + j]

  // derivatives_old_unique_[i] stores the derivative of CV i compressed onto
  // the unique-atom representation: 3 entries per unique atom.
  std::vector< std::vector<double> > derivatives_old_unique_;

  // =====================================================================
  // RNG state.
  // =====================================================================
  std::normal_distribution<double> gaussian_;
  std::mt19937_64 generator_;

  // =====================================================================
  // Output handles.
  // =====================================================================
  std::vector<Value*> s_values_;
  std::vector<Value*> lambda_values_;
  std::vector<Value*> mu_values_;
  Value* det_Z_value_;
  Value* det_Z_value_to_minus_half_;

  // =====================================================================
  // Restart / debug state.
  // =====================================================================
  std::string restart_state_filename_;

  // Debugging aid: after a successful Moonie restart, print detailed state
  // information for the first restarted Moonie MD step (up to three staged
  // calls: SHAKE / middle / RATTLE).
  bool debug_restart_step_pending_;
  int  debug_restart_calls_remaining_;
  
  // Saved old derivatives from the restart sidecar, used for restart diagnostics.
  bool saved_derivatives_loaded_;
  std::vector< std::vector<double> > saved_derivatives_old_unique_;  
  bool skip_first_restart_shake_drift_;
  
public:
  explicit Moonie(const ActionOptions&);
  void calculate() override;
  void update() override;
  static void registerKeywords(Keywords& keys);

private:
  // -------------------- Input / validation / setup ----------------------
  void parseAndValidateInput_();
  void validateActionArguments_();
  void buildTopologies_();
  void buildOverlapMap_();
  void initializeRuntimeState_();

  // -------------------- Restart helpers --------------------------------
  std::string defaultRestartStateFilename_() const;
  void saveRestartState_() const;
  bool loadRestartState_();

  // -------------------- Utility helpers --------------------------------
  PairOverlap& overlap_(unsigned i, unsigned j) {
    return flat_overlap_map_[i*getNumberOfArguments() + j];
  }
  const PairOverlap& overlap_(unsigned i, unsigned j) const {
    return flat_overlap_map_[i*getNumberOfArguments() + j];
  }
  
  double computeVelocityProjection_(unsigned cv_index) const;

  std::vector<double> readCompressedDerivatives_(unsigned cv_index);
  void syncDuplicatePositions_(unsigned cv_index, std::vector<Vector>& pos) const;
  void syncDuplicateVelocities_(unsigned cv_index, std::vector<Vector>& vel) const;

  double constraintMismatch_(unsigned cv_index, double argument_value, double target_value) const;
  void fold_if_requested_(double* value, double period) const;
  double effectiveAuxiliaryInverseMass_(unsigned cv_index) const;
  std::string formattedVector_(const std::vector<double>& v) const;

  // -------------------- Matrix helpers ---------------------------------
  void updateDerivativesOld_();
  void computeZ_();
  void computeDetZ_();
  double determinantWithPivoting_(const std::vector< std::vector<double> >& matrix) const;
  void debugPrintStageState_(const char* where) const;

  
  // -------------------- Dynamics ---------------------------------------
  void shake_();
  void rattle_();
};

PLUMED_REGISTER_ACTION(Moonie,"MOONIE")

void Moonie::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.use("RESTART");
  keys.add("compulsory","MASS","specifies the masses of all auxiliary variables");
  keys.add("compulsory","TEMPERATURE","specifies the temperature of the auxiliary variables");
  keys.add("compulsory","FRICTION","add a friction of the auxiliary variables");
  keys.add("optional","DO_ONLY_BLUE_MOON","disable moonie and do only blue moon");
  keys.add("optional","VALUES_COLVARS","fixed values of collective variables for do only blue moon");
  keys.add("optional","PERIOD_COLVARS","period of each collective variable");

  componentsAreNotOptional(keys);
  keys.addOutputComponent("_s","default","Output of the auxiliary variable for every reaction coordinate");
  keys.addOutputComponent("_lambda","default","Output of the SHAKE lagrange multiplier for every reaction coordinate");
  keys.addOutputComponent("_mu","default","Output of the RATTLE lagrange multiplier for every reaction coordinate");
  keys.addOutputComponent("detZ","default","Output of the determinant of Z");
  keys.addOutputComponent("detZtoMinusHalf","default","Output of the determinant of Z to the power of -0.5");
}

Moonie::Moonie(const ActionOptions& ao)
  : PLUMED_BIAS_INIT(ao)
  , firsttime_(true)
  , restart_state_loaded_(false)
  , have_values_colvars_(false)
  , s_aux_(getNumberOfArguments(), 0.0)
  , vs_aux_(getNumberOfArguments(), 0.0)
  , lambda_multipliers_(getNumberOfArguments(), 0.0)
  , mu_multipliers_(getNumberOfArguments(), 0.0)
  , mass_(getNumberOfArguments(), 0.0)
  , friction_(getNumberOfArguments(), 0.0)
  , a_thermo_(getNumberOfArguments(), 0.0)
  , period_s_aux_(getNumberOfArguments(), 0.0)
  , kbt_(0.0)
  , shake_rattle_switch_(0)
  , do_only_blue_moon_(0)
  , Z_matrix_(getNumberOfArguments(), std::vector<double>(getNumberOfArguments(), 0.0))
  , det_Z_(0.0)
  , det_Z_to_minus_half_(0.0)
  , gaussian_(0.0, 1.0)
  , generator_(5489ULL) // explicit fixed seed; state will be restored on restart
  , s_values_(getNumberOfArguments(), nullptr)
  , lambda_values_(getNumberOfArguments(), nullptr)
  , mu_values_(getNumberOfArguments(), nullptr)
  , det_Z_value_(nullptr)
  , det_Z_value_to_minus_half_(nullptr)
  , restart_state_filename_(defaultRestartStateFilename_())
  , debug_restart_step_pending_(false)
  , debug_restart_calls_remaining_(0)   
  , saved_derivatives_loaded_(false)
  , saved_derivatives_old_unique_()  
  , skip_first_restart_shake_drift_(false)
   {
  parseAndValidateInput_();
  validateActionArguments_();

  // Create all output components.
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    std::string comp = getPntrToArgument(i)->getName()+"_s";
    addComponent(comp);
    componentIsNotPeriodic(comp);
    s_values_[i] = getPntrToComponent(comp);

    comp = getPntrToArgument(i)->getName()+"_lambda";
    addComponent(comp);
    componentIsNotPeriodic(comp);
    lambda_values_[i] = getPntrToComponent(comp);

    comp = getPntrToArgument(i)->getName()+"_mu";
    addComponent(comp);
    componentIsNotPeriodic(comp);
    mu_values_[i] = getPntrToComponent(comp);
  }

  addComponent("detZ");
  componentIsNotPeriodic("detZ");
  det_Z_value_ = getPntrToComponent("detZ");

  addComponent("detZtoMinusHalf");
  componentIsNotPeriodic("detZtoMinusHalf");
  det_Z_value_to_minus_half_ = getPntrToComponent("detZtoMinusHalf");
}


double Moonie::computeVelocityProjection_(unsigned cv_index) const {
  actionAtomistic_vector_[cv_index]->retrieveAtoms();
  std::vector<Vector>& vel = actionAtomistic_vector_[cv_index]->modifyVelocities();
  const CvTopology& topo = cv_topology_[cv_index];

  double proj = 0.0;
  for(unsigned u=0; u<topo.unique_to_local.size(); ++u) {
    const unsigned local = topo.unique_to_local[u];
    const unsigned idx = 3*u;
    proj +=
      derivatives_old_unique_[cv_index][idx    ] * vel[local][0] +
      derivatives_old_unique_[cv_index][idx + 1] * vel[local][1] +
      derivatives_old_unique_[cv_index][idx + 2] * vel[local][2];
  }
  return proj;
}

void Moonie::parseAndValidateInput_() {
  // Use temporary vectors for optional input so that we can distinguish
  // "not provided" from "provided but malformed" cleanly.
  parseVector("MASS", mass_);
  parseVector("FRICTION", friction_);

  double temp=-1.0;
  parse("TEMPERATURE", temp);
  parse("DO_ONLY_BLUE_MOON", do_only_blue_moon_);

  if(do_only_blue_moon_!=0 && do_only_blue_moon_!=1) {
    error("DO_ONLY_BLUE_MOON must be either 0 or 1");
  }

  std::vector<double> values_colvars;
  parseVector("VALUES_COLVARS", values_colvars);
  have_values_colvars_ = !values_colvars.empty();
  if(have_values_colvars_) {
    s_aux_ = values_colvars;
  }

  std::vector<double> period_colvars;
  parseVector("PERIOD_COLVARS", period_colvars);
  if(!period_colvars.empty()) {
    period_s_aux_ = period_colvars;
  }

  if(mass_.size()!=getNumberOfArguments()) {
    error("MASS must contain exactly one entry per ARG");
  }
  if(friction_.size()!=getNumberOfArguments()) {
    error("FRICTION must contain exactly one entry per ARG");
  }
  if(have_values_colvars_ && s_aux_.size()!=getNumberOfArguments()) {
    error("VALUES_COLVARS must contain exactly one entry per ARG");
  }
  if(!period_colvars.empty() && period_s_aux_.size()!=getNumberOfArguments()) {
    error("PERIOD_COLVARS must contain exactly one entry per ARG");
  }

  if(do_only_blue_moon_==1 && !have_values_colvars_) {
    error("VALUES_COLVARS is mandatory when DO_ONLY_BLUE_MOON=1");
  }

  for(unsigned i=0; i<mass_.size(); ++i) {
    if(!(mass_[i] > 0.0)) {
      error("All entries in MASS must be strictly positive");
    }
  }
  for(unsigned i=0; i<friction_.size(); ++i) {
    if(friction_[i] < 0.0) {
      error("All entries in FRICTION must be non-negative");
    }
  }
  for(unsigned i=0; i<period_s_aux_.size(); ++i) {
    if(period_s_aux_[i] < 0.0) {
      error("All entries in PERIOD_COLVARS must be either 0 (non-periodic) or positive");
    }
  }

  if(temp>=0.0) {
    kbt_ = plumed.getAtoms().getKBoltzmann()*temp;
  } else {
    kbt_ = plumed.getAtoms().getKbT();
  }

  checkRead();
/*
  log.printf("  with mass %s\n", formattedVector_(mass_).c_str());
  log.printf("  with friction %s\n", formattedVector_(friction_).c_str());
  log.printf("  with kbt %.16g\n", kbt_);
  log.printf("  DO_ONLY_BLUE_MOON = %d\n", do_only_blue_moon_);
  log.printf("  PERIOD_COLVARS = %s\n", formattedVector_(period_s_aux_).c_str());
  if(do_only_blue_moon_==1) {
    log.printf("  VALUES_COLVARS = %s\n", formattedVector_(s_aux_).c_str());
  }
  log.printf("  Moonie restart sidecar file = %s\n", restart_state_filename_.c_str());
*/  
}

void Moonie::validateActionArguments_() {
  actionAtomistic_vector_.clear();
  actionAtomistic_vector_.reserve(getNumberOfArguments());

  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    auto* aa = dynamic_cast<PLMD::ActionAtomistic*>(getPntrToArgument(i)->getPntrToAction());
    if(!aa) {
      error("MOONIE requires every ARG to come from an atomistic action");
    }
    actionAtomistic_vector_.push_back(aa);
  }
}

std::string Moonie::defaultRestartStateFilename_() const {
  std::ostringstream oss;
  oss << getLabel() << ".moonie.state";
  return oss.str();
}

std::string Moonie::formattedVector_(const std::vector<double>& v) const {
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(8);
  oss << "[";
  for(unsigned i=0; i<v.size(); ++i) {
    if(i>0) oss << ", ";
    oss << v[i];
  }
  oss << "]";
  return oss.str();
}

void Moonie::buildTopologies_() {
  cv_topology_.clear();
  cv_topology_.reserve(getNumberOfArguments());

  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    CvTopology topo;
    std::map<int, unsigned> absolute_to_unique;

    std::vector<Vector>& pos = actionAtomistic_vector_[i]->modifyPositions();
    topo.local_to_unique.resize(pos.size(), 0);

    for(unsigned local=0; local<pos.size(); ++local) {
      const int absolute = actionAtomistic_vector_[i]->getAbsoluteIndex(local).index();
      auto it = absolute_to_unique.find(absolute);
      if(it==absolute_to_unique.end()) {
        const unsigned unique_index = static_cast<unsigned>(topo.unique_absolute_indices.size());
        absolute_to_unique[absolute] = unique_index;
        topo.local_to_unique[local] = unique_index;
        topo.unique_to_local.push_back(local);
        topo.unique_absolute_indices.push_back(absolute);
      } else {
        topo.local_to_unique[local] = it->second;
      }
    }

    cv_topology_.push_back(topo);
  }
}

void Moonie::buildOverlapMap_() {
  const unsigned ncv = getNumberOfArguments();
  flat_overlap_map_.assign(ncv*ncv, PairOverlap());

  for(unsigned i=0; i<ncv; ++i) {
    std::map<int, unsigned> inverse_j;
    for(unsigned j=0; j<ncv; ++j) {
      inverse_j.clear();
      for(unsigned uj=0; uj<cv_topology_[j].unique_absolute_indices.size(); ++uj) {
        inverse_j[cv_topology_[j].unique_absolute_indices[uj]] = uj;
      }

      PairOverlap ov;
      for(unsigned ui=0; ui<cv_topology_[i].unique_absolute_indices.size(); ++ui) {
        const int abs_i = cv_topology_[i].unique_absolute_indices[ui];
        auto it = inverse_j.find(abs_i);
        if(it!=inverse_j.end()) {
          ov.push_back(std::make_pair(ui, it->second));
        }
      }
      overlap_(i,j) = ov;
    }
  }
}

std::vector<double> Moonie::readCompressedDerivatives_(unsigned cv_index) {
  const CvTopology& topo = cv_topology_[cv_index];
  std::vector<double> compressed(3*topo.unique_absolute_indices.size(), 0.0);

  const unsigned nder = getPntrToArgument(cv_index)->getNumberOfDerivatives();
  for(unsigned local=0; local<topo.local_to_unique.size(); ++local) {
    const unsigned u = topo.local_to_unique[local];
    const unsigned raw = 3*local;
    const unsigned out = 3*u;
    if(raw+2 >= nder) {
      error("Unexpected derivative size while compressing repeated atom entries");
    }
    compressed[out    ] += getPntrToArgument(cv_index)->getDerivative(raw    );
    compressed[out + 1] += getPntrToArgument(cv_index)->getDerivative(raw + 1);
    compressed[out + 2] += getPntrToArgument(cv_index)->getDerivative(raw + 2);
  }
  return compressed;
}

void Moonie::syncDuplicatePositions_(unsigned cv_index, std::vector<Vector>& pos) const {
  const CvTopology& topo = cv_topology_[cv_index];
  for(unsigned local=0; local<topo.local_to_unique.size(); ++local) {
    const unsigned rep = topo.unique_to_local[topo.local_to_unique[local]];
    pos[local] = pos[rep];
  }
}

void Moonie::syncDuplicateVelocities_(unsigned cv_index, std::vector<Vector>& vel) const {
  const CvTopology& topo = cv_topology_[cv_index];
  for(unsigned local=0; local<topo.local_to_unique.size(); ++local) {
    const unsigned rep = topo.unique_to_local[topo.local_to_unique[local]];
    vel[local] = vel[rep];
  }
}

void Moonie::fold_if_requested_(double* value, double period) const {
  if(period!=0.0) {
    *value -= std::round(*value/period) * period;
  }
}

double Moonie::constraintMismatch_(unsigned cv_index, double argument_value, double target_value) const {
  double diff = argument_value - target_value;
  fold_if_requested_(&diff, period_s_aux_[cv_index]);
  return diff;
}

double Moonie::effectiveAuxiliaryInverseMass_(unsigned cv_index) const {
  return (do_only_blue_moon_==0) ? (1.0/mass_[cv_index]) : 0.0;
}


bool Moonie::loadRestartState_() {
  std::ifstream in(restart_state_filename_.c_str());
  if(!in) {
    error("Restart requested, but Moonie sidecar state file was not found");
  }

  std::string magic;
  in >> magic;
  if(!in || (magic!="MOONIE_STATE_V1" && magic!="MOONIE_STATE_V2")) {
    error("Failed to parse Moonie restart sidecar state file");
  }

  unsigned ncv = 0;
  in >> ncv;
  if(!in || ncv!=getNumberOfArguments()) {
    error("Moonie restart state file is inconsistent with the current number of ARG values");
  }

  for(unsigned i=0; i<ncv; ++i) in >> s_aux_[i];
  for(unsigned i=0; i<ncv; ++i) in >> vs_aux_[i];
  for(unsigned i=0; i<ncv; ++i) in >> lambda_multipliers_[i];
  for(unsigned i=0; i<ncv; ++i) in >> mu_multipliers_[i];
  in >> shake_rattle_switch_;

  saved_derivatives_loaded_ = false;
  saved_derivatives_old_unique_.clear();

  if(magic=="MOONIE_STATE_V2") {
    /*
    unsigned nder_sets = 0;
    in >> nder_sets;
    if(!in || nder_sets!=getNumberOfArguments()) {
      error("Moonie restart sidecar has inconsistent derivative block size");
    }


    saved_derivatives_old_unique_.resize(nder_sets);
    for(unsigned i=0; i<nder_sets; ++i) {
      unsigned block_size = 0;
      in >> block_size;
      if(!in) {
        error("Moonie restart sidecar derivative block could not be read");
      }
      saved_derivatives_old_unique_[i].resize(block_size);
      for(unsigned j=0; j<block_size; ++j) {
        in >> saved_derivatives_old_unique_[i][j];
      }
    }
    saved_derivatives_loaded_ = true;
    */
  }

  in >> generator_;

  if(!in) {
    error("Moonie restart state file exists but could not be read completely");
  }

  if(shake_rattle_switch_ < -1 || shake_rattle_switch_ > 2) {
    error("Moonie restart sidecar contains an invalid shake_rattle_switch_ value");
  }

  restart_state_loaded_ = true;
  debug_restart_step_pending_ = false;
  debug_restart_calls_remaining_ = 3;
  skip_first_restart_shake_drift_ = true;
/*
  log.printf("  Moonie internal restart state restored from %s\n", restart_state_filename_.c_str());
  log.printf("  restored s_aux = %s\n", formattedVector_(s_aux_).c_str());
  log.printf("  restored vs_aux = %s\n", formattedVector_(vs_aux_).c_str());
  log.printf("  restored lambda = %s\n", formattedVector_(lambda_multipliers_).c_str());
  log.printf("  restored mu = %s\n", formattedVector_(mu_multipliers_).c_str());
  log.printf("  restored shake_rattle_switch_ = %d\n", shake_rattle_switch_);
  log.printf("  restored derivatives_old_unique_ present = %d\n", saved_derivatives_loaded_ ? 1 : 0);
*/

  return true;
}


void Moonie::saveRestartState_() const {
  std::ofstream out(restart_state_filename_.c_str(), std::ios::trunc);
  if(!out) {
    error("Moonie failed to open its restart sidecar state file for writing");
  }
/*
  log.printf("  [MOONIE DEBUG] saving sidecar state to %s\n", restart_state_filename_.c_str());
  log.printf("  [MOONIE DEBUG] saved s_aux = %s\n", formattedVector_(s_aux_).c_str());
  log.printf("  [MOONIE DEBUG] saved vs_aux = %s\n", formattedVector_(vs_aux_).c_str());
  log.printf("  [MOONIE DEBUG] saved lambda = %s\n", formattedVector_(lambda_multipliers_).c_str());
  log.printf("  [MOONIE DEBUG] saved mu = %s\n", formattedVector_(mu_multipliers_).c_str());
  log.printf("  [MOONIE DEBUG] saved shake_rattle_switch_ = %d\n", shake_rattle_switch_);
  
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    log.printf("  [MOONIE DEBUG] saved grad_dot_vel cv=%u = %.16g\n",
               i,
               computeVelocityProjection_(i));
  }
*/

  out << std::setprecision(17);
  out << "MOONIE_STATE_V2\n";
  out << getNumberOfArguments() << "\n";

  for(unsigned i=0; i<getNumberOfArguments(); ++i) out << s_aux_[i] << (i+1<getNumberOfArguments() ? ' ' : '\n');
  for(unsigned i=0; i<getNumberOfArguments(); ++i) out << vs_aux_[i] << (i+1<getNumberOfArguments() ? ' ' : '\n');
  for(unsigned i=0; i<getNumberOfArguments(); ++i) out << lambda_multipliers_[i] << (i+1<getNumberOfArguments() ? ' ' : '\n');
  for(unsigned i=0; i<getNumberOfArguments(); ++i) out << mu_multipliers_[i] << (i+1<getNumberOfArguments() ? ' ' : '\n');

  out << shake_rattle_switch_ << "\n";

/*
  // Save derivatives_old_unique_ dimensions and values.
  out << derivatives_old_unique_.size() << "\n";
  for(unsigned i=0; i<derivatives_old_unique_.size(); ++i) {
    out << derivatives_old_unique_[i].size() << "\n";
    for(unsigned j=0; j<derivatives_old_unique_[i].size(); ++j) {
      out << derivatives_old_unique_[i][j] << (j+1<derivatives_old_unique_[i].size() ? ' ' : '\n');
    }
  }
*/

  out << generator_ << "\n";

  if(!out) {
    error("Moonie failed while writing its restart sidecar state file");
  }
  
  for(unsigned i=0; i<derivatives_old_unique_.size(); ++i) 
  {
    double norm2 = 0.0;
    for(unsigned j=0; j<derivatives_old_unique_[i].size(); ++j) {
      norm2 += derivatives_old_unique_[i][j] * derivatives_old_unique_[i][j];
    }
    /*log.printf("  [MOONIE DEBUG] saved derivative norm cv=%u = %.16g\n",
               i,
               std::sqrt(norm2));
    */
  }
  
}



void Moonie::initializeRuntimeState_() {
  buildTopologies_();
  buildOverlapMap_();
  derivatives_old_unique_.resize(getNumberOfArguments());

  {
    std::ifstream test_in(restart_state_filename_.c_str());
    const bool sidecar_exists = test_in.good();

    /*log.printf("  [MOONIE DEBUG] initializeRuntimeState_: getRestart()=%d firsttime_=%d sidecar_exists=%d file=%s\n",
               getRestart() ? 1 : 0,
               firsttime_ ? 1 : 0,
               sidecar_exists ? 1 : 0,
               restart_state_filename_.c_str());
    */
    if(getRestart() && sidecar_exists) {
      const bool ok = loadRestartState_();
      //log.printf("  [MOONIE DEBUG] loadRestartState_() returned %d\n", ok ? 1 : 0);
      if(!ok) {
        error("Moonie sidecar state file exists, but Moonie failed to restore it");
      }
    } else if(getRestart() && !sidecar_exists) {
      //log.printf("  [MOONIE DEBUG] MD restart detected, but no Moonie sidecar exists; treating this as the first Moonie segment\n");
    } else {
      // Fresh Moonie start: enforce the normal initial stage.
      shake_rattle_switch_ = 0;
    }
  }

  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    if(do_only_blue_moon_==0 && !restart_state_loaded_) {
      s_aux_[i] = getArgument(i);
    }
    a_thermo_[i] = std::exp(-getTimeStep()*getStride()*friction_[i]);

    // Freshly rebuild the old derivatives from the restarted atomistic state.
    derivatives_old_unique_[i] = readCompressedDerivatives_(i);

    // If a saved derivative state is available, compare it to the freshly rebuilt one.
    if(saved_derivatives_loaded_) {
      if(saved_derivatives_old_unique_.size() != getNumberOfArguments()) {
        error("Saved derivative restart state has inconsistent outer size");
      }
      if(saved_derivatives_old_unique_[i].size() != derivatives_old_unique_[i].size()) {
        error("Saved derivative restart state has inconsistent inner size");
      }

      double max_abs_diff = 0.0;
      double rms = 0.0;
      for(unsigned j=0; j<derivatives_old_unique_[i].size(); ++j) {
        const double diff = derivatives_old_unique_[i][j] - saved_derivatives_old_unique_[i][j];
        max_abs_diff = std::max(max_abs_diff, std::fabs(diff));
        rms += diff*diff;
      }
      if(!derivatives_old_unique_[i].empty()) {
        rms = std::sqrt(rms / static_cast<double>(derivatives_old_unique_[i].size()));
      }

      log.printf("  [MOONIE DEBUG] derivative compare cv=%u max_abs_diff=%.16g rms_diff=%.16g size=%u\n",
                 i,
                 max_abs_diff,
                 rms,
                 static_cast<unsigned>(derivatives_old_unique_[i].size()));
    }
  }

  firsttime_ = false;
  /*
  log.printf("  [MOONIE DEBUG] initializeRuntimeState_ done: restart_state_loaded_=%d shake_rattle_switch_=%d\n",
             restart_state_loaded_ ? 1 : 0,
             shake_rattle_switch_);
  log.printf("  [MOONIE DEBUG] post-init s_aux = %s\n", formattedVector_(s_aux_).c_str());
  log.printf("  [MOONIE DEBUG] post-init vs_aux = %s\n", formattedVector_(vs_aux_).c_str());
  log.printf("  [MOONIE DEBUG] post-init lambda = %s\n", formattedVector_(lambda_multipliers_).c_str());
  log.printf("  [MOONIE DEBUG] post-init mu = %s\n", formattedVector_(mu_multipliers_).c_str());
  */
}


void Moonie::updateDerivativesOld_() {
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    actionAtomistic_vector_[i]->share();
    derivatives_old_unique_[i] = readCompressedDerivatives_(i);
  }
}

void Moonie::computeZ_() {
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    for(unsigned j=0; j<getNumberOfArguments(); ++j) {
      Z_matrix_[i][j] = 0.0;
    }
  }

  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    for(unsigned j=i; j<getNumberOfArguments(); ++j) {
      double element = 0.0;
      const PairOverlap& ov = overlap_(i,j);
      for(unsigned p=0; p<ov.size(); ++p) {
        const unsigned ui = ov[p].first;
        const unsigned uj = ov[p].second;
        const unsigned local_rep_i = cv_topology_[i].unique_to_local[ui];
        const double mreal = actionAtomistic_vector_[i]->getMass(local_rep_i);
        const unsigned ii = 3*ui;
        const unsigned jj = 3*uj;
        element += (
          derivatives_old_unique_[i][ii    ] * derivatives_old_unique_[j][jj    ] +
          derivatives_old_unique_[i][ii + 1] * derivatives_old_unique_[j][jj + 1] +
          derivatives_old_unique_[i][ii + 2] * derivatives_old_unique_[j][jj + 2]
        ) / mreal;
      }
      if(i==j) {
        element += effectiveAuxiliaryInverseMass_(i);
      }
      Z_matrix_[i][j] = element;
      Z_matrix_[j][i] = element;
    }
  }
}

double Moonie::determinantWithPivoting_(const std::vector< std::vector<double> >& matrix) const {
  const unsigned n = matrix.size();
  std::vector< std::vector<double> > a = matrix;
  double det = 1.0;
  int sign = 1;

  for(unsigned k=0; k<n; ++k) {
    unsigned pivot = k;
    double maxabs = std::fabs(a[k][k]);
    for(unsigned i=k+1; i<n; ++i) {
      const double candidate = std::fabs(a[i][k]);
      if(candidate>maxabs) {
        maxabs = candidate;
        pivot = i;
      }
    }

    if(maxabs<=std::numeric_limits<double>::epsilon()) {
      return 0.0;
    }

    if(pivot!=k) {
      std::swap(a[pivot], a[k]);
      sign *= -1;
    }

    const double akk = a[k][k];
    det *= akk;
    for(unsigned i=k+1; i<n; ++i) {
      const double factor = a[i][k] / akk;
      a[i][k] = 0.0;
      for(unsigned j=k+1; j<n; ++j) {
        a[i][j] -= factor * a[k][j];
      }
    }
  }

  return sign * det;
}

void Moonie::computeDetZ_() {
  const unsigned ncv = getNumberOfArguments();
  if(ncv>static_cast<unsigned>(kMaxSupportedColvars)) {
    error("Moonie currently supports at most 4 collective variables");
  }

  det_Z_ = determinantWithPivoting_(Z_matrix_);
  if(!(det_Z_ > 0.0)) {
    error("det(Z) is non-positive; the constraint metric became singular or numerically invalid");
  }
  det_Z_to_minus_half_ = 1.0/std::sqrt(det_Z_);
}

void Moonie::shake_() {
  const double dt = getTimeStep()*getStride();

  // OVRVO/Langevin half-step for the auxiliary variables. This is skipped in
  // pure Blue Moon mode, where the auxiliary variables are treated as fixed
  // targets rather than dynamical degrees of freedom (equivalently: infinite
  // auxiliary mass).
  // Also in case of restart we have to skip it
  if(do_only_blue_moon_==0) 
  {
    if(skip_first_restart_shake_drift_) 
    {
      //log.printf("  [MOONIE DEBUG] skipping first restarted SHAKE drift/noise step\n");      
    } 
    else 
    {
      for(unsigned i=0; i<getNumberOfArguments(); ++i) {
        actionAtomistic_vector_[i]->share();
        vs_aux_[i] *= std::sqrt(a_thermo_[i]);
        vs_aux_[i] += std::sqrt(kbt_*(1.0-a_thermo_[i])/mass_[i]) * gaussian_(generator_);
        s_aux_[i] += dt*vs_aux_[i];
        fold_if_requested_(&s_aux_[i], period_s_aux_[i]);
      }
    }
  }
  
  // First SHAKE prediction. We use the stored lambda from the previous step as
  // the initial guess, which is usually a better predictor than zero and also
  // gives smoother continuation after restart.
  for(unsigned i=0; i<getNumberOfArguments(); ++i) 
  {
    actionAtomistic_vector_[i]->retrieveAtoms();
    std::vector<Vector>& pos = actionAtomistic_vector_[i]->modifyPositions();

    const CvTopology& topo = cv_topology_[i];
    for(unsigned u=0; u<topo.unique_to_local.size(); ++u) 
    {
      const unsigned local = topo.unique_to_local[u];
      const double mreal = actionAtomistic_vector_[i]->getMass(local);
      const unsigned idx = 3*u;
      pos[local][0] += derivatives_old_unique_[i][idx    ] * lambda_multipliers_[i] / mreal;
      pos[local][1] += derivatives_old_unique_[i][idx + 1] * lambda_multipliers_[i] / mreal;
      pos[local][2] += derivatives_old_unique_[i][idx + 2] * lambda_multipliers_[i] / mreal;
    }
    syncDuplicatePositions_(i, pos);

    if(do_only_blue_moon_==0) 
    {
      s_aux_[i] -= lambda_multipliers_[i] / mass_[i];
      fold_if_requested_(&s_aux_[i], period_s_aux_[i]);
    }

    actionAtomistic_vector_[i]->applyPositions();
    actionAtomistic_vector_[i]->updatePositions();
    getPntrToArgument(i)->clearDerivatives();
    actionAtomistic_vector_[i]->calculate();
  }

  bool converged = false;
  int iteration_number = 0;
  double stop_iteration_condition = 0.0;

  while(!converged) 
  {
    ++iteration_number;
    if(iteration_number > kMaxShakeIterations) {
      std::ostringstream oss;
      oss << "SHAKE did not converge after " << kMaxShakeIterations
          << " iterations; final residual norm = " << stop_iteration_condition;
      error(oss.str());
    }

    stop_iteration_condition = 0.0;

    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      actionAtomistic_vector_[i]->retrieveAtoms();
      std::vector<Vector>& pos = actionAtomistic_vector_[i]->modifyPositions();

      const std::vector<double> new_derivative_unique = readCompressedDerivatives_(i);
      const CvTopology& topo = cv_topology_[i];

      double denominator = 0.0;
      for(unsigned u=0; u<topo.unique_to_local.size(); ++u) {
        const unsigned local = topo.unique_to_local[u];
        const double mreal = actionAtomistic_vector_[i]->getMass(local);
        const unsigned idx = 3*u;
        denominator += (
          derivatives_old_unique_[i][idx    ] * new_derivative_unique[idx    ] +
          derivatives_old_unique_[i][idx + 1] * new_derivative_unique[idx + 1] +
          derivatives_old_unique_[i][idx + 2] * new_derivative_unique[idx + 2]
        ) / mreal;
      }
      denominator += effectiveAuxiliaryInverseMass_(i);

      const double numerator = constraintMismatch_(i, getArgument(i), s_aux_[i]);
      const double increment = -numerator / denominator;
      lambda_multipliers_[i] += increment;

      for(unsigned u=0; u<topo.unique_to_local.size(); ++u) {
        const unsigned local = topo.unique_to_local[u];
        const double mreal = actionAtomistic_vector_[i]->getMass(local);
        const unsigned idx = 3*u;
        pos[local][0] += derivatives_old_unique_[i][idx    ] * increment / mreal;
        pos[local][1] += derivatives_old_unique_[i][idx + 1] * increment / mreal;
        pos[local][2] += derivatives_old_unique_[i][idx + 2] * increment / mreal;
      }
      syncDuplicatePositions_(i, pos);

      if(do_only_blue_moon_==0) {
        s_aux_[i] -= increment / mass_[i];
        fold_if_requested_(&s_aux_[i], period_s_aux_[i]);
      }

      actionAtomistic_vector_[i]->applyPositions();
      actionAtomistic_vector_[i]->updatePositions();
      getPntrToArgument(i)->clearDerivatives();
      actionAtomistic_vector_[i]->calculate();
    }

    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      const double diff = constraintMismatch_(i, getArgument(i), s_aux_[i]);
      stop_iteration_condition += diff*diff;
    }

    converged = (stop_iteration_condition < kStopIterationAccuracy);
  }

  // Velocity correction corresponding to the converged SHAKE multiplier.
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    actionAtomistic_vector_[i]->retrieveAtoms();
    std::vector<Vector>& vel = actionAtomistic_vector_[i]->modifyVelocities();
    const CvTopology& topo = cv_topology_[i];

    for(unsigned u=0; u<topo.unique_to_local.size(); ++u) 
    {
      const unsigned local = topo.unique_to_local[u];
      const double mreal = actionAtomistic_vector_[i]->getMass(local);
      const unsigned idx = 3*u;
      vel[local][0] += derivatives_old_unique_[i][idx    ] * lambda_multipliers_[i] / (mreal * dt);
      vel[local][1] += derivatives_old_unique_[i][idx + 1] * lambda_multipliers_[i] / (mreal * dt);
      vel[local][2] += derivatives_old_unique_[i][idx + 2] * lambda_multipliers_[i] / (mreal * dt);
    }
    syncDuplicateVelocities_(i, vel);

    if(do_only_blue_moon_==0) {
      vs_aux_[i] -= lambda_multipliers_[i] / (mass_[i] * dt);
    }

    actionAtomistic_vector_[i]->applyVelocities();
    actionAtomistic_vector_[i]->updateVelocities();
  }
}

void Moonie::rattle_() {
  if(do_only_blue_moon_==0) {
  
    if(skip_first_restart_shake_drift_) 
    {
      //log.printf("  [MOONIE DEBUG] skipping first restarted RATTLE drift/noise step\n");
      skip_first_restart_shake_drift_ = false;
    } 
    else 
    {
      for(unsigned i=0; i<getNumberOfArguments(); ++i) 
      {
	      vs_aux_[i] *= std::sqrt(a_thermo_[i]);
	      vs_aux_[i] += std::sqrt(kbt_*(1.0-a_thermo_[i])/mass_[i]) * gaussian_(generator_);      
      }
    }
  }

  updateDerivativesOld_();
  computeZ_();
  computeDetZ_();

  // First RATTLE prediction using the previous mu as predictor.
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    actionAtomistic_vector_[i]->retrieveAtoms();
    std::vector<Vector>& vel = actionAtomistic_vector_[i]->modifyVelocities();
    const CvTopology& topo = cv_topology_[i];

    for(unsigned u=0; u<topo.unique_to_local.size(); ++u) {
      const unsigned local = topo.unique_to_local[u];
      const double mreal = actionAtomistic_vector_[i]->getMass(local);
      const unsigned idx = 3*u;
      vel[local][0] += derivatives_old_unique_[i][idx    ] * mu_multipliers_[i] / mreal;
      vel[local][1] += derivatives_old_unique_[i][idx + 1] * mu_multipliers_[i] / mreal;
      vel[local][2] += derivatives_old_unique_[i][idx + 2] * mu_multipliers_[i] / mreal;
    }
    syncDuplicateVelocities_(i, vel);

    if(do_only_blue_moon_==0) {
      vs_aux_[i] -= mu_multipliers_[i] / mass_[i];
    }

    actionAtomistic_vector_[i]->applyVelocities();
    actionAtomistic_vector_[i]->updateVelocities();
  }

  bool converged = false;
  int iteration_number = 0;
  double stop_iteration_condition = 0.0;

  while(!converged) {
    ++iteration_number;
    if(iteration_number > kMaxRattleIterations) {
      std::ostringstream oss;
      oss << "RATTLE did not converge after " << kMaxRattleIterations
          << " iterations; final residual norm = " << stop_iteration_condition;
      error(oss.str());
    }

    stop_iteration_condition = 0.0;

    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      actionAtomistic_vector_[i]->retrieveAtoms();
      std::vector<Vector>& vel = actionAtomistic_vector_[i]->modifyVelocities();
      const CvTopology& topo = cv_topology_[i];

      double denominator = 0.0;
      for(unsigned u=0; u<topo.unique_to_local.size(); ++u) {
        const unsigned local = topo.unique_to_local[u];
        const double mreal = actionAtomistic_vector_[i]->getMass(local);
        const unsigned idx = 3*u;
        denominator += (
          derivatives_old_unique_[i][idx    ] * derivatives_old_unique_[i][idx    ] +
          derivatives_old_unique_[i][idx + 1] * derivatives_old_unique_[i][idx + 1] +
          derivatives_old_unique_[i][idx + 2] * derivatives_old_unique_[i][idx + 2]
        ) / mreal;
      }
      denominator += effectiveAuxiliaryInverseMass_(i);

      double increment = 0.0;
      for(unsigned u=0; u<topo.unique_to_local.size(); ++u) {
        const unsigned local = topo.unique_to_local[u];
        const unsigned idx = 3*u;
        increment -= (
          derivatives_old_unique_[i][idx    ] * vel[local][0] +
          derivatives_old_unique_[i][idx + 1] * vel[local][1] +
          derivatives_old_unique_[i][idx + 2] * vel[local][2]
        );
      }
      if(do_only_blue_moon_==0) {
        increment += vs_aux_[i];
      }
      increment /= denominator;
      mu_multipliers_[i] += increment;

      for(unsigned u=0; u<topo.unique_to_local.size(); ++u) {
        const unsigned local = topo.unique_to_local[u];
        const double mreal = actionAtomistic_vector_[i]->getMass(local);
        const unsigned idx = 3*u;
        vel[local][0] += derivatives_old_unique_[i][idx    ] * increment / mreal;
        vel[local][1] += derivatives_old_unique_[i][idx + 1] * increment / mreal;
        vel[local][2] += derivatives_old_unique_[i][idx + 2] * increment / mreal;
      }
      syncDuplicateVelocities_(i, vel);

      if(do_only_blue_moon_==0) {
        vs_aux_[i] -= increment / mass_[i];
      }

      actionAtomistic_vector_[i]->applyVelocities();
      actionAtomistic_vector_[i]->updateVelocities();
    }

    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      actionAtomistic_vector_[i]->retrieveAtoms();
      std::vector<Vector>& vel = actionAtomistic_vector_[i]->modifyVelocities();
      const CvTopology& topo = cv_topology_[i];

      double residual = 0.0;
      for(unsigned u=0; u<topo.unique_to_local.size(); ++u) {
        const unsigned local = topo.unique_to_local[u];
        const unsigned idx = 3*u;
        residual += (
          derivatives_old_unique_[i][idx    ] * vel[local][0] +
          derivatives_old_unique_[i][idx + 1] * vel[local][1] +
          derivatives_old_unique_[i][idx + 2] * vel[local][2]
        );
      }
      if(do_only_blue_moon_==0) {
        residual -= vs_aux_[i];
      }
      stop_iteration_condition += residual*residual;
    }

    converged = (stop_iteration_condition < kStopIterationAccuracy);
  }
}

void Moonie::debugPrintStageState_(const char* where) const 
{
  log.printf("  [MOONIE DEBUG] %s stage=%d restart_loaded=%d\n",
             where,
             shake_rattle_switch_,
             restart_state_loaded_ ? 1 : 0);

  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    log.printf("  [MOONIE DEBUG]   cv=%u arg=%.16g s=%.16g vs=%.16g lambda=%.16g mu=%.16g\n",
               i,
               getArgument(i),
               s_aux_[i],
               vs_aux_[i],
               lambda_multipliers_[i],
               mu_multipliers_[i]);

      log.printf("  [MOONIE DEBUG]   cv=%u grad_dot_vel=%.16g\n",
               i,
               computeVelocityProjection_(i));

  }
  
}


void Moonie::calculate() {
  if(firsttime_) 
  {
    initializeRuntimeState_();
  }

  // Detailed per-stage debug for the first restarted Moonie MD step.
  if(debug_restart_step_pending_ && debug_restart_calls_remaining_ > 0) {
    debugPrintStageState_("ENTRY");
  }

  if(shake_rattle_switch_ == 0) 
  {
  
    shake_();

    if(debug_restart_step_pending_ && debug_restart_calls_remaining_ > 0) {
      debugPrintStageState_("AFTER_SHAKE");
    }

    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      lambda_values_[i]->set(lambda_multipliers_[i] * 2.0 / std::pow(getTimeStep()*getStride(), 2));
    }
  }

  if(shake_rattle_switch_ == 2) 
  {
    rattle_();

    if(debug_restart_step_pending_ && debug_restart_calls_remaining_ > 0) {
      debugPrintStageState_("AFTER_RATTLE");
    }

    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      s_values_[i]->set(s_aux_[i]);
      mu_values_[i]->set(mu_multipliers_[i] * 2.0 / (getTimeStep()*getStride()));
    }
    det_Z_value_->set(det_Z_);
    det_Z_value_to_minus_half_->set(det_Z_to_minus_half_);
  }


  // Count down the first restarted Moonie step debug window.
  if(debug_restart_step_pending_ && debug_restart_calls_remaining_ > 0) 
  {
    --debug_restart_calls_remaining_;
    if(debug_restart_calls_remaining_ == 0) {
      debug_restart_step_pending_ = false;
      log.printf("  [MOONIE DEBUG] finished first restarted Moonie-step stage tracing\n");
    }
  }

  if(shake_rattle_switch_ == 2) 
  {
    shake_rattle_switch_ = -1;
  }
  ++shake_rattle_switch_;
}



void Moonie::update() 
{
  if (plumed.getCPT()) 
  {
    saveRestartState_();
  }
}

} // namespace bias
} // namespace PLMD
