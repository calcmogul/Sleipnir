// Copyright (c) Sleipnir contributors

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/function.h>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/ocp.hpp>

#include "docstrings.hpp"

namespace nb = nanobind;

namespace slp {

void bind_ocp(nb::class_<OCP<double>, Problem<double>>& cls) {
  using namespace nb::literals;

  cls.def(nb::init<int, int, std::chrono::duration<double>, int,
                   const std::function<VariableMatrix<double>(
                       const VariableMatrix<double>& x,
                       const VariableMatrix<double>& u)>&,
                   DynamicsType, TimestepMethod, TranscriptionMethod>(),
          "num_states"_a, "num_inputs"_a, "dt"_a, "num_steps"_a, "dynamics"_a,
          "dynamics_type"_a = DynamicsType::EXPLICIT_ODE,
          "timestep_method"_a = TimestepMethod::FIXED,
          "transcription_method"_a = TranscriptionMethod::DIRECT_TRANSCRIPTION,
          DOC(slp, OCP, OCP));

  cls.def("constrain_initial_state",
          &OCP<double>::constrain_initial_state<double>, "initial_state"_a,
          DOC(slp, OCP, constrain_initial_state));
  cls.def("constrain_initial_state", &OCP<double>::constrain_initial_state<int>,
          "initial_state"_a, DOC(slp, OCP, constrain_initial_state));
  cls.def("constrain_initial_state",
          &OCP<double>::constrain_initial_state<Variable<double>>,
          "initial_state"_a, DOC(slp, OCP, constrain_initial_state));
  cls.def(
      "constrain_initial_state",
      [](OCP<double>& self, nb::DRef<Eigen::MatrixXd> initial_state) {
        self.constrain_initial_state(initial_state);
      },
      "initial_state"_a, DOC(slp, OCP, constrain_initial_state));
  cls.def("constrain_initial_state",
          &OCP<double>::constrain_initial_state<VariableMatrix<double>>,
          "initial_state"_a, DOC(slp, OCP, constrain_initial_state));

  cls.def("constrain_final_state", &OCP<double>::constrain_final_state<double>,
          "final_state"_a, DOC(slp, OCP, constrain_final_state));
  cls.def("constrain_final_state", &OCP<double>::constrain_final_state<int>,
          "final_state"_a, DOC(slp, OCP, constrain_final_state));
  cls.def("constrain_final_state",
          &OCP<double>::constrain_final_state<Variable<double>>,
          "final_state"_a, DOC(slp, OCP, constrain_final_state));
  cls.def(
      "constrain_final_state",
      [](OCP<double>& self, nb::DRef<Eigen::MatrixXd> final_state) {
        self.constrain_final_state(final_state);
      },
      "final_state"_a, DOC(slp, OCP, constrain_final_state));
  cls.def("constrain_final_state",
          &OCP<double>::constrain_final_state<VariableMatrix<double>>,
          "final_state"_a, DOC(slp, OCP, constrain_final_state));

  cls.def(
      "for_each_step",
      [](OCP<double>& self,
         const std::function<void(const VariableMatrix<double>& x,
                                  const VariableMatrix<double>& u)>& callback) {
        self.for_each_step(callback);
      },
      "callback"_a, DOC(slp, OCP, for_each_step));

  cls.def("set_lower_input_bound", &OCP<double>::set_lower_input_bound<double>,
          "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));
  cls.def("set_lower_input_bound", &OCP<double>::set_lower_input_bound<int>,
          "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));
  cls.def("set_lower_input_bound",
          &OCP<double>::set_lower_input_bound<Variable<double>>,
          "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));
  cls.def(
      "set_lower_input_bound",
      [](OCP<double>& self, nb::DRef<Eigen::MatrixXd> lower_bound) {
        self.set_lower_input_bound(lower_bound);
      },
      "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));
  cls.def("set_lower_input_bound",
          &OCP<double>::set_lower_input_bound<VariableMatrix<double>>,
          "lower_bound"_a, DOC(slp, OCP, set_lower_input_bound));

  cls.def("set_upper_input_bound", &OCP<double>::set_upper_input_bound<double>,
          "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));
  cls.def("set_upper_input_bound", &OCP<double>::set_upper_input_bound<int>,
          "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));
  cls.def("set_upper_input_bound",
          &OCP<double>::set_upper_input_bound<Variable<double>>,
          "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));
  cls.def(
      "set_upper_input_bound",
      [](OCP<double>& self, nb::DRef<Eigen::MatrixXd> upper_bound) {
        self.set_upper_input_bound(upper_bound);
      },
      "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));
  cls.def("set_upper_input_bound",
          &OCP<double>::set_upper_input_bound<VariableMatrix<double>>,
          "upper_bound"_a, DOC(slp, OCP, set_upper_input_bound));

  cls.def("set_min_timestep", &OCP<double>::set_min_timestep, "min_timestep"_a,
          DOC(slp, OCP, set_min_timestep));
  cls.def("set_max_timestep", &OCP<double>::set_max_timestep, "max_timestep"_a,
          DOC(slp, OCP, set_max_timestep));

  cls.def("X", &OCP<double>::X, DOC(slp, OCP, X));
  cls.def("U", &OCP<double>::U, DOC(slp, OCP, U));
  cls.def("dt", &OCP<double>::dt, DOC(slp, OCP, dt));
  cls.def("initial_state", &OCP<double>::initial_state,
          DOC(slp, OCP, initial_state));
  cls.def("final_state", &OCP<double>::final_state, DOC(slp, OCP, final_state));
}

}  // namespace slp
