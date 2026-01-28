// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/ocp.hpp>

namespace em = emscripten;

namespace slp {

void bind_ocp(em::class_<OCP<double>, Problem<double>>& cls) {
  cls.constructor<
      int, int, std::chrono::duration<double>, int,
      const std::function<VariableMatrix<double>(
          const VariableMatrix<double>& x, const VariableMatrix<double>& u)>&,
      DynamicsType, TimestepMethod, TranscriptionMethod>();

  cls.function("constrain_initial_state",
               &OCP<double>::constrain_initial_state<double>);
  cls.function("constrain_initial_state",
               &OCP<double>::constrain_initial_state<int>);
  cls.function("constrain_initial_state",
               &OCP<double>::constrain_initial_state<Variable<double>>);
  cls.function("constrain_initial_state",
               [](OCP<double>& self, em::DRef<Eigen::MatrixXd> initial_state) {
                 self.constrain_initial_state(initial_state);
               });
  cls.function("constrain_initial_state",
               &OCP<double>::constrain_initial_state<VariableMatrix<double>>);

  cls.function("constrain_final_state",
               &OCP<double>::constrain_final_state<double>);
  cls.function("constrain_final_state",
               &OCP<double>::constrain_final_state<int>);
  cls.function("constrain_final_state",
               &OCP<double>::constrain_final_state<Variable<double>>);
  cls.function("constrain_final_state",
               [](OCP<double>& self, em::DRef<Eigen::MatrixXd> final_state) {
                 self.constrain_final_state(final_state);
               });
  cls.function("constrain_final_state",
               &OCP<double>::constrain_final_state<VariableMatrix<double>>);

  cls.function(
      "for_each_step",
      [](OCP<double>& self,
         const std::function<void(const VariableMatrix<double>& x,
                                  const VariableMatrix<double>& u)>& callback) {
        self.for_each_step(callback);
      });

  cls.function("set_lower_input_bound",
               &OCP<double>::set_lower_input_bound<double>);
  cls.function("set_lower_input_bound",
               &OCP<double>::set_lower_input_bound<int>);
  cls.function("set_lower_input_bound",
               &OCP<double>::set_lower_input_bound<Variable<double>>);
  cls.function("set_lower_input_bound",
               [](OCP<double>& self, em::DRef<Eigen::MatrixXd> lower_bound) {
                 self.set_lower_input_bound(lower_bound);
               });
  cls.function("set_lower_input_bound",
               &OCP<double>::set_lower_input_bound<VariableMatrix<double>>);

  cls.function("set_upper_input_bound",
               &OCP<double>::set_upper_input_bound<double>);
  cls.function("set_upper_input_bound",
               &OCP<double>::set_upper_input_bound<int>);
  cls.function("set_upper_input_bound",
               &OCP<double>::set_upper_input_bound<Variable<double>>);
  cls.function("set_upper_input_bound",
               [](OCP<double>& self, em::DRef<Eigen::MatrixXd> upper_bound) {
                 self.set_upper_input_bound(upper_bound);
               });
  cls.function("set_upper_input_bound",
               &OCP<double>::set_upper_input_bound<VariableMatrix<double>>);

  cls.function("set_min_timestep", &OCP<double>::set_min_timestep);
  cls.function("set_max_timestep", &OCP<double>::set_max_timestep);

  cls.function("X", &OCP<double>::X);
  cls.function("U", &OCP<double>::U);
  cls.function("dt", &OCP<double>::dt);
  cls.function("initial_state", &OCP<double>::initial_state);
  cls.function("final_state", &OCP<double>::final_state);
}

}  // namespace slp
