pub struct Variable {
    variable: cxx::UniquePtr<crate::ffi::Variable>,
}

impl Variable {
    pub fn new() -> Variable {
        Variable {
            variable: crate::ffi::variable_new(),
        }
    }

    pub fn new_with_value(value: f64) -> Variable {
        Variable {
            variable: crate::ffi::variable_new_with_value(value),
        }
    }

    pub fn set_value(&mut self, value: f64) {
        self.variable.pin_mut().SetValue(value);
    }
}

impl Default for Variable {
    fn default() -> Self {
        Self::new()
    }
}
