#[cxx::bridge(namespace = "sleipnirrust")]
mod ffi {
    unsafe extern "C++" {
        include!("sleipnirrust.hpp");

        type Variable;

        fn new_variable() -> UniquePtr<Variable>;
    }
}

pub struct Variable {
    variable: cxx::UniquePtr<crate::ffi::Variable>,
}

impl Variable {
    pub fn new() -> Variable {
        Variable {
            variable: crate::ffi::new_variable(),
        }
    }
}

impl Default for Variable {
    fn default() -> Self {
        Self::new()
    }
}
