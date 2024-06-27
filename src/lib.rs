use cxx::{type_id, ExternType};

unsafe impl ExternType for crate::autodiff::Variable {
    type Id = type_id!("sleipnir::Variable");
    type Kind = cxx::kind::Opaque;
}

#[cxx::bridge(namespace = "sleipnir")]
mod ffi {
    unsafe extern "C++" {
        include!("RustFFI.hpp");

        type Variable;

        fn variable_new() -> UniquePtr<Variable>;
        fn variable_new_with_value(value: f64) -> UniquePtr<Variable>;

        fn SetValue(self: Pin<&mut Variable>, value: f64);
    }
}

mod autodiff;
mod optimization;
