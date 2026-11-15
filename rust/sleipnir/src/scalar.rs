// Copyright (c) Sleipnir contributors

use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

// The C library's erf, which the C++ library also uses
mod c_math {
    unsafe extern "C" {
        pub safe fn erf(x: f64) -> f64;
        pub safe fn erff(x: f32) -> f32;
    }
}

macro_rules! scalar_trait {
    ($($name:ident($($arg:ident),*)),* $(,)?) => {
        /// Floating-point type a problem is solved in. Implemented for `f32`
        /// and `f64`.
        pub trait Scalar:
            crate::ffi::Backend
            + Copy
            + Debug
            + Display
            + PartialEq
            + PartialOrd
            + Add<Output = Self>
            + Sub<Output = Self>
            + Mul<Output = Self>
            + Div<Output = Self>
            + Neg<Output = Self>
            + Send
            + Sync
            + 'static
        {
            /// Additive identity.
            const ZERO: Self;

            /// Multiplicative identity.
            const ONE: Self;

            /// Machine epsilon.
            const EPSILON: Self;

            #[doc(hidden)]
            fn erf(self) -> Self;

            #[doc(hidden)]
            fn from_f64(x: f64) -> Self;

            $(
                #[doc(hidden)]
                fn $name(self $(, $arg: Self)*) -> Self;
            )*
        }

        scalar_trait!(@impl f32, $($name($($arg),*)),*);
        scalar_trait!(@impl f64, $($name($($arg),*)),*);
    };
    (@erf f32, $x:expr) => { c_math::erff($x) };
    (@erf f64, $x:expr) => { c_math::erf($x) };
    (@impl $S:ident, $($name:ident($($arg:ident),*)),*) => {
        impl Scalar for $S {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const EPSILON: Self = $S::EPSILON;

            fn erf(self) -> Self {
                scalar_trait!(@erf $S, self)
            }

            fn from_f64(x: f64) -> Self {
                x as $S
            }

            $(
                fn $name(self $(, $arg: Self)*) -> Self {
                    $S::$name(self $(, $arg)*)
                }
            )*
        }
    };
}

scalar_trait!(
    abs(),
    acos(),
    asin(),
    atan(),
    cbrt(),
    cos(),
    cosh(),
    exp(),
    ln(),
    log10(),
    sin(),
    sinh(),
    sqrt(),
    tan(),
    tanh(),
    powf(exponent),
    atan2(x),
    hypot(other),
    max(other),
    min(other),
);
