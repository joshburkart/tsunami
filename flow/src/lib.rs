#![deny(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]

pub mod bases;
pub mod examples;
mod odeint;
pub mod physics;
#[cfg(test)]
mod test_util;

use ndarray as nd;

pub type Float = f32;
type ComplexFloat = ndrustfft::Complex<Float>;
pub use std::f32::consts as float_consts;

pub trait RawFloatData: nd::RawData<Elem = Float> + nd::Data {}
pub trait RawComplexFloatData: nd::RawData<Elem = ComplexFloat> + nd::Data {}

impl<S: nd::RawData<Elem = Float> + nd::Data> RawFloatData for S {}
impl<S: nd::RawData<Elem = ComplexFloat> + nd::Data> RawComplexFloatData for S {}
