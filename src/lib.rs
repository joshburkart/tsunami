#![deny(clippy::all)]

pub mod examples;
pub mod geom;
pub mod indexing;
pub mod physics;

use nalgebra as na;
use ndarray as nd;
use pyo3::prelude::*;

pub type Float = f64;

pub type Vector2 = na::Vector2<Float>;
pub type Vector3 = na::Vector3<Float>;
pub type UnitVector2 = na::UnitVector2<Float>;
pub type UnitVector3 = na::UnitVector3<Float>;

pub type Array1 = nd::Array1<Float>;
pub type Array2 = nd::Array2<Float>;
pub type Array3 = nd::Array3<Float>;

#[pymodule]
fn riversim(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(examples::advection_1d, m)?)
}
