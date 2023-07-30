#![deny(clippy::all)]

mod derivs;
mod examples;
mod fields;
mod geom;
mod implicit;
mod indexing;
mod linalg;
mod physics;

use nalgebra as na;
use ndarray as nd;
use pyo3::prelude::*;

type Float = f64;

type Vector2 = na::Vector2<Float>;
type Vector3 = na::Vector3<Float>;
type Matrix3 = na::Matrix3<Float>;
type UnitVector2 = na::UnitVector2<Float>;
type UnitVector3 = na::UnitVector3<Float>;
type Point2 = na::Point2<Float>;
type Point3 = na::Point3<Float>;

type Array1 = nd::Array1<Float>;
type Array2 = nd::Array2<Float>;
type Array3 = nd::Array3<Float>;

#[pymodule]
fn flow(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(examples::bump_1d, m)?)?;
    m.add_function(wrap_pyfunction!(examples::ramp_1d, m)?)?;
    m.add_function(wrap_pyfunction!(examples::uniform, m)?)
}
