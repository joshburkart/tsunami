#![deny(clippy::all)]

pub mod geom;
pub mod indexing;
pub mod physics;

use nalgebra as na;
use ndarray::{self as nd, s};

pub type Float = f64;

pub type Vector2 = na::Vector2<Float>;
pub type Vector3 = na::Vector3<Float>;
pub type UnitVector2 = na::UnitVector2<Float>;
pub type UnitVector3 = na::UnitVector3<Float>;

pub type Array1 = nd::Array1<Float>;
pub type Array2 = nd::Array2<Float>;
pub type Array3 = nd::Array3<Float>;
