use nalgebra as na;
use ndarray::{self as nd};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::{
    geom,
    indexing::{self, Index, Indexing},
    Array1, Array2, Array3, Float, Matrix3, Vector3,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ShearField {
    /// Indexed by [`indexing::CellIndexing`]
    cells: nd::Array4<Matrix3>,
}
impl ShearField {
    pub fn new<F: Fn(Float, Float, Float) -> Matrix3>(
        dynamic_geometry: &geom::DynamicGeometry,
        f: F,
    ) -> Self {
        Self {
            cells: dynamic_geometry.make_cell_array(f),
        }
    }

    pub fn zeros(cell_indexing: &indexing::CellIndexing) -> Self {
        Self {
            cells: nd::Array4::zeros(cell_indexing.shape()),
        }
    }

    pub fn cell_value(&self, index: indexing::CellIndex) -> Matrix3 {
        self.cells[index.to_array_index()]
    }

    pub fn cell_value_mut(&mut self, index: indexing::CellIndex) -> &mut Matrix3 {
        &mut self.cells[index.to_array_index()]
    }

    pub fn compute_divergence(&self, dynamic_geometry: &geom::DynamicGeometry) -> VelocityField {
        let mut divergence = VelocityField::zeros(dynamic_geometry.grid().cell_indexing());
        for cell_index in indexing::iter_indices(dynamic_geometry.grid().cell_indexing()) {
            let cell = dynamic_geometry.cell(cell_index);
            let cell_value = self.cell_value(cell_index);
            for face in &cell.faces {
                if let indexing::CellNeighbor::Interior(neighbor_cell_index) = face.neighbor() {
                    *divergence.cell_value_mut(cell_index) += 0.5
                        * face.area()
                        * (cell_value + self.cell_value(neighbor_cell_index))
                            .tr_mul(&face.outward_normal().into_inner());
                } else {
                    *divergence.cell_value_mut(cell_index) +=
                        face.area() * cell_value.tr_mul(&face.outward_normal().into_inner());
                }
            }
            *divergence.cell_value_mut(cell_index) /= cell.volume;
        }
        divergence
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for ShearField {
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.cells
            .iter()
            .zip(other.cells.iter())
            .all(|(left, right)| left.abs_diff_eq(right, epsilon))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VelocityField {
    /// Indexed by [`indexing::CellIndexing`]
    cells: nd::Array4<Vector3>,
}
impl VelocityField {
    pub fn new<F: Fn(Float, Float, Float) -> Vector3>(
        dynamic_geometry: &geom::DynamicGeometry,
        f: F,
    ) -> Self {
        Self {
            cells: dynamic_geometry.make_cell_array(f),
        }
    }

    pub fn zeros(cell_indexing: &indexing::CellIndexing) -> Self {
        Self {
            cells: nd::Array4::zeros(cell_indexing.shape()),
        }
    }

    pub fn cell_value(&self, index: indexing::CellIndex) -> Vector3 {
        self.cells[index.to_array_index()]
    }

    pub fn cell_value_mut(&mut self, index: indexing::CellIndex) -> &mut Vector3 {
        &mut self.cells[index.to_array_index()]
    }

    pub fn compute_shear(&self, dynamic_geometry: &geom::DynamicGeometry) -> ShearField {
        let mut gradient = ShearField::zeros(dynamic_geometry.grid().cell_indexing());
        for cell_index in indexing::iter_indices(dynamic_geometry.grid().cell_indexing()) {
            let cell = dynamic_geometry.cell(cell_index);
            let cell_value = self.cell_value(cell_index);
            for face in &cell.faces {
                if let indexing::CellNeighbor::Interior(neighbor_cell_index) = face.neighbor() {
                    *gradient.cell_value_mut(cell_index) += 0.5
                        * face.area()
                        * (face.outward_normal().into_inner()
                            * (cell_value + self.cell_value(neighbor_cell_index)).transpose());
                } else {
                    *gradient.cell_value_mut(cell_index) +=
                        face.area() * face.outward_normal().into_inner() * cell_value.transpose();
                }
            }
            *gradient.cell_value_mut(cell_index) /= cell.volume;
        }
        gradient
    }

    pub fn compute_laplacian(&self, dynamic_geometry: &geom::DynamicGeometry) -> Self {
        self.compute_shear(dynamic_geometry)
            .compute_divergence(dynamic_geometry)
    }

    pub fn column_average(&self) -> nd::Array3<na::Vector2<Float>> {
        self.cells
            .mapv(|velocity| na::Vector2::new(velocity.x, velocity.y))
            .sum_axis(nd::Axis(2))
            / self.cells.shape()[2] as Float
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for VelocityField {
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.cells
            .iter()
            .zip(other.cells.iter())
            .all(|(cell, other_cell)| cell.abs_diff_eq(other_cell, epsilon))
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct PressureField {
    /// Indexed by [`indexing::VertexIndexing`]
    vertices: Array3,
}
impl PressureField {
    pub fn new<F: Fn(Float, Float, Float) -> Float>(
        dynamic_geometry: &geom::DynamicGeometry,
        f: F,
    ) -> Self {
        Self {
            vertices: dynamic_geometry.make_vertex_array(f),
        }
    }

    pub fn zeros(vertex_indexing: &indexing::VertexIndexing) -> Self {
        Self {
            vertices: Array3::zeros(vertex_indexing.shape()),
        }
    }

    pub fn vertex_value(&self, vertex: indexing::VertexIndex) -> Float {
        self.vertices[vertex.to_array_index()]
    }

    pub fn vertex_value_mut(&mut self, vertex: indexing::VertexIndex) -> &mut Float {
        &mut self.vertices[vertex.to_array_index()]
    }

    pub fn compute_gradient(&self, dynamic_geometry: &geom::DynamicGeometry) -> VelocityField {
        let mut gradient_field = VelocityField::zeros(dynamic_geometry.grid().cell_indexing());
        for cell_index in indexing::iter_indices(dynamic_geometry.grid().cell_indexing()) {
            let cell = dynamic_geometry.cell(cell_index);
            for face in &cell.faces {
                let face_interp_value = match face.vertices() {
                    geom::CellFaceVertices::Vertical(vertices) => {
                        vertices
                            .iter()
                            .map(|vertex| self.vertex_value(*vertex))
                            .sum::<Float>()
                            / vertices.len() as Float
                    }
                    geom::CellFaceVertices::Horizontal(vertices) => {
                        vertices
                            .iter()
                            .map(|vertex| self.vertex_value(*vertex))
                            .sum::<Float>()
                            / vertices.len() as Float
                    }
                };
                *gradient_field.cell_value_mut(cell_index) +=
                    face.outward_normal().into_inner() * face.area() * face_interp_value;
            }
            *gradient_field.cell_value_mut(cell_index) /= cell.volume;
        }
        gradient_field
    }

    pub fn flatten(&self, vertex_indexing: &indexing::VertexIndexing) -> Array1 {
        indexing::flatten_array(vertex_indexing, &self.vertices)
    }

    pub fn unflatten(vertex_indexing: &indexing::VertexIndexing, flattened: &Array1) -> Self {
        let vertices = indexing::unflatten_array(vertex_indexing, flattened);
        Self { vertices }
    }
}
#[pymethods]
impl PressureField {
    #[pyo3(name = "pressure")]
    pub fn pressure_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.vertices.clone().into_pyarray(py)
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for PressureField {
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.vertices.abs_diff_eq(&other.vertices, epsilon)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct HeightField {
    /// Indexed by [`indexing::CellFootprintIndexing`]
    centers: Array3,
}
impl HeightField {
    pub fn new<F: Fn(Float, Float) -> Float>(grid: &geom::Grid, f: F) -> Self {
        Self {
            centers: grid.make_cell_footprint_array(f),
        }
    }

    pub fn center_value(&self, cell_footprint_index: indexing::CellFootprintIndex) -> Float {
        self.centers[cell_footprint_index.to_array_index()]
    }

    pub fn center_value_mut(
        &mut self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> &mut Float {
        &mut self.centers[cell_footprint_index.to_array_index()]
    }

    // TODO: Remove otherwise what's the point of this wrapper struct...
    pub fn centers(&self) -> &Array3 {
        &self.centers
    }
}
impl std::ops::Add for HeightField {
    type Output = Self;

    fn add(mut self, rhs: HeightField) -> Self::Output {
        self += rhs;
        self
    }
}
impl std::ops::AddAssign for HeightField {
    fn add_assign(&mut self, rhs: HeightField) {
        self.centers += &rhs.centers;
    }
}
impl std::ops::Add<Float> for HeightField {
    type Output = Self;

    fn add(mut self, rhs: Float) -> Self::Output {
        self += rhs;
        self
    }
}
impl std::ops::Add<Float> for &HeightField {
    type Output = HeightField;

    fn add(self, rhs: Float) -> Self::Output {
        let mut new = self.clone();
        new += rhs;
        new
    }
}
impl std::ops::Add<HeightField> for Float {
    type Output = HeightField;

    fn add(self, mut rhs: HeightField) -> Self::Output {
        rhs += self;
        rhs
    }
}
impl std::ops::Add<&HeightField> for Float {
    type Output = HeightField;

    fn add(self, rhs: &HeightField) -> Self::Output {
        let mut new = rhs.clone();
        new += self;
        new
    }
}
impl std::ops::AddAssign<Float> for HeightField {
    fn add_assign(&mut self, rhs: Float) {
        self.centers += rhs;
    }
}
impl std::ops::Mul<HeightField> for Float {
    type Output = HeightField;

    fn mul(self, rhs: HeightField) -> Self::Output {
        HeightField {
            centers: self * rhs.centers,
        }
    }
}
impl std::ops::Mul<&HeightField> for Float {
    type Output = HeightField;

    fn mul(self, rhs: &HeightField) -> Self::Output {
        HeightField {
            centers: self * &rhs.centers,
        }
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for HeightField {
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.centers.abs_diff_eq(&other.centers, epsilon)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Terrain {
    // TODO use or remove
    // /// Indexed by [`indexing::CellFootprintIndexing`]
    // centers: Array3,
    /// Indexed by [`indexing::VertexFootprintIndexing`]
    vertices: Array2,
}
impl Terrain {
    pub fn new<F: Fn(Float, Float) -> Float>(grid: &geom::Grid, f: &F) -> Self {
        Self {
            // TODO use or remove
            // centers: self.make_cell_footprint_array(f),
            vertices: grid.make_vertex_footprint_array(f),
        }
    }

    pub fn vertex_value(&self, vertex_footprint_index: indexing::VertexFootprintIndex) -> Float {
        self.vertices[vertex_footprint_index.to_array_index()]
    }
}
