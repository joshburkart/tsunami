use nalgebra as na;
use ndarray::{self as nd};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::{
    geom,
    indexing::{self, Index, Indexing, IntoIndexIterator},
    Array1, Array2, Array3, Array5, Float, Matrix3, Point3, UnitVector3, Vector2, Vector3,
};

#[derive(Clone, Copy)]
pub struct HorizBoundaryConditions {
    pub x: BoundaryConditionPair,
    pub y: BoundaryConditionPair,
}
impl HorizBoundaryConditions {
    pub fn hom_neumann() -> Self {
        Self {
            x: BoundaryConditionPair::hom_neumann(),
            y: BoundaryConditionPair::hom_neumann(),
        }
    }

    pub fn hom_dirichlet() -> Self {
        Self {
            x: BoundaryConditionPair::hom_dirichlet(),
            y: BoundaryConditionPair::hom_dirichlet(),
        }
    }
}
#[derive(Clone, Copy)]
pub struct BoundaryConditions {
    pub horiz: HorizBoundaryConditions,
    pub z: BoundaryConditionPair,
}
#[derive(Clone, Copy)]
pub struct BoundaryConditionPair {
    pub lower: BoundaryCondition,
    pub upper: BoundaryCondition,
}
impl BoundaryConditionPair {
    pub fn hom_neumann() -> Self {
        Self {
            lower: BoundaryCondition::HomNeumann,
            upper: BoundaryCondition::HomNeumann,
        }
    }

    pub fn hom_dirichlet() -> Self {
        Self {
            lower: BoundaryCondition::HomDirichlet,
            upper: BoundaryCondition::HomDirichlet,
        }
    }
}
#[derive(Clone, Copy)]
pub enum BoundaryCondition {
    HomDirichlet,
    HomNeumann,
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarField {
    /// Indexed by [`indexing::VertexIndexing`]
    vertices: Array3,
}
impl ScalarField {
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

    pub fn compute_gradient(&self, dynamic_geometry: &geom::DynamicGeometry) -> VectorField {
        let mut gradient_field = VectorField::zeros(dynamic_geometry.grid().cell_indexing());
        for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
            let cell = dynamic_geometry.cell(cell_index);
            for face in &cell.faces {
                let face_interp_value = match face.vertices() {
                    geom::CellFaceVertices::Vert(vertices) => {
                        vertices
                            .iter()
                            .map(|vertex| self.vertex_value(*vertex))
                            .sum::<Float>()
                            / vertices.len() as Float
                    }
                    geom::CellFaceVertices::Horiz(vertices) => {
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
impl ScalarField {
    pub fn values_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.vertices.clone().into_pyarray(py)
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for ScalarField {
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.vertices.abs_diff_eq(&other.vertices, epsilon)
    }
}

#[pyclass]
#[derive(Clone, Debug, PartialEq)]
pub struct VectorField {
    /// Indexed by [`indexing::CellIndexing`]
    cells: nd::Array4<Vector3>,
}
impl VectorField {
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

    pub fn compute_gradient(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions,
    ) -> TensorField {
        let mut gradient = TensorField::zeros(dynamic_geometry.grid().cell_indexing());
        for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
            *gradient.cell_value_mut(cell_index) =
                compute_cell_differential(dynamic_geometry, self, cell_index, boundary_conditions);
        }
        gradient
    }

    pub fn compute_laplacian(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        gradient: &TensorField,
        boundary_conditions: &BoundaryConditions,
    ) -> Self {
        struct LaplacianComputer<'a> {
            vector_field: &'a VectorField,
            gradient_field: &'a TensorField,
        }

        impl<'a> CellDifferentialOpComputer<Vector3> for LaplacianComputer<'a> {
            fn compute_interior_face_value(
                &self,
                cell: &geom::Cell,
                neighbor_cell: &geom::Cell,
                outward_normal: UnitVector3,
            ) -> Vector3 {
                let average_gradient = 0.5
                    * (self.gradient_field.cell_value(cell.index)
                        + self.gradient_field.cell_value(neighbor_cell.index));

                let displ = neighbor_cell.centroid - cell.centroid;
                let c_corr = 1. / outward_normal.dot(&displ);
                c_corr
                    * (self.vector_field.cell_value(neighbor_cell.index)
                        - self.vector_field.cell_value(cell.index))
                    + average_gradient.tr_mul(&(outward_normal.into_inner() - c_corr * displ))
            }

            fn compute_boundary_face_value(
                &self,
                cell: &geom::Cell,
                block_paired_cell: &geom::Cell,
                _outward_normal: UnitVector3,
                face_centroid: Point3,
                boundary_condition: &BoundaryCondition,
            ) -> Vector3 {
                match boundary_condition {
                    BoundaryCondition::HomDirichlet => {
                        // TODO: 0.5?
                        -0.25
                            * (self.vector_field.cell_value(cell.index)
                                + self.vector_field.cell_value(block_paired_cell.index))
                            / (face_centroid.coords
                                - 0.5 * (cell.centroid.coords + block_paired_cell.centroid.coords))
                                .norm()
                    }
                    BoundaryCondition::HomNeumann => Vector3::zeros(),
                }
            }
        }

        let cell_indexing = dynamic_geometry.grid().cell_indexing();
        let mut laplacian = VectorField::zeros(cell_indexing);
        let laplacian_computer = LaplacianComputer {
            vector_field: self,
            gradient_field: gradient,
        };
        for cell_index in cell_indexing.iter() {
            *laplacian.cell_value_mut(cell_index) = compute_cell_differential(
                dynamic_geometry,
                &laplacian_computer,
                cell_index,
                boundary_conditions,
            );
        }
        laplacian
    }

    pub fn advect_upwind(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions,
    ) -> Self {
        struct AdvectionComputer<'a> {
            velocity: &'a VectorField,
        }
        impl<'a> CellDifferentialOpComputer<Vector3> for AdvectionComputer<'a> {
            fn compute_interior_face_value(
                &self,
                cell: &geom::Cell,
                neighbor_cell: &geom::Cell,
                outward_normal: UnitVector3,
            ) -> Vector3 {
                let cell_velocity = self.velocity.cell_value(cell.index);
                let neighbor_cell_velocity = self.velocity.cell_value(neighbor_cell.index);
                let linear_face_velocity = 0.5 * (cell_velocity + neighbor_cell_velocity);

                let projected_face_velocity = linear_face_velocity.dot(&outward_normal);
                let neighbor_is_upwind = projected_face_velocity < 0.;
                let upwind_face_velocity = if neighbor_is_upwind {
                    neighbor_cell_velocity
                } else {
                    cell_velocity
                };

                upwind_face_velocity.dot(&outward_normal) * linear_face_velocity
            }

            fn compute_boundary_face_value(
                &self,
                cell: &geom::Cell,
                block_paired_cell: &geom::Cell,
                outward_normal: UnitVector3,
                _face_centroid: Point3,
                boundary_condition: &BoundaryCondition,
            ) -> Vector3 {
                let face_velocity = match boundary_condition {
                    BoundaryCondition::HomDirichlet => return Vector3::zeros(),
                    BoundaryCondition::HomNeumann => {
                        0.5 * (self.velocity.cell_value(cell.index)
                            + self.velocity.cell_value(block_paired_cell.index))
                    }
                };

                face_velocity.dot(&outward_normal) * face_velocity
            }
        }

        let cell_indexing = dynamic_geometry.grid().cell_indexing();
        let mut advected = VectorField::zeros(cell_indexing);
        for cell_index in cell_indexing.iter() {
            *advected.cell_value_mut(cell_index) = compute_cell_differential(
                dynamic_geometry,
                &AdvectionComputer { velocity: self },
                cell_index,
                boundary_conditions,
            );
        }
        advected
    }

    pub fn column_average(&self) -> HorizVectorField {
        HorizVectorField {
            cell_footprints: self
                .cells
                .mapv(|velocity| na::Vector2::new(velocity.x, velocity.y))
                .sum_axis(nd::Axis(2))
                / self.cells.shape()[2] as Float,
        }
    }
}
impl std::ops::Div<Float> for VectorField {
    type Output = VectorField;

    fn div(self, rhs: Float) -> Self::Output {
        Self {
            cells: self.cells / rhs,
        }
    }
}
impl std::ops::Mul<&VectorField> for Float {
    type Output = VectorField;

    fn mul(self, rhs: &VectorField) -> Self::Output {
        let mut cells = rhs.cells.clone();
        for cell in &mut cells {
            *cell *= self;
        }
        VectorField { cells }
    }
}
impl std::ops::SubAssign<Vector3> for VectorField {
    fn sub_assign(&mut self, rhs: Vector3) {
        for cell_value in &mut self.cells {
            *cell_value -= rhs;
        }
    }
}
impl std::ops::SubAssign<&VectorField> for VectorField {
    fn sub_assign(&mut self, rhs: &VectorField) {
        self.cells -= &rhs.cells;
    }
}
impl std::ops::AddAssign<&VectorField> for VectorField {
    fn add_assign(&mut self, rhs: &VectorField) {
        self.cells += &rhs.cells;
    }
}
#[pymethods]
impl VectorField {
    pub fn values_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray5<Float> {
        let shape = self.cells.shape();
        let mut values = Array5::zeros((shape[0], shape[1], shape[2], shape[3], 3));
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for l in 0..shape[3] {
                        let vector = &self.cells[[i, j, k, l]];
                        values[[i, j, k, l, 0]] = vector.x;
                        values[[i, j, k, l, 1]] = vector.y;
                        values[[i, j, k, l, 2]] = vector.z;
                    }
                }
            }
        }
        values.into_pyarray(py)
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for VectorField {
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

// Gradient
impl CellDifferentialOpComputer<Matrix3> for VectorField {
    fn compute_interior_face_value(
        &self,
        cell: &geom::Cell,
        neighbor_cell: &geom::Cell,
        outward_normal: UnitVector3,
    ) -> Matrix3 {
        0.5 * outward_normal.into_inner()
            * (self.cell_value(cell.index) + self.cell_value(neighbor_cell.index)).transpose()
    }

    fn compute_boundary_face_value(
        &self,
        cell: &geom::Cell,
        block_paired_cell: &geom::Cell,
        outward_normal: UnitVector3,
        _face_centroid: Point3,
        boundary_condition: &BoundaryCondition,
    ) -> Matrix3 {
        match boundary_condition {
            BoundaryCondition::HomDirichlet => Matrix3::zeros(),
            BoundaryCondition::HomNeumann => {
                outward_normal.into_inner()
                    * 0.5
                    * (self.cell_value(cell.index) + self.cell_value(block_paired_cell.index))
                        .transpose()
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorField {
    /// Indexed by [`indexing::CellIndexing`]
    cells: nd::Array4<Matrix3>,
}
impl TensorField {
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

    pub fn divergence(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions,
    ) -> VectorField {
        struct DivergenceComputer<'a> {
            tensor_field: &'a TensorField,
        }

        impl<'a> CellDifferentialOpComputer<Vector3> for DivergenceComputer<'a> {
            fn compute_interior_face_value(
                &self,
                cell: &geom::Cell,
                neighbor_cell: &geom::Cell,
                outward_normal: UnitVector3,
            ) -> Vector3 {
                0.5 * (self.tensor_field.cell_value(cell.index)
                    + self.tensor_field.cell_value(neighbor_cell.index))
                .tr_mul(&outward_normal)
            }

            fn compute_boundary_face_value(
                &self,
                cell: &geom::Cell,
                block_paired_cell: &geom::Cell,
                outward_normal: UnitVector3,
                _face_centroid: Point3,
                boundary_condition: &BoundaryCondition,
            ) -> Vector3 {
                match boundary_condition {
                    BoundaryCondition::HomDirichlet => Vector3::zeros(),
                    BoundaryCondition::HomNeumann => {
                        0.5 * (self.tensor_field.cell_value(cell.index)
                            + self.tensor_field.cell_value(block_paired_cell.index))
                        .tr_mul(&outward_normal)
                    }
                }
            }
        }

        let mut divergence = VectorField::zeros(dynamic_geometry.grid().cell_indexing());
        let divergence_computer = DivergenceComputer { tensor_field: self };
        for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
            *divergence.cell_value_mut(cell_index) = compute_cell_differential(
                dynamic_geometry,
                &divergence_computer,
                cell_index,
                boundary_conditions,
            );
        }
        divergence
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for TensorField {
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

trait CellDifferentialOpComputer<DV: Copy + Default + std::ops::AddAssign>
where
    DV: std::ops::Mul<Float, Output = DV>,
{
    fn compute_interior_face_value(
        &self,
        cell: &geom::Cell,
        neighbor_cell: &geom::Cell,
        outward_normal: UnitVector3,
    ) -> DV;

    fn compute_boundary_face_value(
        &self,
        cell: &geom::Cell,
        block_paired_cell: &geom::Cell,
        outward_normal: UnitVector3,
        face_centroid: Point3,
        boundary_condition: &BoundaryCondition,
    ) -> DV;
}

fn compute_cell_differential<
    DV: Copy + Default + std::fmt::Debug + std::ops::AddAssign,
    CDC: CellDifferentialOpComputer<DV>,
>(
    dynamic_geometry: &geom::DynamicGeometry,
    field: &CDC,
    cell_index: indexing::CellIndex,
    boundary_conditions: &BoundaryConditions,
) -> DV
where
    DV: std::ops::Mul<Float, Output = DV> + std::ops::Div<Float, Output = DV>,
{
    let cell = dynamic_geometry.cell(cell_index);
    let mut face_accumulator = DV::default();

    for face in &cell.faces {
        let compute_boundary_face_value = |boundary_condition| {
            let block_paired_cell = dynamic_geometry.cell(cell_index.flip());
            field.compute_boundary_face_value(
                cell,
                block_paired_cell,
                face.outward_normal(),
                face.centroid(),
                boundary_condition,
            )
        };
        let face_value = match face.neighbor() {
            indexing::CellNeighbor::Cell(neighbor_cell_index) => field.compute_interior_face_value(
                cell,
                dynamic_geometry.cell(neighbor_cell_index),
                face.outward_normal(),
            ),
            indexing::CellNeighbor::XBoundary(boundary) => match boundary {
                indexing::Boundary::Lower => {
                    compute_boundary_face_value(&boundary_conditions.horiz.x.lower)
                }
                indexing::Boundary::Upper => {
                    compute_boundary_face_value(&boundary_conditions.horiz.x.upper)
                }
            },
            indexing::CellNeighbor::YBoundary(boundary) => match boundary {
                indexing::Boundary::Lower => {
                    compute_boundary_face_value(&boundary_conditions.horiz.y.lower)
                }
                indexing::Boundary::Upper => {
                    compute_boundary_face_value(&boundary_conditions.horiz.y.upper)
                }
            },
            indexing::CellNeighbor::ZBoundary(boundary) => match boundary {
                indexing::Boundary::Lower => {
                    compute_boundary_face_value(&boundary_conditions.z.lower)
                }
                indexing::Boundary::Upper => {
                    compute_boundary_face_value(&boundary_conditions.z.upper)
                }
            },
        };
        face_accumulator += face_value * face.area();
    }
    face_accumulator / cell.volume
}

#[derive(Clone, PartialEq, Debug)]
pub struct HorizScalarField {
    /// Indexed by [`indexing::CellFootprintIndexing`]
    cell_footprints: Array3,
}
impl HorizScalarField {
    pub fn zeros(cell_footprint_indexing: &indexing::CellFootprintIndexing) -> Self {
        Self {
            cell_footprints: Array3::zeros(cell_footprint_indexing.shape()),
        }
    }

    pub fn new<F: Fn(Float, Float) -> Float>(grid: &geom::Grid, f: F) -> Self {
        Self {
            cell_footprints: grid.make_cell_footprint_array(f),
        }
    }

    pub fn cell_footprint_value(
        &self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> Float {
        self.cell_footprints[cell_footprint_index.to_array_index()]
    }

    pub fn cell_footprint_value_mut(
        &mut self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> &mut Float {
        &mut self.cell_footprints[cell_footprint_index.to_array_index()]
    }

    // TODO: Remove otherwise what's the point of this wrapper struct...
    pub fn centers(&self) -> &Array3 {
        &self.cell_footprints
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for HorizScalarField {
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.cell_footprints
            .abs_diff_eq(&other.cell_footprints, epsilon)
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
    pub fn new<F: Fn(Float, Float) -> Float>(grid: &geom::Grid, f: F) -> Self {
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

pub struct HorizVectorField {
    cell_footprints: nd::Array3<Vector2>,
}
impl HorizVectorField {
    pub fn zeros(cell_footprint_indexing: &indexing::CellFootprintIndexing) -> Self {
        Self {
            cell_footprints: nd::Array3::default(cell_footprint_indexing.shape()),
        }
    }

    pub fn cell_footprint_value(
        &self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> Vector2 {
        self.cell_footprints[cell_footprint_index.to_array_index()]
    }

    pub fn cell_footprint_value_mut(
        &mut self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> &mut Vector2 {
        &mut self.cell_footprints[cell_footprint_index.to_array_index()]
    }

    pub fn advect_upwind(
        &self,
        scalar_field: &HorizScalarField,
        grid: &geom::Grid,
        velocity_boundary_conditions: HorizBoundaryConditions,
        scalar_boundary_conditions: HorizBoundaryConditions,
    ) -> HorizScalarField {
        let cell_footprint_indexing = grid.cell_footprint_indexing();
        let mut advected = HorizScalarField::zeros(cell_footprint_indexing);
        for cell_footprint_index in cell_footprint_indexing.iter() {
            let velocity = self.cell_footprint_value(cell_footprint_index);
            let scalar = scalar_field.cell_footprint_value(cell_footprint_index);

            for cell_footprint_pair in
                cell_footprint_indexing.compute_footprint_pairs(cell_footprint_index)
            {
                let cell_footprint_edge = grid.compute_cell_footprint_edge(cell_footprint_pair);
                let outward_normal = cell_footprint_edge.outward_normal;

                let edge_velocity = match cell_footprint_pair.neighbor {
                    indexing::CellFootprintNeighbor::CellFootprint(
                        neighbor_cell_footprint_index,
                    ) => {
                        0.5 * (velocity + self.cell_footprint_value(neighbor_cell_footprint_index))
                    }
                    indexing::CellFootprintNeighbor::XBoundary(boundary) => match boundary {
                        indexing::Boundary::Lower => match velocity_boundary_conditions.x.lower {
                            BoundaryCondition::HomDirichlet => Vector2::zeros(),
                            BoundaryCondition::HomNeumann => velocity,
                        },
                        indexing::Boundary::Upper => match velocity_boundary_conditions.x.upper {
                            BoundaryCondition::HomDirichlet => Vector2::zeros(),
                            BoundaryCondition::HomNeumann => velocity,
                        },
                    },
                    indexing::CellFootprintNeighbor::YBoundary(boundary) => match boundary {
                        indexing::Boundary::Lower => match velocity_boundary_conditions.y.lower {
                            BoundaryCondition::HomDirichlet => Vector2::zeros(),
                            BoundaryCondition::HomNeumann => velocity,
                        },
                        indexing::Boundary::Upper => match velocity_boundary_conditions.y.upper {
                            BoundaryCondition::HomDirichlet => Vector2::zeros(),
                            BoundaryCondition::HomNeumann => velocity,
                        },
                    },
                };

                let projected_edge_velocity = outward_normal.dot(&edge_velocity);
                let neighbor_is_upwind = projected_edge_velocity < 0.;
                let upwind_height_to_advect = if neighbor_is_upwind {
                    match cell_footprint_pair.neighbor {
                        indexing::CellFootprintNeighbor::CellFootprint(
                            neighbor_cell_footprint_index,
                        ) => scalar_field.cell_footprint_value(neighbor_cell_footprint_index),
                        indexing::CellFootprintNeighbor::XBoundary(boundary) => match boundary {
                            indexing::Boundary::Lower => match scalar_boundary_conditions.x.lower {
                                BoundaryCondition::HomDirichlet => 0.,
                                BoundaryCondition::HomNeumann => scalar,
                            },
                            indexing::Boundary::Upper => match scalar_boundary_conditions.x.upper {
                                BoundaryCondition::HomDirichlet => 0.,
                                BoundaryCondition::HomNeumann => scalar,
                            },
                        },
                        indexing::CellFootprintNeighbor::YBoundary(boundary) => match boundary {
                            indexing::Boundary::Lower => match scalar_boundary_conditions.y.lower {
                                BoundaryCondition::HomDirichlet => 0.,
                                BoundaryCondition::HomNeumann => scalar,
                            },
                            indexing::Boundary::Upper => match scalar_boundary_conditions.y.upper {
                                BoundaryCondition::HomDirichlet => 0.,
                                BoundaryCondition::HomNeumann => scalar,
                            },
                        },
                    }
                } else {
                    scalar
                };

                *advected.cell_footprint_value_mut(cell_footprint_index) -=
                    upwind_height_to_advect * cell_footprint_edge.length * projected_edge_velocity;
            }
            *advected.cell_footprint_value_mut(cell_footprint_index) /= grid.footprint_area()
        }

        advected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_advection_flat_geometry() {
        let dynamic_geometry = make_ramp_geometry(false);
        let cell_indexing = dynamic_geometry.grid().cell_indexing();

        // Zero velocity.
        {
            let boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions::hom_neumann(),
                z: BoundaryConditionPair::hom_neumann(),
            };
            let velocity = VectorField::zeros(cell_indexing);

            let advection = velocity.advect_upwind(&dynamic_geometry, &boundary_conditions);

            approx::assert_abs_diff_eq!(advection, velocity, epsilon = 1e-5);
        }

        // Varying velocity, homogeneous Dirichlet boundary conditions.
        {
            let boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions {
                    x: BoundaryConditionPair::hom_dirichlet(),
                    y: BoundaryConditionPair::hom_neumann(),
                },
                z: BoundaryConditionPair::hom_neumann(),
            };
            let velocity =
                VectorField::new(&dynamic_geometry, |x, _, _| Vector3::new(bump(x), 0., 0.));

            let advection = velocity.advect_upwind(&dynamic_geometry, &boundary_conditions);

            let expected_advection = {
                let mut outer_product = TensorField::zeros(cell_indexing);
                for cell_index in cell_indexing.iter() {
                    let velocity_value = velocity.cell_value(cell_index);
                    *outer_product.cell_value_mut(cell_index) =
                        velocity_value * velocity_value.transpose();
                }
                outer_product.divergence(&dynamic_geometry, &boundary_conditions)
            };
            for cell_index in cell_indexing.iter() {
                approx::assert_relative_eq!(
                    advection.cell_value(cell_index),
                    expected_advection.cell_value(cell_index),
                    max_relative = 0.5,
                    epsilon = 2e-2,
                );
            }
        }
    }

    #[test]
    fn test_compute_shear_laplacian_flat_geometry() {
        let dynamic_geometry = make_flat_geometry(7.3, 200, 200, 3);
        let cell_indexing = dynamic_geometry.grid().cell_indexing();

        // Zero velocity.
        {
            let boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions::hom_neumann(),
                z: BoundaryConditionPair::hom_neumann(),
            };
            let velocity = VectorField::zeros(cell_indexing);

            let shear = velocity.compute_gradient(&dynamic_geometry, &boundary_conditions);
            let laplacian =
                velocity.compute_laplacian(&dynamic_geometry, &shear, &boundary_conditions);

            approx::assert_abs_diff_eq!(shear, TensorField::zeros(cell_indexing), epsilon = 1e-5);
            approx::assert_abs_diff_eq!(
                laplacian,
                VectorField::zeros(cell_indexing),
                epsilon = 1e-5
            );
        }

        // Constant velocity.
        {
            let boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions::hom_neumann(),
                z: BoundaryConditionPair::hom_neumann(),
            };
            let velocity = VectorField::new(&dynamic_geometry, |_, _, _| Vector3::new(2., -1., 7.));

            let shear = velocity.compute_gradient(&dynamic_geometry, &boundary_conditions);
            let laplacian =
                velocity.compute_laplacian(&dynamic_geometry, &shear, &boundary_conditions);

            approx::assert_abs_diff_eq!(shear, TensorField::zeros(cell_indexing), epsilon = 1e-5);
            approx::assert_abs_diff_eq!(
                laplacian,
                VectorField::zeros(cell_indexing),
                epsilon = 1e-5
            );
        }

        // Varying velocity with homogeneous Dirichlet boundary conditions.
        {
            let boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions::hom_dirichlet(),
                z: BoundaryConditionPair::hom_neumann(),
            };
            let velocity = VectorField::new(&dynamic_geometry, |x, y, _| {
                Vector3::new(bump(x) * bump(y), -2. * bump(x) * bump(y), 0.)
            });

            let shear = velocity.compute_gradient(&dynamic_geometry, &boundary_conditions);
            let laplacian =
                velocity.compute_laplacian(&dynamic_geometry, &shear, &boundary_conditions);

            let expected_shear = TensorField::new(&dynamic_geometry, |x, y, _| {
                Matrix3::new(
                    bump_deriv(x) * bump(y),
                    -2. * bump_deriv(x) * bump(y),
                    0.,
                    bump(x) * bump_deriv(y),
                    -2. * bump(x) * bump_deriv(y),
                    0.,
                    0.,
                    0.,
                    0.,
                )
            });
            let expected_laplacian = VectorField::new(&dynamic_geometry, |x, y, _| {
                Vector3::new(
                    bump_2nd_deriv(x) * bump(y) + bump(x) * bump_2nd_deriv(y),
                    -2. * bump(x) * bump_2nd_deriv(y) - 2. * bump_2nd_deriv(x) * bump(y),
                    0.,
                )
            });
            for cell_index in cell_indexing.iter() {
                let shear_value = shear.cell_value(cell_index);
                let expected_shear_value = expected_shear.cell_value(cell_index);
                if !approx::relative_eq!(
                    shear_value,
                    expected_shear_value,
                    max_relative = 0.1,
                    epsilon = 0.1
                ) {
                    panic!(
                        "At cell index {cell_index:?}: shear:{shear_value}expected:
                        {expected_shear_value}"
                    );
                }

                let laplacian_value = laplacian.cell_value(cell_index);
                let expected_laplacian_value = expected_laplacian.cell_value(cell_index);
                match dynamic_geometry
                    .grid()
                    .cell_indexing()
                    .classify_cell(cell_index)
                {
                    indexing::CellClassification::Interior => {
                        if !approx::relative_eq!(
                            laplacian_value,
                            expected_laplacian_value,
                            max_relative = 0.2
                        ) {
                            panic!(
                                "At cell index {cell_index:?}: \
                                 laplacian:{laplacian_value}expected:
                                {expected_laplacian_value}"
                            );
                        }
                    }
                    _ => {
                        if !approx::relative_eq!(
                            laplacian_value,
                            expected_laplacian_value,
                            max_relative = 1.
                        ) {
                            println!(
                                "At cell index {cell_index:?}: \
                                 laplacian:{laplacian_value}expected:
                                {expected_laplacian_value}"
                            );
                        }
                    }
                }
            }
        }

        // Varying velocity with homogeneous Neumann boundary conditions.
        {
            let boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions::hom_neumann(),
                z: BoundaryConditionPair::hom_neumann(),
            };

            let velocity = VectorField::new(&dynamic_geometry, |x, y, _| {
                Vector3::new(hill(x), 1.5 * hill(y), 0.)
            });
            let shear = velocity.compute_gradient(&dynamic_geometry, &boundary_conditions);
            let laplacian =
                velocity.compute_laplacian(&dynamic_geometry, &shear, &boundary_conditions);

            let expected_shear = TensorField::new(&dynamic_geometry, |x, y, _| {
                Matrix3::new(
                    hill_deriv(x),
                    0.,
                    0.,
                    0.,
                    1.5 * hill_deriv(y),
                    0.,
                    0.,
                    0.,
                    0.,
                )
            });
            let expected_laplacian = VectorField::new(&dynamic_geometry, |x, y, _| {
                Vector3::new(hill_2nd_deriv(x), 1.5 * hill_2nd_deriv(y), 0.)
            });
            for cell_index in cell_indexing.iter() {
                let shear_value = shear.cell_value(cell_index);
                let expected_shear_value = expected_shear.cell_value(cell_index);
                if !approx::relative_eq!(shear_value, expected_shear_value, epsilon = 0.15) {
                    panic!(
                        "At cell index {cell_index:?}:\n\nshear:\n{shear_value}\n\nexpected:
                         {expected_shear_value}",
                    );
                }

                let laplacian_value = laplacian.cell_value(cell_index);
                let expected_laplacian_value = expected_laplacian.cell_value(cell_index);
                match dynamic_geometry
                    .grid()
                    .cell_indexing()
                    .classify_cell(cell_index)
                {
                    indexing::CellClassification::Interior => {
                        if !approx::relative_eq!(
                            laplacian_value,
                            expected_laplacian_value,
                            max_relative = 1e-1,
                            epsilon = 2e-1
                        ) {
                            panic!(
                                "At cell index \
                                 {cell_index:?}:\n\nlaplacian:\n{laplacian_value}\n\nexpected:
                                {expected_laplacian_value}"
                            );
                        }
                    }
                    _ => {
                        if !approx::relative_eq!(
                            laplacian_value,
                            expected_laplacian_value,
                            max_relative = 1e-1,
                            epsilon = 2e-1
                        ) {
                            panic!(
                                "At cell index \
                                 {cell_index:?}:\n\nlaplacian:\n{laplacian_value}\n\nexpected:
                                 {expected_laplacian_value}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn bump(x: Float) -> Float {
        x * (1. - x) / (0.5 as Float).powi(2)
    }
    fn bump_deriv(x: Float) -> Float {
        (1. - 2. * x) / (0.5 as Float).powi(2)
    }
    fn bump_2nd_deriv(_x: Float) -> Float {
        -2. / (0.5 as Float).powi(2)
    }

    fn hill(x: Float) -> Float {
        x.powi(2) * (1. - x).powi(2) / (0.5 as Float).powi(4)
    }
    fn hill_deriv(x: Float) -> Float {
        (2. * x * (1. - x).powi(2) - x.powi(2) * 2. * (1. - x)) / (0.5 as Float).powi(4)
    }
    fn hill_2nd_deriv(x: Float) -> Float {
        (2. * (1. - x).powi(2) - 4. * x * (1. - x) - 2. * x * 2. * (1. - x) + x.powi(2) * 2.)
            / (0.5 as Float).powi(4)
    }

    #[test]
    fn test_compute_gradient_flat_geometry() {
        test_compute_gradient_impl(&make_flat_geometry(7.3, 5, 5, 5));
    }

    #[test]
    fn test_compute_gradient_ramp_geometry() {
        test_compute_gradient_impl(&make_ramp_geometry(false));
        test_compute_gradient_impl(&make_ramp_geometry(true));
    }

    fn test_compute_gradient_impl(dynamic_geometry: &geom::DynamicGeometry) {
        // Zero pressure.
        {
            let pressure = ScalarField::zeros(dynamic_geometry.grid().vertex_indexing());
            let gradient = pressure.compute_gradient(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                gradient,
                VectorField::zeros(dynamic_geometry.grid().cell_indexing()),
                epsilon = 1e-5
            );
        }

        // Constant pressure.
        {
            let pressure = ScalarField::new(&dynamic_geometry, |_, _, _| 1.9);
            let gradient = pressure.compute_gradient(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                gradient,
                VectorField::zeros(dynamic_geometry.grid().cell_indexing()),
                epsilon = 1e-5
            );
        }

        // Linearly rising pressure.
        {
            let pressure = ScalarField::new(&dynamic_geometry, |_, _, z| 2. * z);
            let gradient = pressure.compute_gradient(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                gradient,
                VectorField::new(&dynamic_geometry, |_, _, _| Vector3::new(0., 0., 2.)),
                epsilon = 1e-1
            );
        }
    }

    fn make_flat_geometry(
        max_z: Float,
        num_x_cells: usize,
        num_y_cells: usize,
        num_z_cells: usize,
    ) -> geom::DynamicGeometry {
        let grid = geom::Grid::new(
            geom::Axis::new(0., 1., num_x_cells),
            geom::Axis::new(0., 1., num_y_cells),
            num_z_cells,
        );
        let height = HorizScalarField::new(&grid, |_, _| max_z);

        let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);
        geom::DynamicGeometry::new(static_geometry, &height)
    }

    fn make_ramp_geometry(swap_xy: bool) -> geom::DynamicGeometry {
        let mut x_axis = geom::Axis::new(0., 1., 300);
        let mut y_axis = geom::Axis::new(0., 0.001, 1);
        if swap_xy {
            (x_axis, y_axis) = (y_axis, x_axis);
        }
        let grid = geom::Grid::new(x_axis, y_axis, 100);
        let height = HorizScalarField::new(&grid, |x, y| 7.3 * x + y + 1.);

        let static_geometry = geom::StaticGeometry::new(grid, &|x, y| 0.1 * (x + y));
        geom::DynamicGeometry::new(static_geometry, &height)
    }
}
