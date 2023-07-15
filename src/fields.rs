use nalgebra as na;
use ndarray::{self as nd};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::{
    derivs, geom,
    indexing::{self, Index, Indexing, IntoIndexIterator},
    Array1, Array2, Float, Matrix3, Point3, UnitVector3, Vector2, Vector3,
};

#[derive(Clone, Debug)]
pub struct BoundaryConditions<V: Value> {
    pub horiz: HorizBoundaryConditions<V>,
    pub z: VertBoundaryFieldPair<V>,
}
#[derive(Clone, Copy, Debug)]
pub struct HorizBoundaryConditions<V: Value> {
    pub x: HorizBoundaryConditionPair<V>,
    pub y: HorizBoundaryConditionPair<V>,
}
impl<V: Value> HorizBoundaryConditions<V> {
    pub fn hom_neumann() -> Self {
        Self {
            x: HorizBoundaryConditionPair::hom_neumann(),
            y: HorizBoundaryConditionPair::hom_neumann(),
        }
    }

    pub fn hom_dirichlet() -> Self {
        Self {
            x: HorizBoundaryConditionPair::hom_dirichlet(),
            y: HorizBoundaryConditionPair::hom_dirichlet(),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct HorizBoundaryConditionPair<V: Value> {
    pub lower: BoundaryCondition<V>,
    pub upper: BoundaryCondition<V>,
}
impl<V: Value> HorizBoundaryConditionPair<V> {
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
#[derive(Clone, Debug)]
pub struct VertBoundaryFieldPair<V: Value> {
    pub lower: VertBoundaryField<V>,
    pub upper: VertBoundaryField<V>,
}
impl<V: Value> VertBoundaryFieldPair<V> {
    pub fn hom_neumann() -> Self {
        Self {
            lower: VertBoundaryField::HomNeumann,
            upper: VertBoundaryField::HomNeumann,
        }
    }

    pub fn hom_dirichlet() -> Self {
        Self {
            lower: VertBoundaryField::HomDirichlet,
            upper: VertBoundaryField::HomDirichlet,
        }
    }
}
#[derive(Clone, Debug)]
pub enum VertBoundaryField<V: Value> {
    HomDirichlet,
    HomNeumann,
    InhomNeumann(AreaField<V>),
    Kinematic(AreaScalarField),
}
impl<V: Value> VertBoundaryField<V> {
    pub fn boundary_condition(
        &self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> BoundaryCondition<V> {
        match self {
            VertBoundaryField::HomDirichlet => BoundaryCondition::HomDirichlet,
            VertBoundaryField::HomNeumann => BoundaryCondition::HomNeumann,
            VertBoundaryField::InhomNeumann(horiz_field) => BoundaryCondition::InhomNeumann(
                horiz_field.cell_footprint_value(cell_footprint_index),
            ),
            VertBoundaryField::Kinematic(dhdt) => {
                BoundaryCondition::Kinematic(dhdt.cell_footprint_value(cell_footprint_index))
            }
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum BoundaryCondition<V: Value> {
    HomDirichlet,
    HomNeumann,
    InhomNeumann(V),
    Kinematic(Float),
}

pub trait Value:
    num_traits::identities::Zero
    + Copy
    + Clone
    + std::fmt::Debug
    + std::ops::Div<Float, Output = Self>
    + std::ops::Mul<Float, Output = Self>
    + std::ops::DivAssign<Float>
    + std::ops::MulAssign<Float>
    + std::ops::Add<Self, Output = Self>
    + std::ops::Sub<Self, Output = Self>
    + std::ops::AddAssign<Self>
    + std::ops::SubAssign<Self>
    + std::ops::Neg<Output = Self>
{
    fn size() -> usize;

    fn flatten(&self, flattened: nd::ArrayViewMut1<'_, Float>);
    fn unflatten(flattened: nd::ArrayView1<'_, Float>) -> Self;
}
impl Value for Float {
    fn size() -> usize {
        1
    }

    fn flatten(&self, mut flattened: nd::ArrayViewMut1<'_, Float>) {
        flattened[0] = *self;
    }

    fn unflatten(flattened: nd::ArrayView1<'_, Float>) -> Self {
        flattened[0]
    }
}
impl Value for Vector2 {
    fn size() -> usize {
        2
    }

    fn flatten(&self, mut flattened: nd::ArrayViewMut1<'_, Float>) {
        flattened[0] = self.x;
        flattened[1] = self.y;
    }

    fn unflatten(flattened: nd::ArrayView1<'_, Float>) -> Self {
        Self::new(flattened[0], flattened[1])
    }
}
impl Value for Vector3 {
    fn size() -> usize {
        3
    }

    fn flatten(&self, mut flattened: nd::ArrayViewMut1<'_, Float>) {
        flattened[0] = self.x;
        flattened[1] = self.y;
        flattened[2] = self.z;
    }

    fn unflatten(flattened: nd::ArrayView1<'_, Float>) -> Self {
        Self::new(flattened[0], flattened[1], flattened[2])
    }
}
impl Value for Matrix3 {
    fn size() -> usize {
        9
    }

    fn flatten(&self, mut flattened: nd::ArrayViewMut1<'_, Float>) {
        flattened[0] = self[(0, 0)];
        flattened[1] = self[(0, 1)];
        flattened[2] = self[(0, 2)];
        flattened[3] = self[(1, 0)];
        flattened[4] = self[(1, 1)];
        flattened[5] = self[(1, 2)];
        flattened[6] = self[(2, 0)];
        flattened[7] = self[(2, 1)];
        flattened[8] = self[(2, 2)];
    }

    fn unflatten(flattened: nd::ArrayView1<'_, Float>) -> Self {
        Self::new(
            flattened[0],
            flattened[1],
            flattened[2],
            flattened[3],
            flattened[4],
            flattened[5],
            flattened[6],
            flattened[7],
            flattened[8],
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VolField<V: Value> {
    /// Indexed by [`indexing::CellIndexing`]
    cells: nd::Array4<V>,
}
impl<V: Value> VolField<V> {
    pub fn new<F: Fn(Float, Float, Float) -> V>(
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

    pub fn cell_value(&self, cell_index: indexing::CellIndex) -> V {
        self.cells[cell_index.to_array_index()]
    }

    pub fn cell_value_mut(&mut self, cell_index: indexing::CellIndex) -> &mut V {
        &mut self.cells[cell_index.to_array_index()]
    }

    pub fn map<OV: Value, F: Fn(&V) -> OV>(&self, f: F) -> VolField<OV> {
        VolField {
            cells: self.cells.map(f),
        }
    }

    pub fn flatten(&self, cell_indexing: &indexing::CellIndexing) -> Array1 {
        let mut flattened = Array1::zeros(cell_indexing.len() * V::size());
        for (flat_index, cell_index) in cell_indexing.iter().enumerate() {
            self.cells[cell_index.to_array_index()].flatten(
                flattened.slice_mut(nd::s![flat_index * V::size()..(flat_index + 1) * V::size()]),
            );
        }
        flattened
    }

    pub fn unflatten(cell_indexing: &indexing::CellIndexing, flattened: &Array1) -> Self {
        let mut cells = nd::Array4::zeros(cell_indexing.shape());
        for (flat_index, cell_index) in cell_indexing.iter().enumerate() {
            cells[cell_index.to_array_index()] = V::unflatten(
                flattened.slice(nd::s![flat_index * V::size()..(flat_index + 1) * V::size()]),
            );
        }
        Self { cells }
    }
}
impl<V: Value> std::ops::Div<Float> for VolField<V> {
    type Output = VolField<V>;

    fn div(self, rhs: Float) -> Self::Output {
        Self {
            cells: self.cells / rhs,
        }
    }
}
impl<V: Value> std::ops::Div<Float> for &VolField<V> {
    type Output = VolField<V>;

    fn div(self, rhs: Float) -> Self::Output {
        VolField {
            cells: &self.cells / rhs,
        }
    }
}
impl<V: Value> std::ops::Mul<&VolField<V>> for Float {
    type Output = VolField<V>;

    fn mul(self, rhs: &VolField<V>) -> Self::Output {
        let mut cells = rhs.cells.clone();
        for cell in &mut cells {
            *cell *= self;
        }
        VolField { cells }
    }
}
impl<V: Value> std::ops::SubAssign<V> for VolField<V> {
    fn sub_assign(&mut self, rhs: V) {
        for cell_value in &mut self.cells {
            *cell_value -= rhs;
        }
    }
}
impl<V: Value> std::ops::AddAssign<&VolField<V>> for VolField<V> {
    fn add_assign(&mut self, rhs: &VolField<V>) {
        self.cells += &rhs.cells;
    }
}
impl<V: Value> std::ops::SubAssign<&VolField<V>> for VolField<V> {
    fn sub_assign(&mut self, rhs: &VolField<V>) {
        self.cells -= &rhs.cells;
    }
}
impl<V: Value> std::ops::MulAssign<Float> for VolField<V> {
    fn mul_assign(&mut self, rhs: Float) {
        for cell_value in &mut self.cells {
            *cell_value *= rhs;
        }
    }
}
impl<V: Value> std::ops::Neg for VolField<V> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { cells: -self.cells }
    }
}
#[cfg(test)]
impl<V: Value> approx::AbsDiffEq for VolField<V>
where
    V: approx::AbsDiffEq<V, Epsilon = Float>,
{
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        <nd::Array4<V> as approx::AbsDiffEq<_>>::abs_diff_eq(&self.cells, &other.cells, epsilon)
    }
}

pub type VolScalarField = VolField<Float>;
impl VolScalarField {
    pub fn gradient(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions<Float>,
    ) -> VolVectorField {
        struct GradientComputer<'a> {
            scalar_field: &'a VolScalarField,
        }
        impl derivs::DifferentialOpComputer<Float, Vector3> for GradientComputer<'_> {
            fn compute_interior_face_value(
                &self,
                cell: &geom::Cell,
                neighbor_cell: &geom::Cell,
                outward_normal: UnitVector3,
            ) -> Vector3 {
                0.5 * outward_normal.into_inner()
                    * (self.scalar_field.cell_value(cell.index)
                        + self.scalar_field.cell_value(neighbor_cell.index))
            }

            fn compute_boundary_face_value(
                &self,
                cell: &geom::Cell,
                block_paired_cell: &geom::Cell,
                outward_normal: UnitVector3,
                face_centroid: Point3,
                boundary_condition: BoundaryCondition<Float>,
            ) -> Vector3 {
                match boundary_condition {
                    BoundaryCondition::HomDirichlet => Vector3::zeros(),
                    BoundaryCondition::HomNeumann => {
                        0.5 * outward_normal.into_inner()
                            * (self.scalar_field.cell_value(cell.index)
                                + self.scalar_field.cell_value(block_paired_cell.index))
                    }
                    BoundaryCondition::InhomNeumann(boundary_deriv) => {
                        let displ = face_centroid.coords
                            - 0.5 * (cell.centroid.coords + block_paired_cell.centroid.coords);
                        outward_normal.into_inner()
                            * (0.5
                                * (self.scalar_field.cell_value(cell.index)
                                    + self.scalar_field.cell_value(block_paired_cell.index))
                                + displ.norm() * boundary_deriv)
                    }
                    BoundaryCondition::Kinematic(_) => unimplemented!(),
                }
            }
        }
        derivs::compute_field_differential(
            dynamic_geometry,
            boundary_conditions,
            GradientComputer { scalar_field: self },
        )
    }

    pub fn values_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray4<Float> {
        self.cells.clone().into_pyarray(py)
    }
}

pub type VolVectorField = VolField<Vector3>;
impl VolVectorField {
    pub fn gradient(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions<Vector3>,
    ) -> VolTensorField {
        struct GradientComputer<'a> {
            vector_field: &'a VolVectorField,
        }
        impl derivs::DifferentialOpComputer<Vector3, Matrix3> for GradientComputer<'_> {
            fn compute_interior_face_value(
                &self,
                cell: &geom::Cell,
                neighbor_cell: &geom::Cell,
                outward_normal: UnitVector3,
            ) -> Matrix3 {
                0.5 * outward_normal.into_inner()
                    * (self.vector_field.cell_value(cell.index)
                        + self.vector_field.cell_value(neighbor_cell.index))
                    .transpose()
            }

            fn compute_boundary_face_value(
                &self,
                cell: &geom::Cell,
                block_paired_cell: &geom::Cell,
                outward_normal: UnitVector3,
                _face_centroid: Point3,
                boundary_condition: BoundaryCondition<Vector3>,
            ) -> Matrix3 {
                match boundary_condition {
                    BoundaryCondition::HomDirichlet => Matrix3::zeros(),
                    BoundaryCondition::HomNeumann => {
                        outward_normal.into_inner()
                            * 0.5
                            * (self.vector_field.cell_value(cell.index)
                                + self.vector_field.cell_value(block_paired_cell.index))
                            .transpose()
                    }
                    BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                    BoundaryCondition::Kinematic(dhdt) => {
                        let cell_value = self.vector_field.cell_value(cell.index);
                        let face_value =
                            compute_kinematic_face_velocity_value(cell_value, dhdt, outward_normal);
                        outward_normal.into_inner() * face_value.transpose()
                    }
                }
            }
        }

        derivs::compute_field_differential(
            dynamic_geometry,
            boundary_conditions,
            GradientComputer { vector_field: self },
        )
    }

    pub fn laplacian(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        gradient: &VolTensorField,
        boundary_conditions: &BoundaryConditions<Vector3>,
    ) -> Self {
        struct LaplacianComputer<'a> {
            vector_field: &'a VolVectorField,
            gradient_field: &'a VolTensorField,
        }

        impl derivs::DifferentialOpComputer<Vector3, Vector3> for LaplacianComputer<'_> {
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
                outward_normal: UnitVector3,
                face_centroid: Point3,
                boundary_condition: BoundaryCondition<Vector3>,
            ) -> Vector3 {
                match boundary_condition {
                    BoundaryCondition::HomDirichlet => {
                        // TODO: 0.25?
                        let cell_value = 0.5
                            * (self.vector_field.cell_value(cell.index)
                                + self.vector_field.cell_value(block_paired_cell.index));
                        -cell_value
                            / (face_centroid.coords
                                - 0.5 * (cell.centroid.coords + block_paired_cell.centroid.coords))
                                .norm()
                    }
                    BoundaryCondition::HomNeumann => Vector3::zeros(),
                    BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                    BoundaryCondition::Kinematic(dhdt) => {
                        let cell_value = self.vector_field.cell_value(cell.index);
                        let face_value =
                            compute_kinematic_face_velocity_value(cell_value, dhdt, outward_normal);
                        (face_value - cell_value)
                            / (face_centroid.coords
                                - 0.5 * (cell.centroid.coords + block_paired_cell.centroid.coords))
                                .norm()
                    }
                }
            }
        }

        derivs::compute_field_differential(
            dynamic_geometry,
            boundary_conditions,
            LaplacianComputer {
                vector_field: self,
                gradient_field: gradient,
            },
        )
    }

    pub fn advect_upwind(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions<Vector3>,
    ) -> Self {
        struct AdvectionComputer<'a> {
            velocity: &'a VolVectorField,
        }
        impl derivs::DifferentialOpComputer<Vector3, Vector3> for AdvectionComputer<'_> {
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
                boundary_condition: BoundaryCondition<Vector3>,
            ) -> Vector3 {
                let face_velocity = match boundary_condition {
                    BoundaryCondition::HomDirichlet => return Vector3::zeros(),
                    BoundaryCondition::HomNeumann => {
                        0.5 * (self.velocity.cell_value(cell.index)
                            + self.velocity.cell_value(block_paired_cell.index))
                    }
                    BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                    BoundaryCondition::Kinematic(dhdt) => {
                        let cell_value = self.velocity.cell_value(cell.index);
                        compute_kinematic_face_velocity_value(cell_value, dhdt, outward_normal)
                    }
                };

                face_velocity.dot(&outward_normal) * face_velocity
            }
        }

        derivs::compute_field_differential(
            dynamic_geometry,
            boundary_conditions,
            AdvectionComputer { velocity: self },
        )
    }

    pub fn column_average(&self) -> AreaVectorField {
        AreaVectorField {
            cell_footprints: self
                .cells
                .mapv(|velocity| na::Vector2::new(velocity.x, velocity.y))
                .sum_axis(nd::Axis(2))
                / self.cells.shape()[2] as Float,
        }
    }

    pub fn values_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray5<Float> {
        let shape = self.cells.shape();
        let mut cells = nd::Array5::zeros([shape[0], shape[1], shape[2], shape[3], 3]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    for l in 0..2 {
                        cells[[i, j, k, l, 0]] = self.cells[[i, j, k, l]].x;
                        cells[[i, j, k, l, 1]] = self.cells[[i, j, k, l]].y;
                        cells[[i, j, k, l, 2]] = self.cells[[i, j, k, l]].z;
                    }
                }
            }
        }
        cells.into_pyarray(py)
    }
}

pub type VolTensorField = VolField<Matrix3>;
impl VolTensorField {
    pub fn divergence(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions<Matrix3>,
    ) -> VolVectorField {
        struct DivergenceComputer<'a> {
            tensor_field: &'a VolTensorField,
        }

        impl derivs::DifferentialOpComputer<Matrix3, Vector3> for DivergenceComputer<'_> {
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
                boundary_condition: BoundaryCondition<Matrix3>,
            ) -> Vector3 {
                match boundary_condition {
                    BoundaryCondition::HomDirichlet => Vector3::zeros(),
                    BoundaryCondition::HomNeumann => {
                        0.5 * (self.tensor_field.cell_value(cell.index)
                            + self.tensor_field.cell_value(block_paired_cell.index))
                        .tr_mul(&outward_normal)
                    }
                    BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                    BoundaryCondition::Kinematic(_) => unimplemented!(),
                }
            }
        }

        derivs::compute_field_differential(
            dynamic_geometry,
            boundary_conditions,
            DivergenceComputer { tensor_field: self },
        )
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct AreaField<V: Value> {
    /// Indexed by [`indexing::CellFootprintIndexing`]
    cell_footprints: nd::Array3<V>,
}
impl<V: Value> AreaField<V> {
    pub fn zeros(cell_footprint_indexing: &indexing::CellFootprintIndexing) -> Self {
        Self {
            cell_footprints: nd::Array3::zeros(cell_footprint_indexing.shape()),
        }
    }

    pub fn new<F: Fn(Float, Float) -> V>(grid: &geom::Grid, f: F) -> Self {
        Self {
            cell_footprints: grid.make_cell_footprint_array(f),
        }
    }

    pub fn cell_footprint_value(&self, cell_footprint_index: indexing::CellFootprintIndex) -> V {
        self.cell_footprints[cell_footprint_index.to_array_index()]
    }

    pub fn cell_footprint_value_mut(
        &mut self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> &mut V {
        &mut self.cell_footprints[cell_footprint_index.to_array_index()]
    }

    // TODO: Remove otherwise what's the point of this wrapper struct...
    pub fn centers(&self) -> &nd::Array3<V> {
        &self.cell_footprints
    }
}
#[cfg(test)]
impl<V: Value> approx::AbsDiffEq for AreaField<V>
where
    V: approx::AbsDiffEq<V, Epsilon = Float>,
{
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        <nd::Array3<V> as approx::AbsDiffEq<_>>::abs_diff_eq(
            &self.cell_footprints,
            &other.cell_footprints,
            epsilon,
        )
    }
}
pub type AreaScalarField = AreaField<Float>;

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

pub struct AreaVectorField {
    cell_footprints: nd::Array3<Vector2>,
}
impl AreaVectorField {
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
        scalar_field: &AreaScalarField,
        grid: &geom::Grid,
        velocity_boundary_conditions: HorizBoundaryConditions<Vector2>,
        scalar_boundary_conditions: HorizBoundaryConditions<Float>,
    ) -> AreaScalarField {
        let cell_footprint_indexing = grid.cell_footprint_indexing();
        let mut advected = AreaScalarField::zeros(cell_footprint_indexing);
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
                            BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                            BoundaryCondition::Kinematic(_) => unimplemented!(),
                        },
                        indexing::Boundary::Upper => match velocity_boundary_conditions.x.upper {
                            BoundaryCondition::HomDirichlet => Vector2::zeros(),
                            BoundaryCondition::HomNeumann => velocity,
                            BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                            BoundaryCondition::Kinematic(_) => unimplemented!(),
                        },
                    },
                    indexing::CellFootprintNeighbor::YBoundary(boundary) => match boundary {
                        indexing::Boundary::Lower => match velocity_boundary_conditions.y.lower {
                            BoundaryCondition::HomDirichlet => Vector2::zeros(),
                            BoundaryCondition::HomNeumann => velocity,
                            BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                            BoundaryCondition::Kinematic(_) => unimplemented!(),
                        },
                        indexing::Boundary::Upper => match velocity_boundary_conditions.y.upper {
                            BoundaryCondition::HomDirichlet => Vector2::zeros(),
                            BoundaryCondition::HomNeumann => velocity,
                            BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                            BoundaryCondition::Kinematic(_) => unimplemented!(),
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
                                BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                                BoundaryCondition::Kinematic(_) => unimplemented!(),
                            },
                            indexing::Boundary::Upper => match scalar_boundary_conditions.x.upper {
                                BoundaryCondition::HomDirichlet => 0.,
                                BoundaryCondition::HomNeumann => scalar,
                                BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                                BoundaryCondition::Kinematic(_) => unimplemented!(),
                            },
                        },
                        indexing::CellFootprintNeighbor::YBoundary(boundary) => match boundary {
                            indexing::Boundary::Lower => match scalar_boundary_conditions.y.lower {
                                BoundaryCondition::HomDirichlet => 0.,
                                BoundaryCondition::HomNeumann => scalar,
                                BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                                BoundaryCondition::Kinematic(_) => unimplemented!(),
                            },
                            indexing::Boundary::Upper => match scalar_boundary_conditions.y.upper {
                                BoundaryCondition::HomDirichlet => 0.,
                                BoundaryCondition::HomNeumann => scalar,
                                BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                                BoundaryCondition::Kinematic(_) => unimplemented!(),
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

/// Compute the face value of the velocity at a boundary specifying the
/// kinematic boundary condition.
fn compute_kinematic_face_velocity_value(
    velocity_cell_value: Vector3,
    dhdt: Float,
    outward_normal: UnitVector3,
) -> Vector3 {
    // u'.n = z.n * dh/dt
    // u' - (u'.n) n = u - (u.n) n
    // u' - (z.n * dh/dt) n = u - (u.n) n
    // u' = u + (z.n * dh/dt - u.n) n
    velocity_cell_value
        + (-outward_normal.dot(&velocity_cell_value) + dhdt * outward_normal.z)
            * outward_normal.into_inner()
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
                z: VertBoundaryFieldPair::hom_neumann(),
            };
            let velocity = VolVectorField::zeros(cell_indexing);

            let advection = velocity.advect_upwind(&dynamic_geometry, &boundary_conditions);

            approx::assert_abs_diff_eq!(advection, velocity, epsilon = 1e-5);
        }

        // Varying velocity, homogeneous Dirichlet boundary conditions.
        {
            let vel_boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions {
                    x: HorizBoundaryConditionPair::hom_dirichlet(),
                    y: HorizBoundaryConditionPair::hom_neumann(),
                },
                z: VertBoundaryFieldPair::hom_neumann(),
            };
            let outer_boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions {
                    x: HorizBoundaryConditionPair::hom_dirichlet(),
                    y: HorizBoundaryConditionPair::hom_neumann(),
                },
                z: VertBoundaryFieldPair::hom_neumann(),
            };
            let velocity =
                VolVectorField::new(&dynamic_geometry, |x, _, _| Vector3::new(bump(x), 0., 0.));

            let advection = velocity.advect_upwind(&dynamic_geometry, &vel_boundary_conditions);

            let expected_advection = {
                let mut outer_product = VolTensorField::zeros(cell_indexing);
                for cell_index in cell_indexing.iter() {
                    let velocity_value = velocity.cell_value(cell_index);
                    *outer_product.cell_value_mut(cell_index) =
                        velocity_value * velocity_value.transpose();
                }
                outer_product.divergence(&dynamic_geometry, &outer_boundary_conditions)
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
                z: VertBoundaryFieldPair::hom_neumann(),
            };
            let velocity = VolVectorField::zeros(cell_indexing);

            let shear = velocity.gradient(&dynamic_geometry, &boundary_conditions);
            let laplacian = velocity.laplacian(&dynamic_geometry, &shear, &boundary_conditions);

            approx::assert_abs_diff_eq!(
                shear,
                VolTensorField::zeros(cell_indexing),
                epsilon = 1e-5
            );
            approx::assert_abs_diff_eq!(
                laplacian,
                VolVectorField::zeros(cell_indexing),
                epsilon = 1e-5
            );
        }

        // Constant velocity.
        {
            let boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions::hom_neumann(),
                z: VertBoundaryFieldPair::hom_neumann(),
            };
            let velocity =
                VolVectorField::new(&dynamic_geometry, |_, _, _| Vector3::new(2., -1., 7.));

            let shear = velocity.gradient(&dynamic_geometry, &boundary_conditions);
            let laplacian = velocity.laplacian(&dynamic_geometry, &shear, &boundary_conditions);

            approx::assert_abs_diff_eq!(
                shear,
                VolTensorField::zeros(cell_indexing),
                epsilon = 1e-5
            );
            approx::assert_abs_diff_eq!(
                laplacian,
                VolVectorField::zeros(cell_indexing),
                epsilon = 1e-5
            );
        }

        // Varying velocity with homogeneous Dirichlet boundary conditions.
        {
            let boundary_conditions = BoundaryConditions {
                horiz: HorizBoundaryConditions::hom_dirichlet(),
                z: VertBoundaryFieldPair::hom_neumann(),
            };
            let velocity = VolVectorField::new(&dynamic_geometry, |x, y, _| {
                Vector3::new(bump(x) * bump(y), -2. * bump(x) * bump(y), 0.)
            });

            let shear = velocity.gradient(&dynamic_geometry, &boundary_conditions);
            let laplacian = velocity.laplacian(&dynamic_geometry, &shear, &boundary_conditions);

            let expected_shear = VolTensorField::new(&dynamic_geometry, |x, y, _| {
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
            let expected_laplacian = VolVectorField::new(&dynamic_geometry, |x, y, _| {
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
                z: VertBoundaryFieldPair::hom_neumann(),
            };

            let velocity = VolVectorField::new(&dynamic_geometry, |x, y, _| {
                Vector3::new(hill(x), 1.5 * hill(y), 0.)
            });
            let shear = velocity.gradient(&dynamic_geometry, &boundary_conditions);
            let laplacian = velocity.laplacian(&dynamic_geometry, &shear, &boundary_conditions);

            let expected_shear = VolTensorField::new(&dynamic_geometry, |x, y, _| {
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
            let expected_laplacian = VolVectorField::new(&dynamic_geometry, |x, y, _| {
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
        let height = AreaScalarField::new(&grid, |_, _| max_z);

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
        let height = AreaScalarField::new(&grid, |x, y| 7.3 * x + y + 1.);

        let static_geometry = geom::StaticGeometry::new(grid, &|x, y| 0.1 * (x + y));
        geom::DynamicGeometry::new(static_geometry, &height)
    }
}
