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

    pub fn no_penetration() -> Self {
        Self {
            lower: BoundaryCondition::NoPenetration,
            upper: BoundaryCondition::NoPenetration,
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
            VertBoundaryField::Kinematic(height_time_deriv) => BoundaryCondition::Kinematic(
                height_time_deriv.cell_footprint_value(cell_footprint_index),
            ),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum BoundaryCondition<V: Value> {
    HomDirichlet,
    HomNeumann,
    InhomNeumann(V),
    Kinematic(Float),
    NoPenetration,
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
impl std::ops::Mul<VolField<Float>> for &VolField<Float> {
    type Output = VolField<Float>;

    fn mul(self, mut rhs: VolField<Float>) -> Self::Output {
        rhs.cells *= &self.cells;
        rhs
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
pub struct AllCloseAssertion<'a, 'b, 'c, V>
where
    V: Value
        + std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>,
{
    left: &'a VolField<V>,
    right: &'b VolField<V>,

    dynamic_geometry: &'c geom::DynamicGeometry,

    rel_tol: Option<Float>,
    abs_tol: Option<Float>,
}
#[cfg(test)]
impl<'a, 'b, 'c, V> AllCloseAssertion<'a, 'b, 'c, V>
where
    V: Value
        + std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>,
{
    pub fn rel_tol(&mut self, rel_tol: Option<Float>) -> &mut Self {
        self.rel_tol = rel_tol;
        self
    }

    pub fn abs_tol(&mut self, abs_tol: Option<Float>) -> &mut Self {
        self.abs_tol = abs_tol;
        self
    }
}

#[cfg(test)]
impl<'a, 'b, 'c, V> Drop for AllCloseAssertion<'a, 'b, 'c, V>
where
    V: Value
        + std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>,
{
    #[track_caller]
    fn drop(&mut self) {
        if self.rel_tol.is_none() && self.abs_tol.is_none() {
            panic!("At least one tolerance must be specified");
        }
        let mut num_failures = 0;
        for cell_index in self.dynamic_geometry.grid().cell_indexing().iter() {
            let mut checker = approx::Relative::default();
            if let Some(rel_tol) = self.rel_tol {
                checker = checker.max_relative(rel_tol);
            }
            if let Some(abs_tol) = self.abs_tol {
                checker = checker.epsilon(abs_tol);
            }
            let left = self.left.cell_value(cell_index);
            let right = self.right.cell_value(cell_index);
            if !checker.eq(&left, &right) {
                if num_failures < 20 {
                    eprintln!("At {cell_index:?}, left = {left}, right = {right}");
                }
                num_failures += 1;
            }
        }
        if num_failures > 0 {
            panic!(
                "Didn't match at {num_failures}/{} cells",
                self.left.cells.len()
            )
        }
    }
}
#[cfg(test)]
impl<
        V: Value
            + std::fmt::Display
            + approx::AbsDiffEq<V, Epsilon = Float>
            + approx::RelativeEq<V, Epsilon = Float>,
    > VolField<V>
{
    #[track_caller]
    pub fn assert_all_close<'a, 'b, 'c>(
        &'a self,
        other: &'b Self,
        dynamic_geometry: &'c geom::DynamicGeometry,
    ) -> AllCloseAssertion<'a, 'b, 'c, V> {
        AllCloseAssertion {
            left: self,
            right: other,
            dynamic_geometry,
            rel_tol: Some(1e-7),
            abs_tol: Some(0.),
        }
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
                face_centroid: Point3,
            ) -> Vector3 {
                outward_normal.into_inner()
                    * linearly_interpolate_to_face(
                        self.scalar_field,
                        &face_centroid,
                        &outward_normal,
                        cell,
                        neighbor_cell,
                    )
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
                    BoundaryCondition::NoPenetration => unimplemented!(),
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
                face_centroid: Point3,
            ) -> Matrix3 {
                outward_normal.into_inner()
                    * linearly_interpolate_to_face(
                        self.vector_field,
                        &face_centroid,
                        &outward_normal,
                        cell,
                        neighbor_cell,
                    )
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
                outward_normal.into_inner()
                    * match boundary_condition {
                        BoundaryCondition::HomDirichlet => Vector3::zeros(),
                        BoundaryCondition::HomNeumann => {
                            0.5 * (self.vector_field.cell_value(cell.index)
                                + self.vector_field.cell_value(block_paired_cell.index))
                        }
                        BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                        BoundaryCondition::Kinematic(height_time_deriv) => {
                            compute_kinematic_face_velocity_value(height_time_deriv)
                        }
                        BoundaryCondition::NoPenetration => {
                            let value = 0.5
                                * (self.vector_field.cell_value(cell.index)
                                    + self.vector_field.cell_value(block_paired_cell.index));
                            value - value.dot(&outward_normal) * outward_normal.into_inner()
                        }
                    }
                    .transpose()
            }
        }

        derivs::compute_field_differential(
            dynamic_geometry,
            boundary_conditions,
            GradientComputer { vector_field: self },
        )
    }

    pub fn divergence(
        &self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions<Vector3>,
    ) -> VolScalarField {
        struct DivergenceComputer<'a> {
            vector_field: &'a VolVectorField,
        }

        impl derivs::DifferentialOpComputer<Vector3, Float> for DivergenceComputer<'_> {
            fn compute_interior_face_value(
                &self,
                cell: &geom::Cell,
                neighbor_cell: &geom::Cell,
                outward_normal: UnitVector3,
                face_centroid: Point3,
            ) -> Float {
                linearly_interpolate_to_face(
                    self.vector_field,
                    &face_centroid,
                    &outward_normal,
                    cell,
                    neighbor_cell,
                )
                .dot(&outward_normal)
            }

            fn compute_boundary_face_value(
                &self,
                cell: &geom::Cell,
                block_paired_cell: &geom::Cell,
                outward_normal: UnitVector3,
                _face_centroid: Point3,
                boundary_condition: BoundaryCondition<Vector3>,
            ) -> Float {
                match boundary_condition {
                    BoundaryCondition::HomDirichlet => Vector3::zeros(),
                    BoundaryCondition::HomNeumann => {
                        0.5 * (self.vector_field.cell_value(cell.index)
                            + self.vector_field.cell_value(block_paired_cell.index))
                    }
                    BoundaryCondition::InhomNeumann(_) => unimplemented!(),
                    BoundaryCondition::Kinematic(height_time_deriv) => {
                        compute_kinematic_face_velocity_value(height_time_deriv)
                    }
                    BoundaryCondition::NoPenetration => Vector3::zeros(),
                }
                .dot(&outward_normal)
            }
        }

        derivs::compute_field_differential(
            dynamic_geometry,
            boundary_conditions,
            DivergenceComputer { vector_field: self },
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
                face_centroid: Point3,
            ) -> Vector3 {
                let explicit_grad_at_face = linearly_interpolate_to_face(
                    self.gradient_field,
                    &face_centroid,
                    &outward_normal,
                    cell,
                    neighbor_cell,
                );

                let displ = neighbor_cell.centroid - cell.centroid;
                let c_corr = 1. / outward_normal.dot(&displ);
                c_corr
                    * (self.vector_field.cell_value(neighbor_cell.index)
                        - self.vector_field.cell_value(cell.index))
                    + explicit_grad_at_face.tr_mul(&(outward_normal.into_inner() - c_corr * displ))
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
                    BoundaryCondition::Kinematic(height_time_deriv) => {
                        let cell_value = self.vector_field.cell_value(cell.index);
                        let face_value = compute_kinematic_face_velocity_value(height_time_deriv);
                        // TODO REVERT?
                        (face_value - cell_value)
                            / (face_centroid.coords - cell.centroid.coords).norm()
                    }
                    BoundaryCondition::NoPenetration => {
                        let cell_value = self.vector_field.cell_value(cell.index);
                        let face_value = cell_value
                            - cell_value.dot(&outward_normal) * outward_normal.into_inner();
                        (face_value - cell_value)
                            / (face_centroid.coords - cell.centroid.coords).norm()
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
        advection_velocity: &Self,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &BoundaryConditions<Vector3>,
    ) -> Self {
        struct AdvectionComputer<'a> {
            velocity: &'a VolVectorField,
            advection_velocity: &'a VolVectorField,
        }
        impl derivs::DifferentialOpComputer<Vector3, Vector3> for AdvectionComputer<'_> {
            fn compute_interior_face_value(
                &self,
                cell: &geom::Cell,
                neighbor_cell: &geom::Cell,
                outward_normal: UnitVector3,
                face_centroid: Point3,
            ) -> Vector3 {
                let (weight, neighbor_weight) = compute_linear_interpolation_weights(
                    &face_centroid,
                    &outward_normal,
                    cell,
                    neighbor_cell,
                );

                let cell_advection_velocity = self.advection_velocity.cell_value(cell.index);
                let neighbor_cell_advection_velocity =
                    self.advection_velocity.cell_value(neighbor_cell.index);
                let linear_face_advection_velocity = weight * cell_advection_velocity
                    + neighbor_weight * neighbor_cell_advection_velocity;

                let projected_face_advection_velocity =
                    linear_face_advection_velocity.dot(&outward_normal);
                let neighbor_is_upwind = projected_face_advection_velocity < 0.;
                let upwind_face_advection_velocity = if neighbor_is_upwind {
                    neighbor_cell_advection_velocity
                } else {
                    cell_advection_velocity
                };

                let linear_face_advected_velocity = weight * self.velocity.cell_value(cell.index)
                    + neighbor_weight * self.velocity.cell_value(neighbor_cell.index);

                upwind_face_advection_velocity.dot(&outward_normal) * linear_face_advected_velocity
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
                    BoundaryCondition::Kinematic(height_time_deriv) => {
                        compute_kinematic_face_velocity_value(height_time_deriv)
                    }
                    BoundaryCondition::NoPenetration => {
                        let unprojected_face_value = 0.5
                            * (self.velocity.cell_value(cell.index)
                                + self.velocity.cell_value(block_paired_cell.index));
                        unprojected_face_value
                            - unprojected_face_value.dot(&outward_normal)
                                * outward_normal.into_inner()
                    }
                };

                face_velocity.dot(&outward_normal) * face_velocity
            }
        }

        derivs::compute_field_differential(
            dynamic_geometry,
            boundary_conditions,
            AdvectionComputer {
                velocity: self,
                advection_velocity,
            },
        )
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
                face_centroid: Point3,
            ) -> Vector3 {
                linearly_interpolate_to_face(
                    self.tensor_field,
                    &face_centroid,
                    &outward_normal,
                    cell,
                    neighbor_cell,
                )
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
                    BoundaryCondition::NoPenetration => todo!(),
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
impl AreaScalarField {
    pub fn values_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.cell_footprints.clone().into_pyarray(py)
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

pub fn compute_linear_interpolation_weights(
    face_centroid: &Point3,
    outward_normal: &UnitVector3,
    cell: &geom::Cell,
    neighbor_cell: &geom::Cell,
) -> (Float, Float) {
    let norm = (neighbor_cell.centroid - cell.centroid).dot(outward_normal);
    let cell_weight = (neighbor_cell.centroid - face_centroid).dot(outward_normal) / norm;
    let neighbor_cell_weight = 1. - cell_weight;
    (cell_weight, neighbor_cell_weight)
}

pub fn linearly_interpolate_to_face<V: Value>(
    field: &VolField<V>,
    face_centroid: &Point3,
    outward_normal: &UnitVector3,
    cell: &geom::Cell,
    neighbor_cell: &geom::Cell,
) -> V {
    let (cell_weight, neighbor_cell_weight) =
        compute_linear_interpolation_weights(face_centroid, outward_normal, cell, neighbor_cell);
    field.cell_value(cell.index) * cell_weight
        + field.cell_value(neighbor_cell.index) * neighbor_cell_weight
}

/// Compute the face value of the velocity at a boundary specifying the
/// kinematic boundary condition.
fn compute_kinematic_face_velocity_value(height_time_deriv: Float) -> Vector3 {
    height_time_deriv * Vector3::z_axis().into_inner()
    // Let u' be the face velocity and u be the cell velocity.
    //
    // BCs:
    //  u'.n = z.n * dh/dt
    //  u' - (u'.n) n = 0
    // Combining:
    //  u' - (z.n * dh/dt) n = 0
    //  u' = (z.n * dh/dt) n
    // (height_time_deriv * outward_normal.z) * outward_normal.into_inner()
    // // BCs:
    // //  u'.n = z.n * dh/dt
    // //  u' - (u'.n) n = u - (u.n) n
    // // Combining:
    // //  u' - (z.n * dh/dt) n = u - (u.n) n
    // //  u' = u + (z.n * dh/dt - u.n) n
    // velocity_cell_value
    //     + (-outward_normal.dot(&velocity_cell_value) + height_time_deriv * outward_normal.z)
    //         * outward_normal.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::implicit;

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

            let advection =
                velocity.advect_upwind(&velocity, &dynamic_geometry, &boundary_conditions);

            advection
                .assert_all_close(&velocity, &dynamic_geometry)
                .abs_tol(Some(1e-5));
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

            let advection =
                velocity.advect_upwind(&velocity, &dynamic_geometry, &vel_boundary_conditions);

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
            let divergence = velocity.divergence(&dynamic_geometry, &boundary_conditions);
            let laplacian = velocity.laplacian(&dynamic_geometry, &shear, &boundary_conditions);
            let implicit_laplacian = implicit::evaluate(
                implicit::ImplicitVolField::<Vector3>::default().laplacian(Some(&shear)),
                &dynamic_geometry,
                &boundary_conditions,
                &velocity,
            );

            shear
                .assert_all_close(&VolTensorField::zeros(cell_indexing), &dynamic_geometry)
                .abs_tol(Some(1e-5));
            divergence
                .assert_all_close(&VolScalarField::zeros(cell_indexing), &dynamic_geometry)
                .abs_tol(Some(1e-5));
            laplacian
                .assert_all_close(&VolVectorField::zeros(cell_indexing), &dynamic_geometry)
                .abs_tol(Some(1e-5));
            implicit_laplacian
                .assert_all_close(&VolVectorField::zeros(cell_indexing), &dynamic_geometry)
                .abs_tol(Some(1e-5));
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
            let divergence = velocity.divergence(&dynamic_geometry, &boundary_conditions);
            let laplacian = velocity.laplacian(&dynamic_geometry, &shear, &boundary_conditions);
            let implicit_laplacian = implicit::evaluate(
                implicit::ImplicitVolField::<Vector3>::default().laplacian(Some(&shear)),
                &dynamic_geometry,
                &boundary_conditions,
                &velocity,
            );

            shear
                .assert_all_close(&VolTensorField::zeros(cell_indexing), &dynamic_geometry)
                .abs_tol(Some(1e-5));
            divergence
                .assert_all_close(&VolScalarField::zeros(cell_indexing), &dynamic_geometry)
                .abs_tol(Some(1e-5));
            laplacian
                .assert_all_close(&VolVectorField::zeros(cell_indexing), &dynamic_geometry)
                .abs_tol(Some(1e-5));
            implicit_laplacian
                .assert_all_close(&VolVectorField::zeros(cell_indexing), &dynamic_geometry)
                .abs_tol(Some(1e-5));
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
            let divergence = velocity.divergence(&dynamic_geometry, &boundary_conditions);
            let laplacian = velocity.laplacian(&dynamic_geometry, &shear, &boundary_conditions);
            let implicit_laplacian = implicit::evaluate(
                implicit::ImplicitVolField::<Vector3>::default().laplacian(Some(&shear)),
                &dynamic_geometry,
                &boundary_conditions,
                &velocity,
            );

            laplacian
                .assert_all_close(&implicit_laplacian, &dynamic_geometry)
                .abs_tol(Some(1e-8))
                .rel_tol(None);

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
            let expected_divergence = VolScalarField::new(&dynamic_geometry, |x, y, _| {
                bump_deriv(x) * bump(y) + -2. * bump(x) * bump_deriv(y)
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
                        "At cell index {cell_index:?}: shear: {shear_value} expected:
                        {expected_shear_value}"
                    );
                }

                let divergence_value = divergence.cell_value(cell_index);
                let expected_divergence_value = expected_divergence.cell_value(cell_index);
                if !approx::relative_eq!(
                    divergence_value,
                    expected_divergence_value,
                    max_relative = 0.1,
                    epsilon = 0.1
                ) {
                    panic!(
                        "At cell index {cell_index:?}: divergence: {divergence_value} expected: \
                         {expected_divergence_value}"
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
                                "At cell index {cell_index:?}: laplacian: {laplacian_value} \
                                 expected:
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
                                "At cell index {cell_index:?}: laplacian:
                            {laplacian_value} expected:
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
        geom::DynamicGeometry::new_from_height(static_geometry, &height)
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
        geom::DynamicGeometry::new_from_height(static_geometry, &height)
    }
}
