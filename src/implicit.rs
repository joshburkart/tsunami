use std::ops;

use ndarray as nd;

use crate::{fields, geom, indexing, linalg, Array1, Float, Vector3};

pub trait ImplicitValue: fields::Value {}
impl ImplicitValue for Float {}
impl ImplicitValue for Vector3 {}

#[derive(Clone, Copy, Default)]
pub struct ImplicitVolField<V: ImplicitValue> {
    _phantom: std::marker::PhantomData<V>,
}

pub struct CellCoef {
    cell_index: indexing::CellIndex,
    /// The coefficient, or [`None`] to indicate `1.`.
    coef: Option<Float>,
}
pub struct Subsystem<V: ImplicitValue, I: Iterator<Item = CellCoef>> {
    coef_iter: I,
    /// The right-hand side set of values, or [`None`] to indicate `0.`s.
    rhs: Option<V>,
}

pub trait ImplicitTerm {
    type ImplicitValue: ImplicitValue;
    type CellCoefIter: Iterator<Item = CellCoef>;

    fn compute_subsystem(
        &self,
        cell_index: indexing::CellIndex,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<Self::ImplicitValue>,
    ) -> Subsystem<Self::ImplicitValue, Self::CellCoefIter>;
}

impl<V: ImplicitValue> ImplicitTerm for ImplicitVolField<V> {
    type CellCoefIter = std::iter::Once<CellCoef>;
    type ImplicitValue = V;

    fn compute_subsystem(
        &self,
        cell_index: indexing::CellIndex,
        _dynamic_geometry: &geom::DynamicGeometry,
        _boundary_conditions: &fields::BoundaryConditions<Self::ImplicitValue>,
    ) -> Subsystem<Self::ImplicitValue, Self::CellCoefIter> {
        Subsystem {
            coef_iter: std::iter::once(CellCoef {
                cell_index,
                coef: None,
            }),
            rhs: None,
        }
    }
}

pub struct ImplicitScalarProduct<T: ImplicitTerm> {
    scalar: Float,
    term: T,
}
impl<T: ImplicitTerm> ImplicitTerm for ImplicitScalarProduct<T> {
    type CellCoefIter = ImplicitScalarProductCellCoefIter<T::CellCoefIter>;
    type ImplicitValue = T::ImplicitValue;

    fn compute_subsystem(
        &self,
        cell_index: indexing::CellIndex,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<T::ImplicitValue>,
    ) -> Subsystem<Self::ImplicitValue, Self::CellCoefIter> {
        let factor_submatrix =
            self.term
                .compute_subsystem(cell_index, dynamic_geometry, boundary_conditions);
        Subsystem {
            coef_iter: ImplicitScalarProductCellCoefIter {
                term_coef_iter: factor_submatrix.coef_iter,
                scalar: self.scalar,
            },
            rhs: factor_submatrix.rhs.map(|rhs| rhs * self.scalar),
        }
    }
}

pub struct ImplicitScalarProductCellCoefIter<I: Iterator<Item = CellCoef>> {
    term_coef_iter: I,
    scalar: Float,
}
impl<I: Iterator<Item = CellCoef>> Iterator for ImplicitScalarProductCellCoefIter<I> {
    type Item = CellCoef;

    fn next(&mut self) -> Option<Self::Item> {
        self.term_coef_iter.next().map(|cell_coef| CellCoef {
            coef: Some(cell_coef.coef.unwrap_or(1.) * self.scalar),
            ..cell_coef
        })
    }
}

pub struct ImplicitTermSum<T1: ImplicitTerm, T2: ImplicitTerm> {
    term_1: T1,
    term_2: T2,
}
impl<V: ImplicitValue, T1: ImplicitTerm<ImplicitValue = V>, T2: ImplicitTerm<ImplicitValue = V>>
    ImplicitTerm for ImplicitTermSum<T1, T2>
{
    type CellCoefIter = std::iter::Chain<T1::CellCoefIter, T2::CellCoefIter>;
    type ImplicitValue = <T1::ImplicitValue as std::ops::Add<T2::ImplicitValue>>::Output;

    fn compute_subsystem(
        &self,
        cell_index: indexing::CellIndex,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<V>,
    ) -> Subsystem<V, Self::CellCoefIter> {
        let Subsystem {
            coef_iter: coef_iter_1,
            rhs: rhs_1,
        } = self
            .term_1
            .compute_subsystem(cell_index, dynamic_geometry, boundary_conditions);
        let Subsystem {
            coef_iter: coef_iter_2,
            rhs: rhs_2,
        } = self
            .term_2
            .compute_subsystem(cell_index, dynamic_geometry, boundary_conditions);

        let coef_iter = coef_iter_1.chain(coef_iter_2);
        let rhs = Some(
            rhs_1.unwrap_or(<T1::ImplicitValue as num_traits::Zero>::zero())
                + rhs_2.unwrap_or(<T2::ImplicitValue as num_traits::Zero>::zero()),
        );

        Subsystem { coef_iter, rhs }
    }
}

impl<'a, V: ImplicitValue> ImplicitTerm for &'a fields::VolField<V> {
    type CellCoefIter = std::iter::Empty<CellCoef>;
    type ImplicitValue = V;

    fn compute_subsystem(
        &self,
        cell_index: indexing::CellIndex,
        _dynamic_geometry: &geom::DynamicGeometry,
        _boundary_conditions: &fields::BoundaryConditions<V>,
    ) -> Subsystem<V, Self::CellCoefIter> {
        Subsystem {
            coef_iter: std::iter::empty(),
            rhs: Some(-self.cell_value(cell_index)),
        }
    }
}

impl ImplicitTerm for Float {
    type CellCoefIter = std::iter::Empty<CellCoef>;
    type ImplicitValue = Self;

    fn compute_subsystem(
        &self,
        _cell_index: indexing::CellIndex,
        _dynamic_geometry: &geom::DynamicGeometry,
        _boundary_conditions: &fields::BoundaryConditions<Self::ImplicitValue>,
    ) -> Subsystem<Self::ImplicitValue, Self::CellCoefIter> {
        Subsystem {
            coef_iter: std::iter::empty(),
            rhs: Some(*self),
        }
    }
}
impl ImplicitTerm for Vector3 {
    type CellCoefIter = std::iter::Empty<CellCoef>;
    type ImplicitValue = Self;

    fn compute_subsystem(
        &self,
        _cell_index: indexing::CellIndex,
        _dynamic_geometry: &geom::DynamicGeometry,
        _boundary_conditions: &fields::BoundaryConditions<Self::ImplicitValue>,
    ) -> Subsystem<Self::ImplicitValue, Self::CellCoefIter> {
        Subsystem {
            coef_iter: std::iter::empty(),
            rhs: Some(-*self),
        }
    }
}

mod op_impls {
    use super::*;

    impl<V: ImplicitValue> ops::Neg for ImplicitVolField<V> {
        type Output = ImplicitScalarProduct<Self>;

        fn neg(self) -> Self::Output {
            ImplicitScalarProduct {
                scalar: -1.,
                term: self,
            }
        }
    }
    impl<V: ImplicitValue> ops::Div<Float> for ImplicitVolField<V> {
        type Output = ImplicitScalarProduct<Self>;

        fn div(self, rhs: Float) -> Self::Output {
            ImplicitScalarProduct {
                scalar: 1. / rhs,
                term: self,
            }
        }
    }
    impl<V: ImplicitValue> ops::Mul<ImplicitVolField<V>> for Float {
        type Output = ImplicitScalarProduct<ImplicitVolField<V>>;

        fn mul(self, rhs: ImplicitVolField<V>) -> Self::Output {
            ImplicitScalarProduct {
                scalar: self,
                term: rhs,
            }
        }
    }

    impl ops::Sub<Vector3> for ImplicitVolField<Vector3> {
        type Output = ImplicitTermSum<ImplicitVolField<Vector3>, Vector3>;

        fn sub(self, rhs: Vector3) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: -rhs,
            }
        }
    }
    impl<T: ImplicitTerm> ops::Sub<Vector3> for ImplicitScalarProduct<T> {
        type Output = ImplicitTermSum<ImplicitScalarProduct<T>, Vector3>;

        fn sub(self, rhs: Vector3) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: -rhs,
            }
        }
    }
    impl ops::Add<Vector3> for ImplicitVolField<Vector3> {
        type Output = ImplicitTermSum<ImplicitVolField<Vector3>, Vector3>;

        fn add(self, rhs: Vector3) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: rhs,
            }
        }
    }
    impl<T: ImplicitTerm> ops::Add<Vector3> for ImplicitScalarProduct<T> {
        type Output = ImplicitTermSum<ImplicitScalarProduct<T>, Vector3>;

        fn add(self, rhs: Vector3) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: rhs,
            }
        }
    }
    impl<
        'a,
        V: ImplicitValue,
        T1: ImplicitTerm<ImplicitValue = V>,
        T2: ImplicitTerm<ImplicitValue = V>,
    > ops::Sub<&'a fields::VolField<V>> for ImplicitTermSum<T1, T2>
    {
        type Output = ImplicitTermSum<
            ImplicitTermSum<T1, T2>,
            ImplicitScalarProduct<&'a fields::VolField<V>>,
        >;

        fn sub(self, rhs: &'a fields::VolField<V>) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: ImplicitScalarProduct {
                    scalar: -1.,
                    term: rhs,
                },
            }
        }
    }
}

pub struct ImplicitSolver {
    linear_solver: linalg::LinearSolver,
}
impl ImplicitSolver {
    pub fn new(linear_solver: linalg::LinearSolver) -> Self {
        Self { linear_solver }
    }

    pub fn solve<T: ImplicitTerm>(
        &self,
        implicit_term: T,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<T::ImplicitValue>,
        guess: Option<&fields::VolField<T::ImplicitValue>>,
    ) -> Result<fields::VolField<T::ImplicitValue>, linalg::LinearSolveError> {
        use fields::Value;
        use indexing::{Indexing, IntoIndexIterator};

        let cell_indexing = dynamic_geometry.grid().cell_indexing();
        let value_size = T::ImplicitValue::size();
        let matrix_size = cell_indexing.len() * value_size;

        let mut matrix = sprs::TriMat::new((matrix_size, matrix_size));
        let mut rhs = Array1::zeros(matrix_size);
        for cell_index in cell_indexing.iter() {
            let subsystem =
                implicit_term.compute_subsystem(cell_index, dynamic_geometry, boundary_conditions);
            let flat_index = cell_indexing.flatten(cell_index);

            if let Some(subsystem_rhs) = subsystem.rhs {
                subsystem_rhs.flatten(rhs.slice_mut(nd::s![
                    flat_index * value_size..(flat_index + 1) * value_size
                ]));
            }

            for CellCoef {
                cell_index: col_cell_index,
                coef,
            } in subsystem.coef_iter
            {
                let coef = coef.unwrap_or(1.);
                let col_flat_index = cell_indexing.flatten(col_cell_index);
                for i in 0..value_size {
                    matrix.add_triplet(
                        flat_index * value_size + i,
                        col_flat_index * value_size + i,
                        coef,
                    );
                }
            }
        }

        let x = guess
            .map(|guess| guess.flatten(cell_indexing))
            .unwrap_or_else(|| Array1::zeros(matrix_size));
        Ok(fields::VolField::unflatten(
            cell_indexing,
            &self.linear_solver.solve(matrix.to_csr().view(), x, rhs)?,
        ))
    }
}

// let u = implicit::ImplicitVolField::<Float>::new(dynamic_geometry,
// boundary_conditions);
// let eq = (u - prev_u) / dt + u.advect_by(prev_u) - nu * u.laplacian() ==
//      -prev_grad_p / rho + g;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_and_scalings() {
        use indexing::IntoIndexIterator;

        let xy_axis = geom::Axis::new(0., 1., 3);
        let grid = geom::Grid::new(xy_axis.clone(), xy_axis, 5);
        let static_geometry = geom::StaticGeometry::new(grid, |_x, _y| 0.);
        let height = fields::AreaField::new(static_geometry.grid(), |_x, _y| 0.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let boundary_conditions = fields::BoundaryConditions {
            horiz: fields::HorizBoundaryConditions::hom_dirichlet(),
            z: fields::VertBoundaryFieldPair::hom_dirichlet(),
        };

        let explicit_field =
            fields::VolVectorField::new(&dynamic_geometry, |x, y, z| Vector3::new(x, y, z));

        let implicit_field = ImplicitVolField::default();

        for solver in [
            ImplicitSolver::new(linalg::LinearSolver::Direct),
            ImplicitSolver::new(linalg::LinearSolver::GaussSeidel {
                max_iters: 20,
                rel_error_tol: 1e-7,
            }),
        ] {
            {
                let x_sol = solver
                    .solve(
                        implicit_field - Vector3::new(1., -2., 3.),
                        &dynamic_geometry,
                        &boundary_conditions,
                        None,
                    )
                    .unwrap();
                for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
                    approx::assert_relative_eq!(
                        x_sol.cell_value(cell_index),
                        Vector3::new(1., -2., 3.)
                    );
                }
            }
            {
                let x_sol = solver
                    .solve(
                        -implicit_field - Vector3::new(1., -2., 3.),
                        &dynamic_geometry,
                        &boundary_conditions,
                        None,
                    )
                    .unwrap();
                for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
                    approx::assert_relative_eq!(
                        x_sol.cell_value(cell_index),
                        Vector3::new(-1., 2., -3.)
                    );
                }
            }
            {
                let x_sol = solver
                    .solve(
                        implicit_field + Vector3::new(1., -2., 3.),
                        &dynamic_geometry,
                        &boundary_conditions,
                        None,
                    )
                    .unwrap();
                for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
                    approx::assert_relative_eq!(
                        x_sol.cell_value(cell_index),
                        Vector3::new(-1., 2., -3.)
                    );
                }
            }
            {
                let x_sol = solver
                    .solve(
                        2. * implicit_field + Vector3::new(1., -2., 3.),
                        &dynamic_geometry,
                        &boundary_conditions,
                        None,
                    )
                    .unwrap();
                for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
                    approx::assert_relative_eq!(
                        x_sol.cell_value(cell_index),
                        Vector3::new(-1. / 2., 2. / 2., -3. / 2.)
                    );
                }
            }
            {
                let x_sol = solver
                    .solve(
                        2. * implicit_field + Vector3::new(1., -2., 3.) - &explicit_field,
                        &dynamic_geometry,
                        &boundary_conditions,
                        None,
                    )
                    .unwrap();
                for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
                    approx::assert_relative_eq!(
                        x_sol.cell_value(cell_index),
                        Vector3::new(-1. / 2., 2. / 2., -3. / 2.)
                            + explicit_field.cell_value(cell_index) / 2.
                    );
                }
            }
        }
    }

    /// Test ensuring matrix terms sum appropriately.
    #[test]
    fn test_matrices_sum() {
        let mut matrix = sprs::TriMat::<Float>::new((2, 2));
        matrix.add_triplet(0, 0, 1.);
        matrix.add_triplet(0, 0, -2.);
        let matrix: sprs::CsMat<Float> = matrix.to_csc();
        approx::assert_relative_eq!(matrix.to_dense()[[0, 0]], -1.);
    }
}
