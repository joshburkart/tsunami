use std::ops;

use ndarray as nd;

use crate::{fields, geom, indexing, linalg, Array1, Float, Point3, UnitVector3, Vector3};

pub trait ImplicitValue: fields::Value {}
impl ImplicitValue for Float {}
impl ImplicitValue for Vector3 {}

#[derive(Clone, Copy, Default)]
pub struct ImplicitVolField<V: ImplicitValue> {
    _phantom: std::marker::PhantomData<V>,
}
impl ImplicitVolField<Float> {
    pub fn laplacian(
        self,
        explicit_grad: Option<&fields::VolVectorField>,
    ) -> DifferentialImplicitSystem<SemiImplicitScalarLaplacianField> {
        DifferentialImplicitSystem {
            differential: SemiImplicitScalarLaplacianField { explicit_grad },
        }
    }
}
impl ImplicitVolField<Vector3> {
    pub fn laplacian(
        self,
        explicit_grad: Option<&fields::VolTensorField>,
    ) -> DifferentialImplicitSystem<SemiImplicitVectorLaplacianField> {
        DifferentialImplicitSystem {
            differential: SemiImplicitVectorLaplacianField { explicit_grad },
        }
    }

    pub fn advect_upwind(
        self,
        explicit_velocity: &fields::VolVectorField,
    ) -> DifferentialImplicitSystem<SemiImplicitVectorUpwindAdvectionField> {
        DifferentialImplicitSystem {
            differential: SemiImplicitVectorUpwindAdvectionField { explicit_velocity },
        }
    }
}

pub struct CellCoef {
    cell_index: indexing::CellIndex,
    coef: Float,
}

pub trait ImplicitSystem {
    type ImplicitValue: ImplicitValue;

    /// Generate the subsystem of an overall field equation linear system
    /// corresponding to a particular cell.
    ///
    /// A closure to call to add a new column coefficient will be supplied via
    /// the `add_cell_coef` argument.
    ///
    /// # Returns
    ///
    /// Any constant value for the subsystem, i.e. the `b` in `Ax + b = 0`.
    fn gen_subsystem<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell_index: indexing::CellIndex,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<Self::ImplicitValue>,
    ) -> Self::ImplicitValue;
}

impl<V: ImplicitValue> ImplicitSystem for ImplicitVolField<V> {
    type ImplicitValue = V;

    fn gen_subsystem<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell_index: indexing::CellIndex,
        _dynamic_geometry: &geom::DynamicGeometry,
        _boundary_conditions: &fields::BoundaryConditions<Self::ImplicitValue>,
    ) -> Self::ImplicitValue {
        add_cell_coef(CellCoef {
            cell_index,
            coef: 1.,
        });
        <Self::ImplicitValue as num_traits::Zero>::zero()
    }
}

pub struct ImplicitScalarProduct<S: ImplicitSystem> {
    scalar: Float,
    term: S,
}
impl<S: ImplicitSystem> ImplicitSystem for ImplicitScalarProduct<S> {
    type ImplicitValue = S::ImplicitValue;

    fn gen_subsystem<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell_index: indexing::CellIndex,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<S::ImplicitValue>,
    ) -> Self::ImplicitValue {
        let mut add_cell_coef_wrapper = |cell_coef: CellCoef| {
            add_cell_coef(CellCoef {
                cell_index: cell_coef.cell_index,
                coef: cell_coef.coef * self.scalar,
            })
        };

        self.term.gen_subsystem(
            &mut add_cell_coef_wrapper,
            cell_index,
            dynamic_geometry,
            boundary_conditions,
        ) * self.scalar
    }
}

pub struct ImplicitTermSum<S1: ImplicitSystem, S2: ImplicitSystem> {
    term_1: S1,
    term_2: S2,
}
impl<
        V: ImplicitValue,
        S1: ImplicitSystem<ImplicitValue = V>,
        S2: ImplicitSystem<ImplicitValue = V>,
    > ImplicitSystem for ImplicitTermSum<S1, S2>
{
    type ImplicitValue = <S1::ImplicitValue as std::ops::Add<S2::ImplicitValue>>::Output;

    fn gen_subsystem<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell_index: indexing::CellIndex,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<V>,
    ) -> V {
        let rhs_1 = self.term_1.gen_subsystem(
            add_cell_coef,
            cell_index,
            dynamic_geometry,
            boundary_conditions,
        );
        let rhs_2 = self.term_2.gen_subsystem(
            add_cell_coef,
            cell_index,
            dynamic_geometry,
            boundary_conditions,
        );
        rhs_1 + rhs_2
    }
}

impl<'a, V: ImplicitValue> ImplicitSystem for &'a fields::VolField<V> {
    type ImplicitValue = V;

    fn gen_subsystem<F: FnMut(CellCoef)>(
        &self,
        _add_cell_coef: &mut F,
        cell_index: indexing::CellIndex,
        _dynamic_geometry: &geom::DynamicGeometry,
        _boundary_conditions: &fields::BoundaryConditions<V>,
    ) -> V {
        self.cell_value(cell_index)
    }
}

impl<V: ImplicitValue> ImplicitSystem for V {
    type ImplicitValue = V;

    fn gen_subsystem<F: FnMut(CellCoef)>(
        &self,
        _add_cell_coef: &mut F,
        _cell_index: indexing::CellIndex,
        _dynamic_geometry: &geom::DynamicGeometry,
        _boundary_conditions: &fields::BoundaryConditions<Self::ImplicitValue>,
    ) -> Self::ImplicitValue {
        *self
    }
}

pub trait DifferentialImplicitSystemImpl {
    type ImplicitValue: ImplicitValue;

    fn handle_interior_face<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell: &geom::Cell,
        neighbor_cell: &geom::Cell,
        outward_normal: UnitVector3,
    ) -> Self::ImplicitValue;

    fn handle_boundary_face<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell: &geom::Cell,
        block_paired_cell: &geom::Cell,
        outward_normal: UnitVector3,
        face_centroid: Point3,
        boundary_condition: fields::BoundaryCondition<Self::ImplicitValue>,
    ) -> Self::ImplicitValue;
}

pub struct DifferentialImplicitSystem<D: DifferentialImplicitSystemImpl> {
    differential: D,
}

impl<D: DifferentialImplicitSystemImpl> ImplicitSystem for DifferentialImplicitSystem<D> {
    type ImplicitValue = D::ImplicitValue;

    fn gen_subsystem<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell_index: indexing::CellIndex,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<Self::ImplicitValue>,
    ) -> Self::ImplicitValue {
        let cell = dynamic_geometry.cell(cell_index);

        let mut lhs_constant = <Self::ImplicitValue as num_traits::Zero>::zero();
        for face in &cell.faces {
            let mut add_cell_coef_wrapper = |cell_coef: CellCoef| {
                add_cell_coef(CellCoef {
                    cell_index: cell_coef.cell_index,
                    coef: cell_coef.coef * face.area() / cell.volume,
                })
            };
            lhs_constant += match face.neighbor() {
                indexing::CellNeighbor::Cell(neighbor_cell_index) => {
                    self.differential.handle_interior_face(
                        &mut add_cell_coef_wrapper,
                        cell,
                        dynamic_geometry.cell(neighbor_cell_index),
                        face.outward_normal(),
                    )
                }
                indexing::CellNeighbor::XBoundary(boundary) => match boundary {
                    indexing::Boundary::Lower => self.differential.handle_boundary_face(
                        &mut add_cell_coef_wrapper,
                        cell,
                        dynamic_geometry.cell(cell_index.flip()),
                        face.outward_normal(),
                        face.centroid(),
                        boundary_conditions.horiz.x.lower,
                    ),
                    indexing::Boundary::Upper => self.differential.handle_boundary_face(
                        &mut add_cell_coef_wrapper,
                        cell,
                        dynamic_geometry.cell(cell_index.flip()),
                        face.outward_normal(),
                        face.centroid(),
                        boundary_conditions.horiz.x.upper,
                    ),
                },
                indexing::CellNeighbor::YBoundary(boundary) => match boundary {
                    indexing::Boundary::Lower => self.differential.handle_boundary_face(
                        &mut add_cell_coef_wrapper,
                        cell,
                        dynamic_geometry.cell(cell_index.flip()),
                        face.outward_normal(),
                        face.centroid(),
                        boundary_conditions.horiz.y.lower,
                    ),
                    indexing::Boundary::Upper => self.differential.handle_boundary_face(
                        &mut add_cell_coef_wrapper,
                        cell,
                        dynamic_geometry.cell(cell_index.flip()),
                        face.outward_normal(),
                        face.centroid(),
                        boundary_conditions.horiz.y.upper,
                    ),
                },
                indexing::CellNeighbor::ZBoundary(boundary) => match boundary {
                    indexing::Boundary::Lower => self.differential.handle_boundary_face(
                        &mut add_cell_coef_wrapper,
                        cell,
                        cell,
                        face.outward_normal(),
                        face.centroid(),
                        boundary_conditions
                            .z
                            .lower
                            .boundary_condition(cell_index.footprint),
                    ),
                    indexing::Boundary::Upper => self.differential.handle_boundary_face(
                        &mut add_cell_coef_wrapper,
                        cell,
                        cell,
                        face.outward_normal(),
                        face.centroid(),
                        boundary_conditions
                            .z
                            .upper
                            .boundary_condition(cell_index.footprint),
                    ),
                },
            } * face.area()
                / cell.volume
        }
        lhs_constant
    }
}

pub struct SemiImplicitScalarLaplacianField<'a> {
    explicit_grad: Option<&'a fields::VolVectorField>,
}
impl<'a> DifferentialImplicitSystemImpl for SemiImplicitScalarLaplacianField<'a> {
    type ImplicitValue = Float;

    fn handle_interior_face<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell: &geom::Cell,
        neighbor_cell: &geom::Cell,
        outward_normal: UnitVector3,
    ) -> Self::ImplicitValue {
        let displ = neighbor_cell.centroid - cell.centroid;
        let c_corr = 1. / outward_normal.dot(&displ);
        add_cell_coef(CellCoef {
            cell_index: cell.index,
            coef: -c_corr,
        });
        add_cell_coef(CellCoef {
            cell_index: neighbor_cell.index,
            coef: c_corr,
        });

        if let Some(explicit_grad) = self.explicit_grad {
            let explicit_grad_at_face = 0.5
                * (explicit_grad.cell_value(cell.index)
                    + explicit_grad.cell_value(neighbor_cell.index));
            let explicit_correction =
                (outward_normal.into_inner() - c_corr * displ).dot(&explicit_grad_at_face);
            explicit_correction
        } else {
            0.
        }
    }

    fn handle_boundary_face<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell: &geom::Cell,
        block_paired_cell: &geom::Cell,
        outward_normal: UnitVector3,
        face_centroid: Point3,
        boundary_condition: fields::BoundaryCondition<Self::ImplicitValue>,
    ) -> Self::ImplicitValue {
        match boundary_condition {
            fields::BoundaryCondition::HomDirichlet => {
                let displ = face_centroid.coords
                    - 0.5 * (cell.centroid.coords + block_paired_cell.centroid.coords);
                let c_corr = 1. / outward_normal.dot(&displ);
                let coef = -0.5 * c_corr;
                add_cell_coef(CellCoef {
                    cell_index: cell.index,
                    coef,
                });
                add_cell_coef(CellCoef {
                    cell_index: block_paired_cell.index,
                    coef,
                });
                0.
            }
            fields::BoundaryCondition::HomNeumann => 0.,
            fields::BoundaryCondition::InhomNeumann(boundary_value) => {
                // TODO: Account for block paired cell if an inhomogeneous Neumann BC is
                // used for X or Y boundaries.
                boundary_value
            }
            fields::BoundaryCondition::Kinematic(_) => unimplemented!(),
        }
    }
}

pub struct SemiImplicitVectorLaplacianField<'a> {
    explicit_grad: Option<&'a fields::VolTensorField>,
}
impl<'a> DifferentialImplicitSystemImpl for SemiImplicitVectorLaplacianField<'a> {
    type ImplicitValue = Vector3;

    fn handle_interior_face<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell: &geom::Cell,
        neighbor_cell: &geom::Cell,
        outward_normal: UnitVector3,
    ) -> Self::ImplicitValue {
        let displ = neighbor_cell.centroid - cell.centroid;
        let c_corr = 1. / outward_normal.dot(&displ);
        add_cell_coef(CellCoef {
            cell_index: cell.index,
            coef: -c_corr,
        });
        add_cell_coef(CellCoef {
            cell_index: neighbor_cell.index,
            coef: c_corr,
        });

        if let Some(explicit_grad) = self.explicit_grad {
            let explicit_grad_at_face = 0.5
                * (explicit_grad.cell_value(cell.index)
                    + explicit_grad.cell_value(neighbor_cell.index));
            let explicit_correction =
                explicit_grad_at_face.tr_mul(&(outward_normal.into_inner() - c_corr * displ));
            explicit_correction
        } else {
            Vector3::zeros()
        }
    }

    fn handle_boundary_face<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell: &geom::Cell,
        block_paired_cell: &geom::Cell,
        outward_normal: UnitVector3,
        face_centroid: Point3,
        boundary_condition: fields::BoundaryCondition<Self::ImplicitValue>,
    ) -> Self::ImplicitValue {
        match boundary_condition {
            fields::BoundaryCondition::HomDirichlet => {
                let displ = face_centroid.coords
                    - 0.5 * (cell.centroid.coords + block_paired_cell.centroid.coords);
                let c_corr = 1. / outward_normal.dot(&displ);
                let coef = -0.5 * c_corr;
                add_cell_coef(CellCoef {
                    cell_index: cell.index,
                    coef,
                });
                add_cell_coef(CellCoef {
                    cell_index: block_paired_cell.index,
                    coef,
                });
                Vector3::zeros()
            }
            fields::BoundaryCondition::HomNeumann => Vector3::zeros(),
            fields::BoundaryCondition::InhomNeumann(boundary_value) => {
                // TODO: Account for block paired cell if an inhomogeneous Neumann BC is
                // used for X or Y boundaries.
                boundary_value
            }
            fields::BoundaryCondition::Kinematic(height_time_deriv) => {
                height_time_deriv * Vector3::z_axis().into_inner()
                // BoundaryCondition::Kinematic(height_time_deriv) => {
                //     let cell_value = self.vector_field.cell_value(cell.index);
                //     let face_value =
                //         compute_kinematic_face_velocity_value(cell_value, height_time_deriv, outward_normal);
                //     (face_value - cell_value)
                //         / (face_centroid.coords
                //             - 0.5 * (cell.centroid.coords + block_paired_cell.centroid.coords))
                //             .norm()
                // }
            }
        }
    }
}

pub struct SemiImplicitVectorUpwindAdvectionField<'a> {
    explicit_velocity: &'a fields::VolVectorField,
}
impl<'a> DifferentialImplicitSystemImpl for SemiImplicitVectorUpwindAdvectionField<'a> {
    type ImplicitValue = Vector3;

    fn handle_interior_face<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell: &geom::Cell,
        neighbor_cell: &geom::Cell,
        outward_normal: UnitVector3,
    ) -> Self::ImplicitValue {
        let cell_velocity = self.explicit_velocity.cell_value(cell.index);
        let neighbor_cell_velocity = self.explicit_velocity.cell_value(neighbor_cell.index);
        let linear_face_velocity = 0.5 * (cell_velocity + neighbor_cell_velocity);

        let projected_face_velocity = linear_face_velocity.dot(&outward_normal);
        let neighbor_is_upwind = projected_face_velocity < 0.;
        let upwind_face_velocity = if neighbor_is_upwind {
            neighbor_cell_velocity
        } else {
            cell_velocity
        };

        let coef = 0.5 * upwind_face_velocity.dot(&outward_normal);
        add_cell_coef(CellCoef {
            cell_index: cell.index,
            coef,
        });
        add_cell_coef(CellCoef {
            cell_index: neighbor_cell.index,
            coef,
        });
        Vector3::zeros()
    }

    fn handle_boundary_face<F: FnMut(CellCoef)>(
        &self,
        add_cell_coef: &mut F,
        cell: &geom::Cell,
        block_paired_cell: &geom::Cell,
        outward_normal: UnitVector3,
        _face_centroid: Point3,
        boundary_condition: fields::BoundaryCondition<Self::ImplicitValue>,
    ) -> Self::ImplicitValue {
        match boundary_condition {
            fields::BoundaryCondition::HomDirichlet => Vector3::zeros(),
            fields::BoundaryCondition::HomNeumann => {
                let explicit_face_velocity = 0.5
                    * (self.explicit_velocity.cell_value(cell.index)
                        + self.explicit_velocity.cell_value(block_paired_cell.index));
                let coef = 0.5 * explicit_face_velocity.dot(&outward_normal);
                add_cell_coef(CellCoef {
                    cell_index: cell.index,
                    coef,
                });
                add_cell_coef(CellCoef {
                    cell_index: block_paired_cell.index,
                    coef,
                });
                Vector3::zeros()
            }
            fields::BoundaryCondition::InhomNeumann(_) => unimplemented!(),
            fields::BoundaryCondition::Kinematic(height_time_deriv) => {
                height_time_deriv.powi(2) * outward_normal.z * Vector3::z_axis().into_inner()
            }
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
    impl<D: DifferentialImplicitSystemImpl> ops::Mul<DifferentialImplicitSystem<D>> for Float {
        type Output = ImplicitScalarProduct<DifferentialImplicitSystem<D>>;

        fn mul(self, rhs: DifferentialImplicitSystem<D>) -> Self::Output {
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
    impl<S: ImplicitSystem> ops::Sub<Vector3> for ImplicitScalarProduct<S> {
        type Output = ImplicitTermSum<ImplicitScalarProduct<S>, Vector3>;

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
    impl<S: ImplicitSystem> ops::Add<Vector3> for ImplicitScalarProduct<S> {
        type Output = ImplicitTermSum<ImplicitScalarProduct<S>, Vector3>;

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
            S1: ImplicitSystem<ImplicitValue = V>,
            S2: ImplicitSystem<ImplicitValue = V>,
        > ops::Sub<&'a fields::VolField<V>> for ImplicitTermSum<S1, S2>
    {
        type Output = ImplicitTermSum<
            ImplicitTermSum<S1, S2>,
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
    impl<
            V: ImplicitValue,
            S1: ImplicitSystem<ImplicitValue = V>,
            S2: ImplicitSystem<ImplicitValue = V>,
        > ops::Sub<ImplicitScalarProduct<S2>> for ImplicitScalarProduct<S1>
    {
        type Output = ImplicitTermSum<ImplicitScalarProduct<S1>, ImplicitScalarProduct<S2>>;

        fn sub(self, rhs: ImplicitScalarProduct<S2>) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: ImplicitScalarProduct {
                    scalar: -rhs.scalar,
                    term: rhs.term,
                },
            }
        }
    }
    impl<
            'a,
            V: ImplicitValue,
            S1: ImplicitSystem<ImplicitValue = V>,
            S2: ImplicitSystem<ImplicitValue = V>,
            D: DifferentialImplicitSystemImpl<ImplicitValue = V>,
        > ops::Sub<DifferentialImplicitSystem<D>> for ImplicitTermSum<S1, S2>
    {
        type Output = ImplicitTermSum<
            ImplicitTermSum<S1, S2>,
            ImplicitScalarProduct<DifferentialImplicitSystem<D>>,
        >;

        fn sub(self, rhs: DifferentialImplicitSystem<D>) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: ImplicitScalarProduct {
                    scalar: -1.,
                    term: rhs,
                },
            }
        }
    }
    impl<
            'a,
            V: ImplicitValue,
            S1: ImplicitSystem<ImplicitValue = V>,
            S2: ImplicitSystem<ImplicitValue = V>,
            D: DifferentialImplicitSystemImpl<ImplicitValue = V>,
        > ops::Add<DifferentialImplicitSystem<D>> for ImplicitTermSum<S1, S2>
    {
        type Output = ImplicitTermSum<ImplicitTermSum<S1, S2>, DifferentialImplicitSystem<D>>;

        fn add(self, rhs: DifferentialImplicitSystem<D>) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: rhs,
            }
        }
    }
    impl<
            'a,
            V: ImplicitValue,
            S1: ImplicitSystem<ImplicitValue = V>,
            S2: ImplicitSystem<ImplicitValue = V>,
        > ops::Add<V> for ImplicitTermSum<S1, S2>
    {
        type Output = ImplicitTermSum<ImplicitTermSum<S1, S2>, V>;

        fn add(self, rhs: V) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: rhs,
            }
        }
    }
    impl<
            'a,
            V: ImplicitValue,
            S1: ImplicitSystem<ImplicitValue = V>,
            S2: ImplicitSystem<ImplicitValue = V>,
        > ops::Div<Float> for ImplicitTermSum<S1, S2>
    {
        type Output = ImplicitScalarProduct<ImplicitTermSum<S1, S2>>;

        fn div(self, rhs: Float) -> Self::Output {
            ImplicitScalarProduct {
                scalar: 1. / rhs,
                term: self,
            }
        }
    }
    impl<'a, D: DifferentialImplicitSystemImpl> ops::Sub<&'a fields::VolField<Float>>
        for DifferentialImplicitSystem<D>
    {
        type Output = ImplicitTermSum<
            DifferentialImplicitSystem<D>,
            ImplicitScalarProduct<&'a fields::VolField<Float>>,
        >;

        fn sub(self, rhs: &'a fields::VolField<Float>) -> Self::Output {
            ImplicitTermSum {
                term_1: self,
                term_2: ImplicitScalarProduct {
                    scalar: -1.,
                    term: rhs,
                },
            }
        }
    }
    impl<'a, V: ImplicitValue> ops::Sub<&'a fields::VolField<V>> for ImplicitVolField<V> {
        type Output =
            ImplicitTermSum<ImplicitVolField<V>, ImplicitScalarProduct<&'a fields::VolField<V>>>;

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

#[derive(Debug)]
pub struct ImplicitSolver {
    pub linear_solver: linalg::LinearSolver,
    pub ignore_max_iters: bool,
}
impl ImplicitSolver {
    pub fn find_root<S: ImplicitSystem>(
        &self,
        implicit_system: S,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<S::ImplicitValue>,
        guess: Option<&fields::VolField<S::ImplicitValue>>,
    ) -> Result<fields::VolField<S::ImplicitValue>, ImplicitSolveError> {
        use indexing::Indexing;

        let cell_indexing = dynamic_geometry.grid().cell_indexing();

        let (matrix, lhs_constant) =
            gen_system(implicit_system, dynamic_geometry, boundary_conditions);

        let x = guess
            .map(|guess| guess.flatten(cell_indexing))
            .unwrap_or_else(|| Array1::zeros(matrix.shape().0));
        // Negate `lhs_constant` to produce `rhs`. Up to this point we've been assuming
        // it's a root-finding problem of the form `Ax + lhs_constant = 0`, but
        // the `LinearSolver` convention is `Ax = rhs`.
        Ok(fields::VolField::unflatten(
            cell_indexing,
            &self
                .linear_solver
                .solve(&matrix, x, -lhs_constant)
                .or_else(|err| match err {
                    linalg::LinearSolveError::NotDiagonallyDominant {
                        row_index,
                        abs_diag,
                        sum_abs_row,
                    } => Err(ImplicitSolveError::NotDiagonallyDominant {
                        cell_index: cell_indexing.unflatten(row_index),
                        abs_diag,
                        sum_abs_row,
                    }),
                    linalg::LinearSolveError::MaxItersReached {
                        iters,
                        rel_error,
                        abs_error,
                        rel_error_tol,
                        abs_error_tol,
                        solution,
                    } => {
                        if self.ignore_max_iters {
                            Ok(solution)
                        } else {
                            Err(ImplicitSolveError::MaxItersReached {
                                iters,
                                rel_error,
                                abs_error,
                                rel_error_tol,
                                abs_error_tol,
                            })
                        }
                    }
                    linalg::LinearSolveError::SingularMatrix => {
                        Err(ImplicitSolveError::SingularSystem)
                    }
                })?,
        ))
    }
}

fn gen_system<S: ImplicitSystem>(
    implicit_system: S,
    dynamic_geometry: &geom::DynamicGeometry,
    boundary_conditions: &fields::BoundaryConditions<S::ImplicitValue>,
) -> (sprs::TriMat<Float>, Array1) {
    use fields::Value;
    use indexing::{Indexing, IntoIndexIterator};

    let cell_indexing = dynamic_geometry.grid().cell_indexing();
    let value_size = S::ImplicitValue::size();
    let matrix_size = cell_indexing.len() * value_size;

    let mut matrix = sprs::TriMat::new((matrix_size, matrix_size));
    let mut lhs_constant = Array1::zeros(matrix_size);
    for (flat_index, cell_index) in cell_indexing.iter().enumerate() {
        let mut add_cell_coef = |CellCoef {
                                     cell_index: col_cell_index,
                                     coef,
                                 }| {
            let col_flat_index = cell_indexing.flatten(col_cell_index);
            for i in 0..value_size {
                matrix.add_triplet(
                    flat_index * value_size + i,
                    col_flat_index * value_size + i,
                    coef,
                );
            }
        };
        let subsystem_lhs_constant = implicit_system.gen_subsystem(
            &mut add_cell_coef,
            cell_index,
            dynamic_geometry,
            boundary_conditions,
        );
        subsystem_lhs_constant.flatten(lhs_constant.slice_mut(nd::s![
            flat_index * value_size..(flat_index + 1) * value_size
        ]));
    }

    (matrix, lhs_constant)
}

#[cfg(test)]
pub fn evaluate<S: ImplicitSystem>(
    implicit_system: S,
    dynamic_geometry: &geom::DynamicGeometry,
    boundary_conditions: &fields::BoundaryConditions<S::ImplicitValue>,
    field: &fields::VolField<S::ImplicitValue>,
) -> fields::VolField<S::ImplicitValue> {
    let (matrix, lhs_constant) = gen_system(implicit_system, dynamic_geometry, boundary_conditions);
    let matrix: sprs::CsMat<Float> = matrix.to_csr();
    let x = field.flatten(dynamic_geometry.grid().cell_indexing());
    fields::VolField::unflatten(
        dynamic_geometry.grid().cell_indexing(),
        &(&matrix * &x + lhs_constant),
    )
}

#[derive(Debug)]
pub enum ImplicitSolveError {
    NotDiagonallyDominant {
        cell_index: indexing::CellIndex,
        abs_diag: Float,
        sum_abs_row: Float,
    },
    MaxItersReached {
        rel_error: Option<Float>,
        abs_error: Float,
        rel_error_tol: Float,
        abs_error_tol: Float,
        iters: usize,
    },
    SingularSystem,
}

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
            ImplicitSolver {
                linear_solver: linalg::LinearSolver::Direct,
                ignore_max_iters: false,
            },
            ImplicitSolver {
                linear_solver: linalg::LinearSolver::GaussSeidel {
                    max_iters: 20,
                    rel_error_tol: 1e-7,
                    abs_error_tol: 1e-7,
                },
                ignore_max_iters: false,
            },
        ] {
            {
                let x_sol = solver
                    .find_root(
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
                    .find_root(
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
                    .find_root(
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
                    .find_root(
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
                    .find_root(
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

    /// Ensure matrix terms sum appropriately.
    #[test]
    fn test_matrices_sum() {
        let mut matrix = sprs::TriMat::<Float>::new((2, 2));
        matrix.add_triplet(0, 0, 1.);
        matrix.add_triplet(0, 0, -2.);
        let matrix: sprs::CsMat<Float> = matrix.to_csc();
        approx::assert_relative_eq!(matrix.to_dense()[[0, 0]], -1.);
    }
}
