use pyo3::prelude::*;

use crate::{
    fields, geom, implicit,
    indexing::{self, IntoIndexIterator},
    linalg, Float, Vector2, Vector3,
};

#[derive(Clone)]
pub struct Fields {
    pub height: fields::AreaScalarField,
    pub velocity: fields::VolVectorField,
    pub pressure: fields::VolScalarField,
    pub pressure_grad: fields::VolVectorField,
    pub hydrostatic_column_pressure: fields::VolScalarField,
}

pub struct Problem {
    pub rain_rate: Option<fields::AreaScalarField>,

    pub grav_accel: Float,
    pub kinematic_viscosity: Float,

    horiz_2d_velocity_boundary_conditions: fields::HorizBoundaryConditions<Vector2>,
    horiz_velocity_boundary_conditions: fields::HorizBoundaryConditions<Vector3>,
    pub pressure_horiz_boundary_conditions: fields::HorizBoundaryConditions<Float>,
    pub height_boundary_conditions: fields::HorizBoundaryConditions<Float>,
}
impl Default for Problem {
    fn default() -> Self {
        Self {
            rain_rate: None,
            grav_accel: 9.8,
            kinematic_viscosity: 1e-6,

            horiz_velocity_boundary_conditions: fields::HorizBoundaryConditions::hom_dirichlet(),
            horiz_2d_velocity_boundary_conditions: fields::HorizBoundaryConditions::hom_dirichlet(),
            pressure_horiz_boundary_conditions: fields::HorizBoundaryConditions::hom_neumann(),
            height_boundary_conditions: fields::HorizBoundaryConditions::hom_neumann(),
        }
    }
}
impl Problem {
    pub fn make_pressure_solver(&self) -> implicit::ImplicitSolver {
        let rel_error_tol = 0.1;
        implicit::ImplicitSolver {
            linear_solver: linalg::LinearSolver::GaussSeidel {
                max_iters: 100,
                abs_error_tol: rel_error_tol * self.grav_accel,
                rel_error_tol,
            },
            ignore_max_iters: false,
        }
    }
}

#[pyclass]
pub struct Solver {
    problem: Problem,
    dynamic_geometry: Option<geom::DynamicGeometry>,
    fields: Fields,

    pressure_solver: implicit::ImplicitSolver,
}
impl Solver {
    pub fn new(
        problem: Problem,
        pressure_solver: implicit::ImplicitSolver,
        initial_dynamic_geometry: geom::DynamicGeometry,
        initial_height: fields::AreaScalarField,
        initial_velocity: fields::VolVectorField,
    ) -> Self {
        let pressure_boundary_conditions =
            compute_pressure_boundary_conditions(&problem, &initial_dynamic_geometry);
        let velocity_boundary_conditions = compute_velocity_boundary_conditions(
            &problem,
            &fields::AreaScalarField::zeros(
                initial_dynamic_geometry.grid().cell_footprint_indexing(),
            ),
        );

        let hydrostatic_column_pressure =
            compute_hydrostatic_column_pressure(&problem, &initial_dynamic_geometry);
        let pressure = Self::compute_initial_pressure(
            &initial_dynamic_geometry,
            &pressure_boundary_conditions,
            &pressure_solver,
            &initial_velocity.gradient(&initial_dynamic_geometry, &velocity_boundary_conditions),
            Some(
                &hydrostatic_column_pressure
                    .gradient(&initial_dynamic_geometry, &pressure_boundary_conditions),
            ),
            Some(&hydrostatic_column_pressure),
        );
        let pressure_grad =
            pressure.gradient(&initial_dynamic_geometry, &pressure_boundary_conditions);
        Self {
            problem,
            pressure_solver,
            dynamic_geometry: Some(initial_dynamic_geometry),
            fields: Fields {
                height: initial_height,
                velocity: initial_velocity,
                pressure,
                pressure_grad,
                hydrostatic_column_pressure,
            },
        }
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn problem_mut(&mut self) -> &mut Problem {
        &mut self.problem
    }

    pub fn dynamic_geometry(&self) -> &geom::DynamicGeometry {
        self.dynamic_geometry.as_ref().unwrap()
    }

    pub fn fields(&self) -> &Fields {
        &self.fields
    }

    pub fn step(&mut self, dt: Float) {
        let dhdt = compute_height_time_deriv(
            self.dynamic_geometry.as_ref().unwrap(),
            self.problem.rain_rate.as_ref(),
            &self.fields,
            self.problem.horiz_2d_velocity_boundary_conditions,
            self.problem.height_boundary_conditions,
        );
        for cell_footprint_index in self
            .dynamic_geometry
            .as_ref()
            .unwrap()
            .grid()
            .cell_footprint_indexing()
            .iter()
        {
            let old_h = self
                .fields
                .height
                .cell_footprint_value(cell_footprint_index);
            let new_h = (self
                .fields
                .height
                .cell_footprint_value(cell_footprint_index)
                + dt * dhdt.cell_footprint_value(cell_footprint_index))
            .max(0.);
            // Attempt to conserve momentum.
            for cell_index in self
                .dynamic_geometry
                .as_ref()
                .unwrap()
                .grid()
                .cell_indexing()
                .column(cell_footprint_index)
            {
                *self.fields.velocity.cell_value_mut(cell_index) *= old_h / new_h;
            }
            *self
                .fields
                .height
                .cell_footprint_value_mut(cell_footprint_index) = new_h;
        }

        self.dynamic_geometry = Some(geom::DynamicGeometry::new(
            self.dynamic_geometry.take().unwrap().into_static_geometry(),
            &self.fields.height,
        ));
        let dynamic_geometry = self.dynamic_geometry.as_ref().unwrap();

        let velocity_boundary_conditions =
            compute_velocity_boundary_conditions(&self.problem, &dhdt);
        //  compute_velocity_boundary_conditions(
        //     &self.problem,
        //     &fields::AreaScalarField::zeros(
        //         self.dynamic_geometry
        //             .as_ref()
        //             .unwrap()
        //             .grid()
        //             .cell_footprint_indexing(),
        //     ),
        // );

        // Interpolate velocity to new height map. TODO
        // fields.velocity =
        //     interpolate_onto(&z_lattice.centers, &fields.velocity,
        // &new_z_axis.centers);

        // Compute new pressure field.
        let pressure_boundary_conditions =
            compute_pressure_boundary_conditions(&self.problem, dynamic_geometry);
        self.fields.hydrostatic_column_pressure =
            compute_hydrostatic_column_pressure(&self.problem, dynamic_geometry);
        let shear = self
            .fields
            .velocity
            .gradient(dynamic_geometry, &velocity_boundary_conditions);

        // Perform velocity update.
        let mut dvdt = self.problem.kinematic_viscosity
            * &self.fields.velocity.laplacian(
                &dynamic_geometry,
                &shear,
                &velocity_boundary_conditions,
            );
        dvdt -= crate::Vector3::new(0., 0., self.problem.grav_accel);
        dvdt -= &self
            .fields
            .velocity
            .advect_upwind(dynamic_geometry, &velocity_boundary_conditions);
        self.fields.velocity += &(dt * &dvdt);

        // self.fields.pressure = Self::compute_initial_pressure(
        //     &dynamic_geometry,
        //     &pressure_boundary_conditions,
        //     &self.pressure_solver,
        //     &self
        //         .fields
        //         .velocity
        //         .gradient(&dynamic_geometry, &velocity_boundary_conditions),
        //     Some(&self.fields.pressure_grad),
        //     Some(&self.fields.pressure),
        // );
        self.fields.pressure = Self::compute_incremental_pressure(
            &dynamic_geometry,
            &pressure_boundary_conditions,
            &self.pressure_solver,
            &self
                .fields
                .velocity
                .divergence(&dynamic_geometry, &velocity_boundary_conditions),
            dt,
            &self.fields.pressure_grad,
            &self.fields.pressure,
        );
        self.fields.pressure_grad = self
            .fields
            .pressure
            .gradient(&dynamic_geometry, &pressure_boundary_conditions);
        self.fields.velocity -= &(dt * &self.fields.pressure_grad);
    }

    fn compute_incremental_pressure(
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<Float>,
        implicit_solver: &implicit::ImplicitSolver,
        velocity_divergence: &fields::VolScalarField,
        dt: Float,
        pressure_grad: &fields::VolVectorField,
        prev_pressure: &fields::VolScalarField,
    ) -> fields::VolScalarField {
        let p = implicit::ImplicitVolField::default();
        let rhs = 1. / dt * velocity_divergence;
        let pressure_system = p.laplacian(Some(&pressure_grad)) - &rhs;

        implicit_solver
            .find_root(
                pressure_system,
                &dynamic_geometry,
                &boundary_conditions,
                Some(prev_pressure),
            )
            .unwrap()
    }

    fn compute_initial_pressure(
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<Float>,
        implicit_solver: &implicit::ImplicitSolver,
        shear: &fields::VolTensorField,
        pressure_grad: Option<&fields::VolVectorField>,
        prev_pressure: Option<&fields::VolScalarField>,
    ) -> fields::VolScalarField {
        let p = implicit::ImplicitVolField::default();
        let s = shear.map(|shear| -(shear.component_mul(&shear.transpose())).sum());
        let pressure_system = p.laplacian(pressure_grad) - &s;

        implicit_solver
            .find_root(
                pressure_system,
                &dynamic_geometry,
                &boundary_conditions,
                prev_pressure,
            )
            .unwrap()
    }
}
#[pymethods]
impl Solver {
    #[getter]
    pub fn grid(&self) -> geom::Grid {
        self.dynamic_geometry.as_ref().unwrap().grid().clone()
    }

    #[getter]
    #[pyo3(name = "z_lattice")]
    pub fn z_lattice_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.dynamic_geometry.as_ref().unwrap().z_lattice_py(py)
    }

    #[getter]
    #[pyo3(name = "pressure")]
    pub fn pressure_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray4<Float> {
        self.fields.pressure.values_py(py)
    }

    #[getter]
    #[pyo3(name = "velocity")]
    pub fn velocity_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray5<Float> {
        self.fields.velocity.values_py(py)
    }

    #[getter]
    #[pyo3(name = "hydrostatic_column_pressure")]
    pub fn hydrostatic_pressure_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray4<Float> {
        self.fields.hydrostatic_column_pressure.values_py(py)
    }

    #[pyo3(name = "step")]
    pub fn step_py(&mut self, dt: Float) {
        self.step(dt)
    }

    #[getter]
    #[pyo3(name = "courant_dt")]
    pub fn courant_dt_py(&self) -> Float {
        let mut t = Float::MAX;
        let dynamic_geometry = self.dynamic_geometry.as_ref().unwrap();
        for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
            let v = self.fields.velocity.cell_value(cell_index);
            t = t.min(dynamic_geometry.grid().x_axis().spacing() / (v.x.abs() + 1e-10));
            t = t.min(dynamic_geometry.grid().y_axis().spacing() / (v.y.abs() + 1e-10));
            t = t.min(
                self.fields
                    .height
                    .cell_footprint_value(cell_index.footprint)
                    / (dynamic_geometry.grid().cell_indexing().num_z_cells() as Float)
                    / (v.z.abs() + 1e-10),
            );
        }
        t
    }
}

/// Linearly interpolate a velocity vector field array `v` with vertical
/// coordinate array `old_height` onto a new vertical coordinate array
/// `new_height`. Use constant extrapolation if a value of `new_height` falls
/// outside the range of `old_height`.
// fn interpolate_onto(
//     dynamic_geometry: &geom::DynamicGeometry,
//     old_height: &fields::HorizScalarField,
//     new_height: &fields::HorizScalarField,
//     velocity: &fields::VectorField,
// ) -> fields::VectorField {
//     for cell_footprint_index in
// dynamic_geometry.grid().cell_footprint_indexing().iter() {

//     }
// }
// fn interpolate_onto(
//     old_height: &fields::HorizScalarField,
//     v: &fields::VectorField,
//     new_height: &fields::HorizScalarField,
// ) -> fields::VectorField {
//     let mut new_v =
//         fields::VectorField::zeros((v.dim().0, v.dim().1, new_height.dim().2,
// v.dim().3));     let dim = old_height.dim();
//     for i in 0..dim.0 {
//         for j in 0..dim.1 {
//             let zij = old_height.slice(s![i, j, ..]);
//             let vij = v.slice(s![i, j, .., ..]);
//             let new_zij = new_height.slice(s![i, j, ..]);
//             let mut new_vij = new_v.slice_mut(s![i, j, .., ..]);

//             let mut k = 0usize;
//             for (new_k, &new_zijk) in new_zij.iter().enumerate() {
//                 while k < dim.2 && zij[k] < new_zijk {
//                     k += 1;
//                 }
//                 new_vij.slice_mut(s![new_k, ..]).assign(&if k == 0 {
//                     vij.slice(s![0, ..]).into_owned()
//                 } else if k == dim.2 {
//                     vij.slice(s![dim.2 - 1, ..]).into_owned()
//                 } else {
//                     //    <---------delta_zijk--------->
//                     //    |          |                 |
//                     // zij_left  new_zijk          zij_right
//                     let zij_left = zij[k - 1];
//                     let zij_right = zij[k];
//                     let delta_zijk = zij_right - zij_left;
//                     let alpha = if delta_zijk == 0. {
//                         0.
//                     } else {
//                         (new_zijk - zij_left) / delta_zijk
//                     };
//                     (1. - alpha) * &vij.slice(s![k - 1, ..]) + alpha *
// &vij.slice(s![k, ..])                 });
//             }
//         }
//     }
//     new_v
// }

pub fn compute_height_time_deriv(
    dynamic_geometry: &geom::DynamicGeometry,
    rain_rate: Option<&fields::AreaScalarField>,
    fields: &Fields,
    velocity_boundary_conditions: fields::HorizBoundaryConditions<Vector2>,
    height_boundary_conditions: fields::HorizBoundaryConditions<Float>,
) -> fields::AreaScalarField {
    let mut dhdt = fields
        .velocity
        .column_average(dynamic_geometry)
        .advect_upwind(
            &fields.height,
            dynamic_geometry.grid(),
            velocity_boundary_conditions,
            height_boundary_conditions,
        );

    if let Some(rain_rate) = rain_rate {
        for cell_footprint_index in dynamic_geometry.grid().cell_footprint_indexing().iter() {
            *dhdt.cell_footprint_value_mut(cell_footprint_index) +=
                rain_rate.cell_footprint_value(cell_footprint_index);
        }
    }

    dhdt
}

// #[derive(Debug)]
// pub struct VelocitySolver {
//     linear_solver: linalg::LinearSolver,
// }
// impl VelocitySolver {
//     pub fn solve(
//         &self,
//         problem: &Problem,
//         dynamic_geometry: &geom::DynamicGeometry,
//         boundary_conditions: &fields::BoundaryConditions<Vector3>,
//         prev_velocity: &fields::VolVectorField,
//         prev_pressure_grad: &fields::VolVectorField,
//         dt: Float,
//     ) -> fields::VolVectorField {
//         use implicit::SolveEq;

//         let prev_u = prev_velocity;
//         let prev_grad_p = prev_pressure_grad;
//         let nu = problem.kinematic_viscosity;
//         let rho = problem.density;
//         let g = Vector3::new(0., 0., -problem.grav_accel);

//         let u = implicit::ImplicitVolField::<Float>::new(dynamic_geometry,
// boundary_conditions);         let eq =
//             (u - prev_u) / dt + prev_u.advect(u) - nu * u.laplacian() ==
// -prev_grad_p / rho + g;         self.linear_solver.solve_eq(eq, u)
//     }
// }

fn compute_velocity_boundary_conditions(
    problem: &Problem,
    dhdt: &fields::AreaScalarField,
) -> fields::BoundaryConditions<Vector3> {
    fields::BoundaryConditions {
        horiz: problem.horiz_velocity_boundary_conditions.clone(),
        z: fields::VertBoundaryFieldPair {
            lower: fields::VertBoundaryField::HomDirichlet,
            upper: fields::VertBoundaryField::Kinematic(dhdt.clone()),
            // upper: fields::VertBoundaryField::HomDirichlet,
        },
    }
}

fn compute_pressure_boundary_conditions(
    problem: &Problem,
    dynamic_geometry: &geom::DynamicGeometry,
) -> fields::BoundaryConditions<Float> {
    let cell_footprint_indexing = dynamic_geometry.grid().cell_footprint_indexing();
    let mut lower_boundary_field = fields::AreaField::zeros(cell_footprint_indexing);
    for cell_footprint_index in cell_footprint_indexing.iter() {
        *lower_boundary_field.cell_footprint_value_mut(cell_footprint_index) = -problem.grav_accel
            * dynamic_geometry
                .cell(indexing::CellIndex {
                    footprint: cell_footprint_index,
                    z: 0,
                })
                .lower_z_face
                .outward_normal()
                .z;
    }
    fields::BoundaryConditions {
        horiz: problem.pressure_horiz_boundary_conditions.clone(),
        z: fields::VertBoundaryFieldPair {
            lower: fields::VertBoundaryField::InhomNeumann(lower_boundary_field),
            upper: fields::VertBoundaryField::HomDirichlet,
        },
    }
}

fn compute_hydrostatic_column_pressure(
    problem: &Problem,
    dynamic_geometry: &geom::DynamicGeometry,
) -> fields::VolScalarField {
    let mut hydrostatic_column_pressure =
        fields::VolScalarField::zeros(dynamic_geometry.grid().cell_indexing());
    for cell_footprint in dynamic_geometry.grid().cell_footprint_indexing().iter() {
        let surface_z = dynamic_geometry
            .cell(indexing::CellIndex {
                footprint: cell_footprint,
                z: dynamic_geometry.grid().cell_indexing().num_z_cells() - 1,
            })
            .upper_z_face
            .centroid()
            .z;
        for cell_index in dynamic_geometry
            .grid()
            .cell_indexing()
            .column(cell_footprint)
        {
            let z = dynamic_geometry.cell(cell_index).centroid.z;
            *hydrostatic_column_pressure.cell_value_mut(cell_index) =
                problem.grav_accel * (surface_z - z);
        }
    }
    hydrostatic_column_pressure
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3;

    #[test]
    fn test_pressure_solver() {
        let x_axis = geom::Axis::new(-1., 1., 3);
        let y_axis = geom::Axis::new(10., 11., 4);
        let grid = geom::Grid::new(x_axis, y_axis, 50);
        let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

        let height = fields::AreaScalarField::new(static_geometry.grid(), |_, _| 10.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let problem = Problem::default();

        let hydrostatic_column_pressure =
            compute_hydrostatic_column_pressure(&problem, &dynamic_geometry);
        let boundary_conditions = compute_pressure_boundary_conditions(&problem, &dynamic_geometry);
        let mut pressure_solver = problem.make_pressure_solver();
        match &mut pressure_solver.linear_solver {
            linalg::LinearSolver::GaussSeidel {
                abs_error_tol,
                rel_error_tol,
                max_iters,
            } => {
                *abs_error_tol /= 1000.;
                *rel_error_tol /= 1000.;
                *max_iters = 100000;
            }
            linalg::LinearSolver::Direct => {}
        };
        pressure_solver.ignore_max_iters = false;

        let pressure = Solver::compute_initial_pressure(
            &dynamic_geometry,
            &boundary_conditions,
            &pressure_solver,
            &fields::VolTensorField::zeros(dynamic_geometry.grid().cell_indexing()),
            None,
            None,
        );
        hydrostatic_column_pressure
            .assert_all_close(&pressure, &dynamic_geometry)
            .rel_tol(Some(1e-1));

        let pressure = Solver::compute_initial_pressure(
            &dynamic_geometry,
            &boundary_conditions,
            &pressure_solver,
            &fields::VolTensorField::zeros(dynamic_geometry.grid().cell_indexing()),
            Some(&pressure.gradient(&dynamic_geometry, &boundary_conditions)),
            Some(&hydrostatic_column_pressure),
        );
        hydrostatic_column_pressure.assert_all_close(&pressure, &dynamic_geometry);
    }

    #[test]
    fn test_hydrostatic_column_pressure() {
        let x_axis = geom::Axis::new(-1., 1., 3);
        let y_axis = geom::Axis::new(10., 11., 4);
        let grid = geom::Grid::new(x_axis, y_axis, 50);
        let static_geometry = geom::StaticGeometry::new(grid, |_, _| 0.);

        let height = fields::AreaScalarField::new(static_geometry.grid(), |_, _| 10.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let problem = Problem::default();

        let pressure = compute_hydrostatic_column_pressure(&problem, &dynamic_geometry);
        let boundary_conditions = compute_pressure_boundary_conditions(&problem, &dynamic_geometry);
        let pressure_gradient = pressure.gradient(&dynamic_geometry, &boundary_conditions);

        for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
            let pressure_gradient_value = pressure_gradient.cell_value(cell_index);
            if !approx::relative_eq!(
                pressure_gradient_value,
                Vector3::new(0., 0., -problem.grav_accel),
                epsilon = 0.01,
                max_relative = 1e-4,
            ) {
                panic!("{cell_index:?}:{pressure_gradient_value}");
            }
        }
    }

    #[test]
    fn test_rain() {
        let x_axis = geom::Axis::new(-1., 1., 3);
        let y_axis = geom::Axis::new(10., 11., 4);
        let grid = geom::Grid::new(x_axis, y_axis, 5);
        let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

        let height = fields::AreaScalarField::new(static_geometry.grid(), |_, _| 3.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let fields = Fields {
            height: height.clone(),
            velocity: fields::VolVectorField::zeros(dynamic_geometry.grid().cell_indexing()),
            pressure: fields::VolScalarField::zeros(dynamic_geometry.grid().cell_indexing()),
            pressure_grad: fields::VolVectorField::zeros(dynamic_geometry.grid().cell_indexing()),
            hydrostatic_column_pressure: fields::VolScalarField::zeros(
                dynamic_geometry.grid().cell_indexing(),
            ),
        };

        // No rain.
        {
            let rain_rate =
                fields::AreaScalarField::zeros(dynamic_geometry.grid().cell_footprint_indexing());

            let dhdt = compute_height_time_deriv(
                &dynamic_geometry,
                Some(&rain_rate),
                &fields,
                fields::HorizBoundaryConditions::hom_dirichlet(),
                fields::HorizBoundaryConditions::hom_dirichlet(),
            );

            approx::assert_abs_diff_eq!(dhdt, rain_rate);
        }
        // Some rain.
        {
            let rain_rate = fields::AreaScalarField::new(dynamic_geometry.grid(), |_, _| 1.5e-2);

            let dhdt = compute_height_time_deriv(
                &dynamic_geometry,
                Some(&rain_rate),
                &fields,
                fields::HorizBoundaryConditions::hom_dirichlet(),
                fields::HorizBoundaryConditions::hom_dirichlet(),
            );

            approx::assert_abs_diff_eq!(dhdt, rain_rate);
        }
    }
}
