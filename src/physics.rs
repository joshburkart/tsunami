use pyo3::prelude::*;

use crate::{
    fields, geom, implicit,
    indexing::{self, IntoIndexIterator},
    linalg, Float, Vector3,
};

#[derive(Clone)]
pub struct Fields {
    pub height_time_deriv: fields::AreaScalarField,

    pub volume: fields::VolScalarField,
    pub volume_time_deriv: fields::VolScalarField,

    pub velocity: fields::VolVectorField,
    pub velocity_divergence: fields::VolScalarField,

    pub pressure: fields::VolScalarField,
    pub pressure_grad: fields::VolVectorField,
    pub hydrostatic_column_pressure: fields::VolScalarField,
}

pub struct Problem {
    pub rain_rate: Option<fields::AreaScalarField>,

    pub grav_accel: Float,
    pub kinematic_viscosity: Float,

    pub horiz_velocity_boundary_conditions: fields::HorizBoundaryConditions<Vector3>,
    pub pressure_horiz_boundary_conditions: fields::HorizBoundaryConditions<Float>,
}
impl Default for Problem {
    fn default() -> Self {
        Self {
            rain_rate: None,
            grav_accel: 9.8,
            kinematic_viscosity: 1e-6,

            horiz_velocity_boundary_conditions: fields::HorizBoundaryConditions {
                x: fields::HorizBoundaryConditionPair::hom_neumann(),
                y: fields::HorizBoundaryConditionPair::hom_neumann(),
            },
            pressure_horiz_boundary_conditions: fields::HorizBoundaryConditions::hom_neumann(),
        }
    }
}
impl Problem {
    pub fn make_implicit_solver(&self) -> implicit::ImplicitSolver {
        implicit::ImplicitSolver {
            linear_solver: linalg::LinearSolver::Klu,
            ignore_max_iters: true,
        }
        // let rel_error_tol = 0.001;
        // implicit::ImplicitSolver {
        //     linear_solver: linalg::LinearSolver::GaussSeidel {
        //         max_iters: 10000,
        //         abs_error_tol: rel_error_tol * self.grav_accel,
        //         rel_error_tol,
        //     },
        //     ignore_max_iters: false,
        // }
    }
}

#[pyclass]
pub struct Solver {
    problem: Problem,
    dynamic_geometry: Option<geom::DynamicGeometry>,
    fields: Fields,

    implicit_solver: implicit::ImplicitSolver,
}
impl Solver {
    pub fn new(
        problem: Problem,
        implicit_solver: implicit::ImplicitSolver,
        initial_dynamic_geometry: geom::DynamicGeometry,
        initial_height: fields::AreaScalarField,
        initial_velocity: fields::VolVectorField,
    ) -> Self {
        let grid = initial_dynamic_geometry.grid();

        let volume = compute_initial_volume(initial_dynamic_geometry.grid(), &initial_height);

        let pressure_boundary_conditions =
            compute_pressure_boundary_conditions(&problem, &initial_dynamic_geometry);
        let height_time_deriv = fields::AreaScalarField::zeros(grid.cell_footprint_indexing());
        let velocity_boundary_conditions =
            compute_velocity_boundary_conditions(&problem, &height_time_deriv);

        let hydrostatic_column_pressure =
            compute_hydrostatic_column_pressure(&problem, &initial_dynamic_geometry);
        let pressure = Self::compute_initial_pressure(
            &initial_dynamic_geometry,
            &pressure_boundary_conditions,
            &implicit_solver,
            &initial_velocity.gradient(&initial_dynamic_geometry, &velocity_boundary_conditions),
            Some(
                &hydrostatic_column_pressure
                    .gradient(&initial_dynamic_geometry, &pressure_boundary_conditions),
            ),
            Some(&hydrostatic_column_pressure),
        );
        let pressure_grad =
            pressure.gradient(&initial_dynamic_geometry, &pressure_boundary_conditions);
        let velocity_divergence =
            initial_velocity.divergence(&initial_dynamic_geometry, &velocity_boundary_conditions);
        let volume_time_deriv =
            fields::VolScalarField::zeros(initial_dynamic_geometry.grid().cell_indexing());
        Self {
            problem,
            implicit_solver,
            dynamic_geometry: Some(initial_dynamic_geometry),
            fields: Fields {
                height_time_deriv,
                volume,
                volume_time_deriv,
                velocity: initial_velocity,
                velocity_divergence,
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
        // Determine cell volume time derivatives.
        let advection_velocity_boundary_conditions = compute_velocity_boundary_conditions(
            &self.problem,
            &fields::AreaScalarField::zeros(
                self.dynamic_geometry
                    .as_ref()
                    .unwrap()
                    .grid()
                    .cell_footprint_indexing(),
            ),
        );
        let advection_velocity = self
            .fields
            .velocity
            .map(|velocity| Vector3::new(velocity.x, velocity.y, 0.));

        let advection_velocity_divergence = advection_velocity.divergence(
            self.dynamic_geometry(),
            &advection_velocity_boundary_conditions,
        );
        let volume_time_deriv = -(&self.fields.volume * advection_velocity_divergence.clone());

        // Update geometry.
        {
            // Evolve cell volumes.
            let grid = self.dynamic_geometry.as_ref().unwrap().grid();
            let cell_indexing = grid.cell_indexing();
            for cell_index in cell_indexing.iter() {
                let cell_volume = self.fields.volume.cell_value(cell_index);
                let new_cell_volume = (cell_volume + dt * volume_time_deriv.cell_value(cell_index))
                    .max(geom::MIN_VOLUME);
                *self.fields.volume.cell_value_mut(cell_index) = new_cell_volume;
            }

            // Compute column height time derivatives.
            for cell_footprint_index in grid.cell_footprint_indexing().iter() {
                let mut column_volume_time_deriv = 0.;
                for cell_index in grid.cell_indexing().column(cell_footprint_index) {
                    column_volume_time_deriv += volume_time_deriv.cell_value(cell_index);
                }
                *self
                    .fields
                    .height_time_deriv
                    .cell_footprint_value_mut(cell_footprint_index) =
                    column_volume_time_deriv / grid.footprint_area();
            }

            // Update dynamic geometry.
            self.dynamic_geometry = Some(geom::DynamicGeometry::new_from_volume(
                self.dynamic_geometry.take().unwrap().into_static_geometry(),
                &self.fields.volume,
            ));
            self.fields.volume_time_deriv = volume_time_deriv;
        }

        let dynamic_geometry = self.dynamic_geometry.as_ref().unwrap();
        let velocity_boundary_conditions =
            compute_velocity_boundary_conditions(&self.problem, &self.fields.height_time_deriv);
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
        let pressure_boundary_conditions =
            compute_pressure_boundary_conditions(&self.problem, dynamic_geometry);
        self.fields.hydrostatic_column_pressure =
            compute_hydrostatic_column_pressure(&self.problem, dynamic_geometry);
        let shear = self
            .fields
            .velocity
            .gradient(dynamic_geometry, &velocity_boundary_conditions);

        // // Perform explicit velocity update.
        // {
        //     let mut dvdt = self.problem.kinematic_viscosity
        //         * &self.fields.velocity.laplacian(
        //             &dynamic_geometry,
        //             &shear,
        //             &velocity_boundary_conditions,
        //         );
        //     dvdt -= &self.fields.velocity.advect_upwind(
        //         &advection_velocity,
        //         &dynamic_geometry,
        //         &velocity_boundary_conditions,
        //     );
        //     for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
        //         *dvdt.cell_value_mut(cell_index) += advection_velocity_divergence
        //             .cell_value(cell_index)
        //             * self.fields.velocity.cell_value(cell_index);
        //     }
        //     dvdt -= crate::Vector3::new(0., 0., self.problem.grav_accel);
        //     self.fields.velocity += &(dt * &dvdt);
        // }
        // Perform implicit velocity update.
        {
            let implicit_velocity = implicit::ImplicitVolField::<Vector3>::default();
            let system = (implicit_velocity - &self.fields.velocity) / dt
                - self.problem.kinematic_viscosity * implicit_velocity.laplacian(Some(&shear))
                + implicit_velocity.advect_upwind(&advection_velocity)
                - implicit_velocity * &advection_velocity_divergence
                + crate::Vector3::new(0., 0., self.problem.grav_accel);

            self.fields.velocity = self
                .implicit_solver
                .find_root(
                    system,
                    dynamic_geometry,
                    &velocity_boundary_conditions,
                    Some(&self.fields.velocity),
                )
                .unwrap();
        }

        // Perform pressure correction.
        {
            self.fields.pressure = Self::compute_incremental_pressure(
                &dynamic_geometry,
                &pressure_boundary_conditions,
                &self.implicit_solver,
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

        // // Run follow-up iterations of pressure correction.
        // let mut prev_pressure = self.fields.pressure.clone();
        // let mut prev_pressure_grad = self.fields.pressure_grad.clone();
        // for _ in 0..5 {
        //     let pressure_corrector = Self::compute_incremental_pressure(
        //         &dynamic_geometry,
        //         &pressure_boundary_conditions,
        //         &self.implicit_solver,
        //         &self
        //             .fields
        //             .velocity
        //             .divergence(&dynamic_geometry, &velocity_boundary_conditions),
        //         dt,
        //         &prev_pressure_grad,
        //         &prev_pressure,
        //     );
        //     let pressure_corrector_grad =
        //         pressure_corrector.gradient(&dynamic_geometry, &pressure_boundary_conditions);
        //     prev_pressure = pressure_corrector;
        //     prev_pressure_grad = pressure_corrector_grad;

        //     self.fields.velocity -= &(dt * &prev_pressure_grad);
        // }

        self.fields.velocity_divergence = self
            .fields
            .velocity
            .divergence(&dynamic_geometry, &velocity_boundary_conditions);
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
        let p = implicit::ImplicitVolField::<Float>::default();
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
        let p = implicit::ImplicitVolField::<Float>::default();
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
    #[pyo3(name = "volume")]
    pub fn volume_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray4<Float> {
        self.fields.volume.values_py(py)
    }
    #[getter]
    #[pyo3(name = "volume_time_deriv")]
    pub fn volume_time_deriv_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray4<Float> {
        self.fields.volume_time_deriv.values_py(py)
    }

    #[getter]
    #[pyo3(name = "height_time_deriv")]
    pub fn height_time_deriv_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.fields.height_time_deriv.values_py(py)
    }

    #[getter]
    #[pyo3(name = "velocity_divergence")]
    pub fn velocity_divergence_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray4<Float> {
        self.fields.velocity_divergence.values_py(py)
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
                self.fields.volume.cell_value(cell_index)
                    / dynamic_geometry.grid().footprint_area()
                    / (v.z.abs() + 1e-10),
            );
        }
        t
    }
}

fn compute_velocity_boundary_conditions(
    problem: &Problem,
    height_time_deriv: &fields::AreaScalarField,
) -> fields::BoundaryConditions<Vector3> {
    fields::BoundaryConditions {
        horiz: problem.horiz_velocity_boundary_conditions.clone(),
        z: fields::VertBoundaryFieldPair {
            lower: fields::VertBoundaryField::HomDirichlet,
            upper: fields::VertBoundaryField::Kinematic(height_time_deriv.clone()),
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

fn compute_initial_volume(
    grid: &geom::Grid,
    height: &fields::AreaScalarField,
) -> fields::VolScalarField {
    let mut volume = fields::VolScalarField::zeros(grid.cell_indexing());
    for cell_footprint_index in grid.cell_footprint_indexing().iter() {
        let column_volume =
            height.cell_footprint_value(cell_footprint_index) * grid.footprint_area();
        let cell_volume = column_volume / grid.cell_indexing().num_z_cells() as Float;
        for cell_index in grid.cell_indexing().column(cell_footprint_index) {
            *volume.cell_value_mut(cell_index) = cell_volume;
        }
    }
    volume
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
        let dynamic_geometry = geom::DynamicGeometry::new_from_height(static_geometry, &height);

        let problem = Problem::default();

        let hydrostatic_column_pressure =
            compute_hydrostatic_column_pressure(&problem, &dynamic_geometry);
        let boundary_conditions = compute_pressure_boundary_conditions(&problem, &dynamic_geometry);
        let mut implicit_solver = problem.make_implicit_solver();
        match &mut implicit_solver.linear_solver {
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
            linalg::LinearSolver::Klu => {}
        };
        implicit_solver.ignore_max_iters = false;

        let pressure = Solver::compute_initial_pressure(
            &dynamic_geometry,
            &boundary_conditions,
            &implicit_solver,
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
            &implicit_solver,
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
        let dynamic_geometry = geom::DynamicGeometry::new_from_height(static_geometry, &height);

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
}
