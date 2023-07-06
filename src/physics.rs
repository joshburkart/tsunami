use pyo3::prelude::*;

use crate::{
    fields, geom,
    indexing::{self, Indexing, IntoIndexIterator},
    linalg, Array1, Float, Vector2, Vector3,
};

const MIN_HEIGHT: Float = 1e-14;

#[derive(Clone)]
pub struct Fields {
    pub height: fields::HorizScalarField,
    pub velocity: fields::VectorField,
    pub pressure: fields::ScalarField,
    pub hydrostatic_column_pressure: fields::ScalarField,
}

pub struct Problem {
    pub rain_rate: Option<fields::HorizScalarField>,

    pub density: Float,
    pub grav_accel: Float,
    pub kinematic_viscosity: Float,

    velocity_boundary_conditions: fields::BoundaryConditions<Vector3>,
    horiz_velocity_boundary_conditions: fields::HorizBoundaryConditions<Vector2>,
    pub pressure_horiz_boundary_conditions: fields::HorizBoundaryConditions<Float>,
    pub height_boundary_conditions: fields::HorizBoundaryConditions<Float>,
}
impl Default for Problem {
    fn default() -> Self {
        let density = 1000.;
        let grav_accel = 9.8;
        Self {
            rain_rate: None,
            density,
            grav_accel,
            kinematic_viscosity: 1e-6,

            velocity_boundary_conditions: fields::BoundaryConditions {
                horiz: fields::HorizBoundaryConditions::hom_dirichlet(),
                z: fields::VertBoundaryFieldPair {
                    lower: fields::VertBoundaryField::HomDirichlet,
                    upper: fields::VertBoundaryField::HomNeumann,
                },
            },
            horiz_velocity_boundary_conditions: fields::HorizBoundaryConditions::hom_dirichlet(),
            pressure_horiz_boundary_conditions: fields::HorizBoundaryConditions::hom_neumann(),
            height_boundary_conditions: fields::HorizBoundaryConditions::hom_neumann(),
        }
    }
}

#[pyclass]
pub struct Solver {
    problem: Problem,
    dynamic_geometry: Option<geom::DynamicGeometry>,
    fields: Fields,

    pressure_solver: PressureSolver,
}
impl Solver {
    pub fn new(
        problem: Problem,
        pressure_solver: PressureSolver,
        initial_dynamic_geometry: geom::DynamicGeometry,
        initial_height: fields::HorizScalarField,
        initial_velocity: fields::VectorField,
    ) -> Self {
        let hydrostatic_column_pressure =
            compute_hydrostatic_column_pressure(&problem, &initial_dynamic_geometry);
        let pressure = pressure_solver.solve(
            &problem,
            &initial_dynamic_geometry,
            &compute_pressure_boundary_conditions(&problem, &initial_dynamic_geometry),
            &initial_velocity.gradient(
                &initial_dynamic_geometry,
                &problem.velocity_boundary_conditions,
            ),
            &hydrostatic_column_pressure,
            &hydrostatic_column_pressure,
        );
        Self {
            problem,
            pressure_solver,
            dynamic_geometry: Some(initial_dynamic_geometry),
            fields: Fields {
                height: initial_height,
                velocity: initial_velocity,
                pressure,
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
            self.dynamic_geometry.as_ref().unwrap().grid(),
            self.problem.rain_rate.as_ref(),
            &self.fields,
            self.problem.horiz_velocity_boundary_conditions,
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
            let new_center = (self
                .fields
                .height
                .cell_footprint_value(cell_footprint_index)
                + dt * dhdt.cell_footprint_value(cell_footprint_index))
            .max(0.);
            *self
                .fields
                .height
                .cell_footprint_value_mut(cell_footprint_index) = new_center;
        }

        self.dynamic_geometry = Some(geom::DynamicGeometry::new(
            self.dynamic_geometry.take().unwrap().into_static_geometry(),
            &self.fields.height,
        ));
        let dynamic_geometry = self.dynamic_geometry.as_ref().unwrap();

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
            .gradient(dynamic_geometry, &self.problem.velocity_boundary_conditions);
        self.fields.pressure = self.pressure_solver.solve(
            &self.problem,
            dynamic_geometry,
            &pressure_boundary_conditions,
            &fields::TensorField::zeros(dynamic_geometry.grid().cell_indexing()),
            // &shear, //TODO
            &self.fields.hydrostatic_column_pressure,
            &self.fields.pressure,
        );

        // Perform velocity update.
        let mut dvdt = self
            .fields
            .pressure
            .gradient(&dynamic_geometry, &pressure_boundary_conditions)
            / self.problem.density;
        dvdt -= crate::Vector3::new(0., 0., self.problem.grav_accel);
        dvdt += &(self.problem.kinematic_viscosity
            * &self.fields.velocity.laplacian(
                &dynamic_geometry,
                &shear,
                &self.problem.velocity_boundary_conditions,
            ));
        dvdt -= &self
            .fields
            .velocity
            .advect_upwind(dynamic_geometry, &self.problem.velocity_boundary_conditions);
        self.fields.velocity += &(dt * &dvdt);
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
}

// /// Linearly interpolate a velocity vector field array `v` with vertical
// coordinate array `z` onto a /// new vertical coordinate array `new_z`. Use
// constant extrapolation if a value of `new_z` falls /// outside the range of
// `z`. fn interpolate_onto(z: &geom::ScalarField, v: &geom::VectorField, new_z:
// &geom::ScalarField) -> geom::VectorField {     let mut new_v =
// geom::VectorField::zeros((v.dim().0, v.dim().1, new_z.dim().2, v.dim().3));
//     let dim = z.dim();
//     for i in 0..dim.0 {
//         for j in 0..dim.1 {
//             let zij = z.slice(s![i, j, ..]);
//             let vij = v.slice(s![i, j, .., ..]);
//             let new_zij = new_z.slice(s![i, j, ..]);
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
    grid: &geom::Grid,
    rain_rate: Option<&fields::HorizScalarField>,
    fields: &Fields,
    velocity_boundary_conditions: fields::HorizBoundaryConditions<Vector2>,
    height_boundary_conditions: fields::HorizBoundaryConditions<Float>,
) -> fields::HorizScalarField {
    let mut dhdt = fields.velocity.column_average().advect_upwind(
        &fields.height,
        grid,
        velocity_boundary_conditions,
        height_boundary_conditions,
    );

    if let Some(rain_rate) = rain_rate {
        for cell_footprint_index in grid.cell_footprint_indexing().iter() {
            *dhdt.cell_footprint_value_mut(cell_footprint_index) +=
                rain_rate.cell_footprint_value(cell_footprint_index);
        }
    }

    dhdt
}

#[derive(Debug)]
pub enum PressureSolver {
    GaussSeidel {
        max_iters: usize,
        rel_error_tol: Float,
    },
    Direct,
}
impl PressureSolver {
    pub fn solve(
        &self,
        problem: &Problem,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<Float>,
        shear: &fields::TensorField,
        hydrostatic_column_pressure: &fields::ScalarField,
        guess_pressure: &fields::ScalarField,
    ) -> fields::ScalarField {
        let (pressure_matrix, pressure_rhs) = Self::make_pressure_system(
            problem,
            dynamic_geometry,
            boundary_conditions,
            hydrostatic_column_pressure,
            shear,
        );

        let flattened_pressure = match *self {
            PressureSolver::GaussSeidel {
                max_iters,
                rel_error_tol,
            } => linalg::solve_linear_system_gauss_seidel(
                pressure_matrix.view(),
                guess_pressure.flatten(dynamic_geometry.grid().cell_indexing()),
                pressure_rhs,
                max_iters,
                rel_error_tol,
            ),
            PressureSolver::Direct => {
                linalg::solve_linear_system_direct(pressure_matrix.to_dense(), pressure_rhs)
            }
        };
        let flattened_pressure = match flattened_pressure {
            Err(linalg::LinearSolveError::NotDiagonallyDominant {
                row_index,
                abs_diag,
                sum_abs_row,
                delta,
            }) => {
                let cell = dynamic_geometry.grid().cell_indexing().unflatten(row_index);
                panic!(
                    "Pressure system was not diagonally dominant at cell {cell:?} ({:?}): |diag| \
                     = {abs_diag:.4} vs. sum_j |M_ij| = {sum_abs_row:.4}, delta = {delta}",
                    dynamic_geometry.grid().cell_indexing().classify_cell(cell)
                )
            }
            Err(error) => panic!("Failed to solve pressure equations: {error:?}"),
            Ok(result) => result,
        };

        fields::ScalarField::unflatten(dynamic_geometry.grid().cell_indexing(), &flattened_pressure)
    }

    fn make_pressure_system(
        problem: &Problem,
        dynamic_geometry: &geom::DynamicGeometry,
        boundary_conditions: &fields::BoundaryConditions<Float>,
        hydrostatic_column_pressure: &fields::ScalarField,
        shear: &fields::TensorField,
    ) -> (sprs::CsMat<Float>, Array1) {
        let mut rhs_field = shear.map(|shear| (shear * shear.transpose()).sum());
        rhs_field *= problem.density;

        let cell_indexing = dynamic_geometry.grid().cell_indexing();
        let mut matrix = sprs::TriMat::new((cell_indexing.len(), cell_indexing.len()));
        let mut rhs = Array1::zeros(cell_indexing.len());

        for flat_index in 0..cell_indexing.len() {
            let cell_index = cell_indexing.unflatten(flat_index);
            let cell = dynamic_geometry.cell(cell_index);

            // if let indexing::CellIndex {
            //     footprint:
            //         indexing::CellFootprintIndex {
            //             x: 5,
            //             y: 0,
            //             triangle: indexing::Triangle::LowerRight,
            //         },
            //     z: 0,
            // } = cell_index
            // {
            //     println!("");
            // }

            let mut add_coef = |col_cell_index, value| {
                // if let indexing::CellIndex {
                //     footprint:
                //         indexing::CellFootprintIndex {
                //             x: 5,
                //             y: 0,
                //             triangle: indexing::Triangle::LowerRight,
                //         },
                //     z: 0,
                // } = cell_index
                // {
                //     println!("  {col_cell_index:?}: {value}");
                // }
                matrix.add_triplet(flat_index, cell_indexing.flatten(col_cell_index), value)
            };

            if cell.volume / dynamic_geometry.grid().footprint_area() < MIN_HEIGHT {
                add_coef(cell_index, 1.);
                rhs[flat_index] = hydrostatic_column_pressure.cell_value(cell_index);
                continue;
            }

            let mut cell_coef = 0.;
            for face in &cell.faces {
                let mut handle_boundary_face =
                    |block_paired_cell: &geom::Cell, boundary_condition| match boundary_condition {
                        fields::BoundaryCondition::HomDirichlet => {
                            let displ = face.centroid().coords
                                - 0.5 * (cell.centroid.coords + block_paired_cell.centroid.coords);
                            let c_corr = 1. / face.outward_normal().dot(&displ);
                            let coef = -0.5 * c_corr * face.area() / cell.volume;
                            add_coef(block_paired_cell.index, coef);
                            cell_coef += coef;
                        }
                        fields::BoundaryCondition::HomNeumann => {}
                        fields::BoundaryCondition::InhomNeumann(boundary_value) => {
                            // TODO: Account for block paired cell if an inhomogeneous Neumann BC is
                            // used for X or Y boundaries.
                            rhs[flat_index] -= boundary_value * face.area() / cell.volume;
                        }
                    };
                match face.neighbor() {
                    indexing::CellNeighbor::Cell(neighbor_cell_index) => {
                        let neighbor_cell = dynamic_geometry.cell(neighbor_cell_index);
                        let displ = neighbor_cell.centroid - cell.centroid;
                        let c_corr = 1. / face.outward_normal().dot(&displ);
                        let coef = c_corr * face.area() / cell.volume;
                        cell_coef -= coef;
                        add_coef(neighbor_cell_index, coef);
                    }
                    indexing::CellNeighbor::XBoundary(boundary) => match boundary {
                        indexing::Boundary::Lower => handle_boundary_face(
                            dynamic_geometry.cell(cell_index.flip()),
                            boundary_conditions.horiz.x.lower,
                        ),
                        indexing::Boundary::Upper => handle_boundary_face(
                            dynamic_geometry.cell(cell_index.flip()),
                            boundary_conditions.horiz.x.upper,
                        ),
                    },
                    indexing::CellNeighbor::YBoundary(boundary) => match boundary {
                        indexing::Boundary::Lower => handle_boundary_face(
                            dynamic_geometry.cell(cell_index.flip()),
                            boundary_conditions.horiz.y.lower,
                        ),
                        indexing::Boundary::Upper => handle_boundary_face(
                            dynamic_geometry.cell(cell_index.flip()),
                            boundary_conditions.horiz.y.upper,
                        ),
                    },
                    indexing::CellNeighbor::ZBoundary(boundary) => match boundary {
                        indexing::Boundary::Lower => handle_boundary_face(
                            cell,
                            boundary_conditions
                                .z
                                .lower
                                .boundary_condition(cell_index.footprint),
                        ),
                        indexing::Boundary::Upper => handle_boundary_face(
                            cell,
                            boundary_conditions
                                .z
                                .upper
                                .boundary_condition(cell_index.footprint),
                        ),
                    },
                }
            }
            add_coef(cell_index, cell_coef);
            rhs[flat_index] += rhs_field.cell_value(cell_index);

            // if let indexing::CellIndex {
            //     footprint:
            //         indexing::CellFootprintIndex {
            //             x: 5,
            //             y: 0,
            //             triangle: indexing::Triangle::LowerRight,
            //         },
            //     z: 0,
            // } = cell_index
            // {
            //     println!("  rhs: {}", rhs[flat_index]);
            // }
        }
        (matrix.to_csr(), rhs)
    }
}
impl Default for PressureSolver {
    fn default() -> Self {
        Self::GaussSeidel {
            max_iters: 30000,
            rel_error_tol: 1e-1,
        }
    }
}

fn compute_pressure_boundary_conditions(
    problem: &Problem,
    dynamic_geometry: &geom::DynamicGeometry,
) -> fields::BoundaryConditions<Float> {
    let cell_footprint_indexing = dynamic_geometry.grid().cell_footprint_indexing();
    let mut lower_boundary_field = fields::HorizField::zeros(cell_footprint_indexing);
    for cell_footprint_index in cell_footprint_indexing.iter() {
        *lower_boundary_field.cell_footprint_value_mut(cell_footprint_index) = -problem.density
            * problem.grav_accel
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
) -> fields::ScalarField {
    let mut hydrostatic_column_pressure =
        fields::ScalarField::zeros(dynamic_geometry.grid().cell_indexing());
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
                problem.density * problem.grav_accel * (surface_z - z);
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

        let height = fields::HorizScalarField::new(static_geometry.grid(), |_, _| 10.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let problem = Problem::default();

        let hydrostatic_column_pressure =
            compute_hydrostatic_column_pressure(&problem, &dynamic_geometry);
        let pressure = PressureSolver::default().solve(
            &problem,
            &dynamic_geometry,
            &compute_pressure_boundary_conditions(&problem, &dynamic_geometry),
            &fields::TensorField::zeros(dynamic_geometry.grid().cell_indexing()),
            &hydrostatic_column_pressure,
            &hydrostatic_column_pressure,
        );

        approx::assert_abs_diff_eq!(hydrostatic_column_pressure, pressure, epsilon = 1e-5);
    }

    #[test]
    fn test_hydrostatic_column_pressure() {
        let x_axis = geom::Axis::new(-1., 1., 3);
        let y_axis = geom::Axis::new(10., 11., 4);
        let grid = geom::Grid::new(x_axis, y_axis, 50);
        let static_geometry = geom::StaticGeometry::new(grid, |_, _| 0.);

        let height = fields::HorizScalarField::new(static_geometry.grid(), |_, _| 10.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let problem = Problem::default();

        let pressure = compute_hydrostatic_column_pressure(&problem, &dynamic_geometry);
        let boundary_conditions = compute_pressure_boundary_conditions(&problem, &dynamic_geometry);
        let pressure_gradient = pressure.gradient(&dynamic_geometry, &boundary_conditions);

        for cell_index in dynamic_geometry.grid().cell_indexing().iter() {
            let pressure_gradient_value = pressure_gradient.cell_value(cell_index);
            if !approx::relative_eq!(
                pressure_gradient_value,
                Vector3::new(0., 0., -problem.density * problem.grav_accel),
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

        let height = fields::HorizScalarField::new(static_geometry.grid(), |_, _| 3.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let fields = Fields {
            height: height.clone(),
            velocity: fields::VectorField::new(&dynamic_geometry, |_, _, _| Vector3::zeros()),
            pressure: fields::ScalarField::zeros(dynamic_geometry.grid().cell_indexing()),
            hydrostatic_column_pressure: fields::ScalarField::zeros(
                dynamic_geometry.grid().cell_indexing(),
            ),
        };

        // No rain.
        {
            let rain_rate =
                fields::HorizScalarField::zeros(dynamic_geometry.grid().cell_footprint_indexing());

            let dhdt = compute_height_time_deriv(
                &dynamic_geometry.grid(),
                Some(&rain_rate),
                &fields,
                fields::HorizBoundaryConditions::hom_dirichlet(),
                fields::HorizBoundaryConditions::hom_dirichlet(),
            );

            approx::assert_abs_diff_eq!(dhdt, rain_rate);
        }
        // Some rain.
        {
            let rain_rate = fields::HorizScalarField::new(dynamic_geometry.grid(), |_, _| 1.5e-2);

            let dhdt = compute_height_time_deriv(
                &dynamic_geometry.grid(),
                Some(&rain_rate),
                &fields,
                fields::HorizBoundaryConditions::hom_dirichlet(),
                fields::HorizBoundaryConditions::hom_dirichlet(),
            );

            approx::assert_abs_diff_eq!(dhdt, rain_rate);
        }
    }
}
