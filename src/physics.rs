use pyo3::prelude::*;

use crate::{geom, indexing, math, Array1, Float};

const MIN_HEIGHT: Float = 1e-14;

#[derive(Clone)]
pub struct Fields {
    pub height: geom::HeightField,
    pub velocity: geom::VelocityField,
    pub pressure: geom::PressureField,
    pub hydrostatic_pressure: geom::PressureField,
}

pub struct Problem {
    pub rain_rate: Option<geom::HeightField>,

    pub fluid_density: Float,
    pub grav_accel: Float,

    pub x_boundary_condition: HorizBoundaryCondition,
    pub y_boundary_condition: HorizBoundaryCondition,
}

pub enum HorizBoundaryCondition {
    HomogeneousNeumann,
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
        initial_height: geom::HeightField,
        initial_velocity: geom::VelocityField,
    ) -> Self {
        let hydrostatic_pressure =
            compute_hydrostatic_pressure(&problem, &initial_dynamic_geometry);
        let pressure = pressure_solver.solve(
            &problem,
            &initial_dynamic_geometry,
            &initial_velocity,
            &hydrostatic_pressure,
            &hydrostatic_pressure,
        );
        Self {
            problem,
            pressure_solver,
            dynamic_geometry: Some(initial_dynamic_geometry),
            fields: Fields {
                height: initial_height,
                velocity: initial_velocity,
                pressure,
                hydrostatic_pressure,
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
        );
        for cell_footprint_index in indexing::iter_indices(
            self.dynamic_geometry
                .as_ref()
                .unwrap()
                .grid()
                .cell_footprint_indexing(),
        ) {
            let new_center = (self.fields.height.center_value(cell_footprint_index)
                + dt * dhdt.center_value(cell_footprint_index))
            .max(0.);
            *self.fields.height.center_value_mut(cell_footprint_index) = new_center;
        }

        self.dynamic_geometry = Some(geom::DynamicGeometry::new(
            self.dynamic_geometry.take().unwrap().into_static_geometry(),
            &self.fields.height,
        ));

        // Interpolate velocity to new height map.
        // fields.velocity =
        //     interpolate_onto(&z_lattice.centers, &fields.velocity,
        // &new_z_axis.centers);

        // Compute new pressure field.
        self.fields.hydrostatic_pressure =
            compute_hydrostatic_pressure(&self.problem, self.dynamic_geometry.as_ref().unwrap());
        self.fields.pressure = self.pressure_solver.solve(
            &self.problem,
            self.dynamic_geometry.as_ref().unwrap(),
            &self.fields.velocity,
            &self.fields.hydrostatic_pressure,
            &self.fields.pressure,
        );

        // Perform velocity update.
        // TODO
    }
}
#[pymethods]
impl Solver {
    #[getter]
    pub fn grid(&self) -> geom::Grid {
        self.dynamic_geometry.as_ref().unwrap().grid().clone()
    }

    #[getter]
    pub fn z_lattice<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.dynamic_geometry.as_ref().unwrap().z_lattice_py(py)
    }

    #[getter]
    #[pyo3(name = "pressure")]
    pub fn pressure_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.fields.pressure.pressure_py(py)
    }

    #[getter]
    #[pyo3(name = "hydrostatic_pressure")]
    pub fn hydrostatic_pressure_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.fields.hydrostatic_pressure.pressure_py(py)
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
    rain_rate: Option<&geom::HeightField>,
    fields: &Fields,
) -> geom::HeightField {
    use indexing::Index;

    // TODO: Make a struct for 2D velocity
    let column_averaged_velocity_field = fields.velocity.column_average();

    let mut dhdt = 0. * &fields.height;
    let cell_footprint_indexing = grid.cell_footprint_indexing();
    for cell_footprint_index in indexing::iter_indices(cell_footprint_indexing) {
        let column_averaged_velocity =
            column_averaged_velocity_field[cell_footprint_index.to_array_index()];
        let cell_footprint_pairs =
            cell_footprint_indexing.compute_footprint_pairs(cell_footprint_index);
        for cell_footprint_pair in cell_footprint_pairs {
            let cell_footprint_edge = grid.compute_cell_footprint_edge(cell_footprint_pair);
            let outward_normal = cell_footprint_edge.outward_normal;

            let height = fields.height.center_value(cell_footprint_index);
            let neighbor_height = match cell_footprint_pair.neighbor {
                indexing::CellFootprintNeighbor::CellFootprint(neighbor_cell_footprint) => {
                    fields.height.center_value(neighbor_cell_footprint)
                }
                indexing::CellFootprintNeighbor::Boundary(_) => 0.,
            };

            let projected_column_averaged_velocity = outward_normal.dot(&column_averaged_velocity);
            let neighbor_is_upwind = projected_column_averaged_velocity < 0.;
            let upwind_height_to_advect = if neighbor_is_upwind {
                neighbor_height
            } else {
                height
            };
            // TODO: Use or remove
            // let linear_height_to_advect = 0.5 * (height + neighbor_height);

            // let upwind_weight = 1.;
            // let height_to_advect = (upwind_weight * upwind_height_to_advect
            //     + linear_height_to_advect)
            //     / (1. + upwind_weight);

            *dhdt.center_value_mut(cell_footprint_index) -= upwind_height_to_advect
                * cell_footprint_edge.length
                * projected_column_averaged_velocity
                / grid.footprint_area();
        }

        if let Some(rain_rate) = rain_rate {
            *dhdt.center_value_mut(cell_footprint_index) +=
                rain_rate.center_value(cell_footprint_index);
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
        velocity: &geom::VelocityField,
        hydrostatic_pressure: &geom::PressureField,
        guess_pressure: &geom::PressureField,
    ) -> geom::PressureField {
        let pressure_matrix = Self::make_pressure_matrix(dynamic_geometry);
        let pressure_rhs = Self::make_pressure_rhs_vector(
            problem,
            dynamic_geometry,
            hydrostatic_pressure,
            velocity,
        );

        let flattened_pressure = match *self {
            PressureSolver::GaussSeidel {
                max_iters,
                rel_error_tol,
            } => math::solve_linear_system_gauss_seidel(
                pressure_matrix.view(),
                guess_pressure.flatten(dynamic_geometry.grid().vertex_indexing()),
                pressure_rhs,
                max_iters,
                rel_error_tol,
            ),
            PressureSolver::Direct => {
                math::solve_linear_system_direct(pressure_matrix.to_dense(), pressure_rhs)
            }
        };
        let flattened_pressure = match flattened_pressure {
            Err(math::LinearSolveError::NotDiagonallyDominant {
                row_index,
                abs_diag,
                sum_abs_row,
            }) => {
                use indexing::Indexing;
                let vertex = dynamic_geometry
                    .grid()
                    .vertex_indexing()
                    .unflatten(row_index);
                panic!(
                    "Pressure system was not diagonally dominant at vertex {vertex:?} ({:?}): \
                     |diag| = {abs_diag:.4} vs. sum_j |M_ij| = {sum_abs_row:.4}",
                    dynamic_geometry
                        .grid()
                        .vertex_indexing()
                        .classify_vertex(vertex)
                )
            }
            Err(error) => panic!("Failed to solve pressure equations: {error:?}"),
            Ok(result) => result,
        };

        geom::PressureField::unflatten(
            dynamic_geometry.grid().vertex_indexing(),
            &flattened_pressure,
        )
    }

    fn make_pressure_matrix(dynamic_geometry: &geom::DynamicGeometry) -> sprs::CsMat<Float> {
        use indexing::{Indexing, VertexClassification};

        let vertex_indexing = dynamic_geometry.grid().vertex_indexing();
        let mut matrix = sprs::TriMat::new((vertex_indexing.len(), vertex_indexing.len()));

        let dx_inv = 1. / dynamic_geometry.grid().x_axis().spacing();
        let dx_inv_sq = dx_inv.powi(2);
        let dy_inv = 1. / dynamic_geometry.grid().y_axis().spacing();
        let dy_inv_sq = dy_inv.powi(2);

        for flat_index in 0..vertex_indexing.len() {
            let mut add_entry = |vertex, value| {
                matrix.add_triplet(flat_index, vertex_indexing.flatten(vertex), value)
            };

            let vertex = vertex_indexing.unflatten(flat_index);
            let dz = dynamic_geometry.z_lattice().z_spacing(vertex.footprint);
            if dz < MIN_HEIGHT {
                add_entry(vertex, 1.);
                continue;
            }

            let dz_inv = 1. / dz;
            let dz_inv_sq = dz_inv.powi(2);

            match dynamic_geometry
                .grid()
                .vertex_indexing()
                .classify_vertex(vertex)
            {
                // Floor boundary condition: hydrostatic equilibrium.
                VertexClassification::Floor => {
                    add_entry(vertex, -dz_inv);
                    add_entry(vertex.increment_z(1), dz_inv);
                }
                // Surface boundary condition: pressure is zero.
                VertexClassification::Surface => add_entry(vertex, 1.),
                // Horizontal boundary conditions: hydrostatic equilibrium.
                VertexClassification::Left
                | VertexClassification::LowerLeft
                | VertexClassification::UpperLeft
                | VertexClassification::Right
                | VertexClassification::LowerRight
                | VertexClassification::UpperRight
                | VertexClassification::Lower
                | VertexClassification::Upper => {
                    add_entry(vertex, 1.);
                }
                // Interior boundary conditions: Poisson's equation.
                VertexClassification::Interior => {
                    add_entry(vertex, -2. * (dx_inv_sq + dy_inv_sq + dz_inv_sq));
                    add_entry(vertex.increment_x(-1), dx_inv_sq);
                    add_entry(vertex.increment_x(1), dx_inv_sq);
                    add_entry(vertex.increment_y(-1), dy_inv_sq);
                    add_entry(vertex.increment_y(1), dy_inv_sq);
                    // TODO: Need to include Jacobian terms
                    add_entry(vertex.increment_z(-1), dz_inv_sq);
                    add_entry(vertex.increment_z(1), dz_inv_sq);
                }
            }
        }
        matrix.to_csr()
    }

    fn make_pressure_rhs_vector(
        problem: &Problem,
        dynamic_geometry: &geom::DynamicGeometry,
        hydrostatic_pressure: &geom::PressureField,
        _velocity: &geom::VelocityField,
    ) -> Array1 {
        let vertex_indexing = dynamic_geometry.grid().vertex_indexing();

        let mut rhs = geom::PressureField::zeros(vertex_indexing);
        for vertex in indexing::iter_indices(vertex_indexing) {
            *rhs.vertex_value_mut(vertex) = {
                use indexing::VertexClassification;

                let dz = dynamic_geometry.z_lattice().z_spacing(vertex.footprint);
                if dz < MIN_HEIGHT {
                    hydrostatic_pressure.vertex_value(vertex)
                } else {
                    match vertex_indexing.classify_vertex(vertex) {
                        // Terrain boundary condition: hydrostatic equilibrium.
                        VertexClassification::Floor => -problem.fluid_density * problem.grav_accel,
                        // Surface boundary condition: pressure is zero.
                        VertexClassification::Surface => 0.,
                        // Horizontal boundary conditions: hydrostatic equilibrium.
                        VertexClassification::Left
                        | VertexClassification::Right
                        | VertexClassification::Lower
                        | VertexClassification::Upper
                        | VertexClassification::LowerLeft
                        | VertexClassification::LowerRight
                        | VertexClassification::UpperLeft
                        | VertexClassification::UpperRight => {
                            hydrostatic_pressure.vertex_value(vertex)
                        }
                        // Interior PDE is TEMPORARILY (TODO) homogeneous. Should instead be the
                        // trace of the square of the shear tensor.
                        VertexClassification::Interior => 0.,
                    }
                }
            };
        }
        rhs.flatten(vertex_indexing)
    }
}
impl Default for PressureSolver {
    fn default() -> Self {
        Self::GaussSeidel {
            max_iters: 300,
            rel_error_tol: 1e-4,
        }
    }
}

pub fn compute_hydrostatic_pressure(
    problem: &Problem,
    dynamic_geometry: &geom::DynamicGeometry,
) -> geom::PressureField {
    let mut hydrostatic_pressure =
        geom::PressureField::zeros(dynamic_geometry.grid().vertex_indexing());
    for vertex_footprint in
        indexing::iter_indices(dynamic_geometry.grid().vertex_footprint_indexing())
    {
        let surface_z = dynamic_geometry
            .z_lattice()
            .vertex_value(indexing::VertexIndex {
                footprint: vertex_footprint,
                z: dynamic_geometry.grid().vertex_indexing().num_z_points() - 1,
            });
        for vertex_index in dynamic_geometry
            .grid()
            .vertex_indexing()
            .column(vertex_footprint)
        {
            let z = dynamic_geometry.z_lattice().vertex_value(vertex_index);
            *hydrostatic_pressure.vertex_value_mut(vertex_index) =
                problem.fluid_density * problem.grav_accel * (surface_z - z);
        }
    }
    hydrostatic_pressure
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

        let height = geom::HeightField::new(static_geometry.grid(), |_, _| 10.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let problem = Problem {
            rain_rate: None,
            fluid_density: 1000.,
            grav_accel: 9.8,
            x_boundary_condition: HorizBoundaryCondition::HomogeneousNeumann,
            y_boundary_condition: HorizBoundaryCondition::HomogeneousNeumann,
        };

        let hydrostatic_pressure = compute_hydrostatic_pressure(&problem, &dynamic_geometry);
        let pressure = PressureSolver::default().solve(
            &problem,
            &dynamic_geometry,
            &geom::VelocityField::zeros(&dynamic_geometry),
            &hydrostatic_pressure,
            &hydrostatic_pressure,
        );

        approx::assert_abs_diff_eq!(hydrostatic_pressure, pressure, epsilon = 1e-5);
    }

    #[test]
    fn test_hydrostatic_pressure() {
        let x_axis = geom::Axis::new(-1., 1., 3);
        let y_axis = geom::Axis::new(10., 11., 4);
        let grid = geom::Grid::new(x_axis, y_axis, 5);
        let static_geometry =
            geom::StaticGeometry::new(grid, &|x, _y| (x * 5. * std::f64::consts::PI).sin());

        let height = geom::HeightField::new(static_geometry.grid(), |_, _| 10.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let problem = Problem {
            rain_rate: None,
            fluid_density: 1000.,
            grav_accel: 9.8,
            x_boundary_condition: HorizBoundaryCondition::HomogeneousNeumann,
            y_boundary_condition: HorizBoundaryCondition::HomogeneousNeumann,
        };

        let pressure = compute_hydrostatic_pressure(&problem, &dynamic_geometry);
        for vertex_footprint in
            indexing::iter_indices(dynamic_geometry.grid().vertex_footprint_indexing())
        {
            approx::assert_abs_diff_eq!(
                pressure.vertex_value(indexing::VertexIndex {
                    footprint: vertex_footprint,
                    z: dynamic_geometry.grid().vertex_indexing().num_z_points() - 1
                }),
                0.
            );
            for z in 1..dynamic_geometry.grid().vertex_indexing().num_z_points() {
                let vertex = indexing::VertexIndex {
                    footprint: vertex_footprint,
                    z,
                };
                approx::assert_relative_eq!(
                    (pressure.vertex_value(vertex) - pressure.vertex_value(vertex.increment_z(-1)))
                        / dynamic_geometry.z_lattice().z_spacing(vertex_footprint),
                    -problem.fluid_density * problem.grav_accel,
                    max_relative = 1e-6,
                );
            }
        }
    }

    #[test]
    fn test_rain() {
        let x_axis = geom::Axis::new(-1., 1., 3);
        let y_axis = geom::Axis::new(10., 11., 4);
        let grid = geom::Grid::new(x_axis, y_axis, 5);
        let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

        let height = geom::HeightField::new(static_geometry.grid(), |_, _| 3.);
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let fields = Fields {
            height: height.clone(),
            velocity: geom::VelocityField::new(&dynamic_geometry, |_, _, _| Vector3::zeros()),
            pressure: geom::PressureField::zeros(dynamic_geometry.grid().vertex_indexing()),
            hydrostatic_pressure: geom::PressureField::zeros(
                dynamic_geometry.grid().vertex_indexing(),
            ),
        };

        // No rain.
        {
            let rain_rate = geom::HeightField::new(dynamic_geometry.grid(), |_, _| 0.);

            let dhdt =
                compute_height_time_deriv(&dynamic_geometry.grid(), Some(&rain_rate), &fields);

            approx::assert_abs_diff_eq!(dhdt, 0. * &dhdt);
        }
        // Some rain.
        {
            let rain_rate = geom::HeightField::new(dynamic_geometry.grid(), |_, _| 1.5e-2);

            let dhdt =
                compute_height_time_deriv(&dynamic_geometry.grid(), Some(&rain_rate), &fields);

            approx::assert_abs_diff_eq!(dhdt, 1.5e-2 + 0. * &dhdt);
        }
    }
}
