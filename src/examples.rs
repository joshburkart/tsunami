use pyo3::prelude::*;

use crate::{fields, geom, physics, Vector3};

// TODO: Make this the only public module in order to surface and delete dead
// code

#[pyfunction]
pub fn advection_1d(num_x_cells: usize, num_z_cells: usize) -> physics::Solver {
    let x_axis = geom::Axis::new(0., 5., num_x_cells);
    let y_axis = geom::Axis::new(0., 0.01, 2);
    let grid = geom::Grid::new(x_axis, y_axis, num_z_cells);
    let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

    let initial_height = fields::AreaScalarField::new(static_geometry.grid(), |x, _| {
        // 0.1 * (x * std::f64::consts::PI / 5.).cos() + 1.
        0.3 * (-((x - 2.5) / (0.9)).powi(2)).exp() + 0.8
    });
    let initial_dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &initial_height);

    let mut problem = physics::Problem::default();
    problem.kinematic_viscosity = 1e-4;

    let velocity = fields::VolVectorField::new(&initial_dynamic_geometry, |_, _, _| {
        Vector3::new(0., 0., 0.)
    });
    physics::Solver::new(
        problem,
        physics::PressureSolver::default(),
        initial_dynamic_geometry,
        initial_height,
        velocity,
    )
}

#[pyfunction]
pub fn singularity_1d(num_x_cells: usize, num_z_cells: usize) -> physics::Solver {
    let x_axis = geom::Axis::new(-1., 1., num_x_cells);
    let y_axis = geom::Axis::new(0., 0.1, 2);
    let grid = geom::Grid::new(x_axis, y_axis, num_z_cells);
    let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

    let initial_height = fields::AreaScalarField::new(static_geometry.grid(), |_, _| 1.);
    let initial_dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &initial_height);

    let problem = physics::Problem::default();

    let velocity = fields::VolVectorField::new(&initial_dynamic_geometry, |x, _, _| {
        Vector3::new(-(x / 0.5) * (-(x / 0.5).powi(2)).exp(), 0., 0.)
    });
    physics::Solver::new(
        problem,
        physics::PressureSolver::default(),
        initial_dynamic_geometry,
        initial_height,
        velocity,
    )
}

#[pyfunction]
pub fn uniform(num_x_cells: usize, num_y_cells: usize, num_z_cells: usize) -> physics::Solver {
    let x_axis = geom::Axis::new(0., 1., num_x_cells);
    let y_axis = geom::Axis::new(0., 1., num_y_cells);
    let grid = geom::Grid::new(x_axis, y_axis, num_z_cells);
    let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

    let initial_height = fields::AreaScalarField::new(static_geometry.grid(), |_, _| 1.);
    let initial_dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &initial_height);

    let problem = physics::Problem::default();

    let velocity = fields::VolVectorField::new(&initial_dynamic_geometry, |_, _, _| {
        Vector3::new(0., 0., 0.)
    });
    physics::Solver::new(
        problem,
        physics::PressureSolver::default(),
        initial_dynamic_geometry,
        initial_height,
        velocity,
    )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_advection_1d() {
        let mut solver = advection_1d(30, 3);
        for _ in 0..10 {
            solver.step(0.001);
        }
    }

    #[test]
    fn test_singularity_1d() {
        let mut solver = singularity_1d(50, 3);
        for _ in 0..10 {
            solver.step(0.01);
        }
    }

    #[test]
    fn test_uniform() {
        let mut solver = uniform(50, 5, 5);
        for _ in 0..10 {
            solver.step(0.01);
        }
    }
}
