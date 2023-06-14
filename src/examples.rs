use pyo3::prelude::*;

use crate::{geom, physics, Float, Vector3};

// TODO: Make this the only public module in order to surface and delete dead
// code

#[pyfunction]
pub fn advection_1d(num_x_cells: usize, num_z_cells: usize) -> physics::Solver {
    let x_axis = geom::Axis::new(0., 1., num_x_cells);
    let y_axis = geom::Axis::new(0., 0.01, 2);
    let grid = geom::Grid::new(x_axis, y_axis, num_z_cells);
    let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

    fn initial_height(x: Float, _y: Float) -> Float {
        (-((x - 0.7) / (0.05)).powi(2)).exp() + 0.1
    }
    let initial_height = geom::HeightField::new(static_geometry.grid(), initial_height);
    let initial_dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &initial_height);

    let problem = physics::Problem {
        rain_rate: None,
        fluid_density: 1e3,
        grav_accel: 9.8,
        x_boundary_condition: physics::HorizBoundaryCondition::HomogeneousNeumann,
        y_boundary_condition: physics::HorizBoundaryCondition::HomogeneousNeumann,
    };

    let velocity = geom::VelocityField::new(&initial_dynamic_geometry, |_, _, _| {
        Vector3::new(-0.03, 0., 0.)
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

    let initial_height = geom::HeightField::new(static_geometry.grid(), |_, _| 1.);
    let initial_dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &initial_height);

    let problem = physics::Problem {
        rain_rate: None,
        fluid_density: 1e3,
        grav_accel: 9.8,
        x_boundary_condition: physics::HorizBoundaryCondition::HomogeneousNeumann,
        y_boundary_condition: physics::HorizBoundaryCondition::HomogeneousNeumann,
    };

    let velocity = geom::VelocityField::new(&initial_dynamic_geometry, |x, _, _| {
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

    let initial_height = geom::HeightField::new(static_geometry.grid(), |_, _| 1.);
    let initial_dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &initial_height);

    let problem = physics::Problem {
        rain_rate: None,
        fluid_density: 1e3,
        grav_accel: 9.8,
        x_boundary_condition: physics::HorizBoundaryCondition::HomogeneousNeumann,
        y_boundary_condition: physics::HorizBoundaryCondition::HomogeneousNeumann,
    };

    let velocity = geom::VelocityField::new(&initial_dynamic_geometry, |_, _, _| {
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
            solver.step(0.01);
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
