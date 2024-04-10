use pyo3::prelude::*;

use crate::{fields, geom, physics, Float, Vector3};

#[pyfunction]
pub fn bump_1d(
    num_x_cells: usize,
    num_z_cells: usize,
    kinematic_viscosity: Float,
    amplitude: Float,
    width: Float,
) -> physics::Solver {
    let x_axis = geom::Axis::new(0., 3., num_x_cells);
    let y_axis = geom::Axis::new(0., 0.01, 1);
    let grid = geom::Grid::new(x_axis, y_axis, num_z_cells);
    let terrain_func = |x: Float, _| -> Float {
        0.06 * (x * std::f64::consts::PI / 1.).sin().powi(2)
        // 0.1 * (-((x - 2.5) / (0.3)).powi(2)).exp()
    };
    let static_geometry = geom::StaticGeometry::new(grid, &terrain_func);

    let initial_height = fields::AreaScalarField::new(static_geometry.grid(), |x, y| {
        // 0.1 * (x * std::f64::consts::PI / 5.).cos() + 1.
        // -0.2 * (-((x - 2.5) / (0.3)).powi(6)).exp() + 0.8
        amplitude * (-((x - 1.) / (width)).powi(2)).exp() + 0.8 - terrain_func(x, y)
    });
    let initial_dynamic_geometry =
        geom::DynamicGeometry::new_from_height(static_geometry, &initial_height);

    let mut problem = physics::Problem::default();
    problem.kinematic_viscosity = kinematic_viscosity;
    problem.horiz_velocity_boundary_conditions.x = fields::HorizBoundaryConditionPair {
        lower: fields::BoundaryCondition::NoPenetration,
        upper: fields::BoundaryCondition::NoPenetration,
    };

    let velocity = fields::VolVectorField::new(&initial_dynamic_geometry, |_, _, _| {
        Vector3::new(0., 0., 0.)
    });
    let implicit_solver = problem.make_implicit_solver();
    // let implicit_solver = crate::implicit::ImplicitSolver {
    //     linear_solver: crate::linalg::LinearSolver::Direct,
    //     ignore_max_iters: true,
    // };
    physics::Solver::new(
        problem,
        implicit_solver,
        initial_dynamic_geometry,
        initial_height,
        velocity,
    )
}

#[pyfunction]
pub fn hill_1d(
    num_x_cells: usize,
    num_z_cells: usize,
    kinematic_viscosity: Float,
    height: Float,
) -> physics::Solver {
    let x_axis = geom::Axis::new(0., 3., num_x_cells);
    let y_axis = geom::Axis::new(0., 0.01, 1);
    let grid = geom::Grid::new(x_axis, y_axis, num_z_cells);
    let wave = |x: Float| 1. / 2. * ((x - 1.5) * std::f64::consts::PI * 3. / 2.).cos().powi(2);
    let hill = |x: Float| (-((x - 1.5) / 0.5).powi(2)).exp();
    let terrain_func = |x: Float, _| -> Float { height * (hill(x) + wave(x)) + 0.4 };
    let static_geometry = geom::StaticGeometry::new(grid, &terrain_func);

    let initial_height = fields::AreaScalarField::new(static_geometry.grid(), |_, _| {
        0.02 //+ terrain_func(1.5, y) - 0.1 * terrain_func(x, y)
    });
    let initial_dynamic_geometry =
        geom::DynamicGeometry::new_from_height(static_geometry, &initial_height);

    let mut problem = physics::Problem::default();
    problem.kinematic_viscosity = kinematic_viscosity;
    problem.horiz_velocity_boundary_conditions.x = fields::HorizBoundaryConditionPair {
        lower: fields::BoundaryCondition::HomNeumann,
        upper: fields::BoundaryCondition::HomNeumann,
    };
    problem.rain_rate = Some(fields::AreaField::new(
        initial_dynamic_geometry.grid(),
        |x, _| hill(x) * 0.02,
    ));
    // problem.max_speed = Some(10.);

    let velocity = fields::VolVectorField::new(&initial_dynamic_geometry, |_, _, _| {
        Vector3::new(0., 0., 0.)
    });
    let implicit_solver = problem.make_implicit_solver();
    // let implicit_solver = crate::implicit::ImplicitSolver {
    //     linear_solver: crate::linalg::LinearSolver::Direct,
    //     ignore_max_iters: true,
    // };
    physics::Solver::new(
        problem,
        implicit_solver,
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
    let initial_dynamic_geometry =
        geom::DynamicGeometry::new_from_height(static_geometry, &initial_height);

    let problem = physics::Problem::default();

    let velocity = fields::VolVectorField::new(&initial_dynamic_geometry, |_, _, _| {
        Vector3::new(0., 0., 0.)
    });
    let implicit_solver = problem.make_implicit_solver();
    physics::Solver::new(
        problem,
        implicit_solver,
        initial_dynamic_geometry,
        initial_height,
        velocity,
    )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bump_1d() {
        let mut solver = bump_1d(30, 3, 0.01, 0.1, 0.1);
        for i in 0..10 {
            println!("step {i}");
            solver.step(0.001);
        }
    }

    #[test]
    fn test_ramp_1d() {
        let mut solver = hill_1d(50, 3, 0.1, 0.1);
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
