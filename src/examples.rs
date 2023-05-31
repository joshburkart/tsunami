use pyo3::prelude::*;

use crate::{geom, physics, Float, Vector3};

#[pyfunction]
pub fn advection_1d(num_x_cells: usize, num_z_cells_per_column: usize) -> physics::Solver {
    let x_axis = geom::Axis::new(0., 1., num_x_cells);
    let y_axis = geom::Axis::new(0., 0.01, 1);
    let grid = geom::Grid::new(x_axis, y_axis, num_z_cells_per_column);
    let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

    fn initial_height(x: Float, _y: Float) -> Float {
        (-((x - 0.5) / (0.1)).powi(2)).exp() + 0.1
        // (x * (2. * std::f64::consts::PI) * 5.5).sin().powi(2) + 1.
    }
    let initial_height = geom::HorizScalarField::new(static_geometry.grid(), initial_height);
    let initial_dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &initial_height);

    let initial_fields = physics::Fields {
        height: initial_height,
        pressure: geom::ScalarField::new(&initial_dynamic_geometry, |_, _, _| 0.),
        velocity: geom::VectorField::new(&initial_dynamic_geometry, |_, _, _| {
            Vector3::new(0.1, 0., 0.)
        }),
    };

    let rain_rate = geom::HorizScalarField::new(initial_dynamic_geometry.grid(), |_, _| 0.);

    physics::Solver::new(initial_dynamic_geometry, initial_fields, rain_rate)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_perform_height_update() {
        let mut solver = advection_1d(200, 2);
        for _ in 0..1 {
            solver.step(0.01);
        }
    }
}
