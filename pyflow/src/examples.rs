use flow::float_consts::PI;
use flow::{bases, physics, Float};
use ndarray as nd;
use numpy::IntoPyArray;
use pyo3::prelude::*;

#[pyclass]
pub struct RectangularFields {
    height: nd::Array2<Float>,
    velocity: nd::Array3<Float>,
}
#[pymethods]
impl RectangularFields {
    #[getter]
    #[pyo3(name = "height")]
    pub fn height_py<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<Float>> {
        self.height.clone().into_pyarray_bound(py)
    }
    #[getter]
    #[pyo3(name = "velocity")]
    pub fn velocity_py<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray3<Float>> {
        self.velocity.clone().into_pyarray_bound(py)
    }
}

#[pyclass]
pub struct RectangularSolver {
    solver: physics::Solver<bases::ylm::SphericalHarmonicBasis>,
}
#[pymethods]
impl RectangularSolver {
    pub fn integrate(&mut self, delta_t: Float) {
        self.solver.integrate(delta_t)
    }

    #[getter]
    pub fn fields(&self) -> RectangularFields {
        RectangularFields {
            height: self.solver.fields().height_grid(),
            velocity: self.solver.fields().velocity_grid(),
        }
    }

    #[getter]
    pub fn x_axis<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<Float>> {
        use bases::Basis;

        self.solver.problem().basis.axes()[0]
            .clone()
            .into_pyarray_bound(py)
    }
    #[getter]
    pub fn y_axis<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<Float>> {
        use bases::Basis;

        self.solver.problem().basis.axes()[1]
            .clone()
            .into_pyarray_bound(py)
    }
}

#[pyfunction]
pub fn bump_2d_spectral(
    num_points: usize,
    kinematic_viscosity: Float,
    base_height: Float,
    amplitude: Float,
) -> RectangularSolver {
    use bases::Basis;
    let bump_size = 0.1;

    // Want to generate a power so that a periodic "bump" is generated of width `bump_size`. Start
    // from FWHM definition, given circumference = lengths[i]:
    //
    // ```
    // 1/2 = cos(pi * bump_size / lengths[i])^(2n)
    // ```
    //
    // Solve for `n`:
    //
    // ```
    // n = log(1/2) / (2 * log(cos(pi * bump_size / lengths[i])))
    // ```
    let pow = |bump_size: Float, length: Float| {
        (0.5 as Float).ln() / (PI * (bump_size / length)).cos().ln() / 2.
    };
    let basis = std::sync::Arc::new(bases::ylm::SphericalHarmonicBasis::new(num_points));
    let terrain_height = basis.scalar_to_spectral(&basis.make_scalar(|_, _| 1.));
    let mut initial_fields = physics::Fields::zeros(basis.clone());
    let powx = pow(bump_size, 1.);
    let powy = pow(bump_size, 1.);
    initial_fields.assign_height(&basis.scalar_to_spectral(&basis.make_scalar(|mu, phi| {
        base_height
            + amplitude
                * (mu.acos() + PI / 4.).sin().powi(2).max(0.).powf(powx)
                * (phi).sin().max(0.).powi(2).powf(powy)
    })));
    let problem = physics::Problem {
        basis,
        terrain_height,
        grav_accel: 9.8,
        kinematic_viscosity,
        rtol: 1e-3,
        atol: 1e-4,
    };
    let solver = physics::Solver::new(problem, initial_fields);
    RectangularSolver { solver }
}

// #[pyfunction]
// pub fn bump_1d(
//     num_x_cells: usize,
//     num_z_cells: usize,
//     kinematic_viscosity: Float,
//     amplitude: Float,
//     width: Float,
// ) -> physics::Solver {
//     let x_axis = geom::Axis::new(0., 3., num_x_cells);
//     let y_axis = geom::Axis::new(0., 0.01, 1);
//     let grid = geom::Grid::new(x_axis, y_axis, num_z_cells);
//     let terrain_func = |x: Float, _| -> Float {
//         0.06 * (x * PI / 1.).sin().powi(2)
//         // 0.1 * (-((x - 2.5) / (0.3)).powi(2)).exp()
//     };
//     let static_geometry = geom::StaticGeometry::new(grid, &terrain_func);

//     let initial_height = fields::AreaScalarField::new(static_geometry.grid(), |x, y| {
//         // 0.1 * (x * PI / 5.).cos() + 1.
//         // -0.2 * (-((x - 2.5) / (0.3)).powi(6)).exp() + 0.8
//         amplitude * (-((x - 1.) / (width)).powi(2)).exp() + 0.8 - terrain_func(x, y)
//     });
//     let initial_dynamic_geometry =
//         geom::DynamicGeometry::new_from_height(static_geometry, &initial_height);

//     let mut problem = physics::Problem::default();
//     problem.kinematic_viscosity = kinematic_viscosity;
//     problem.horiz_velocity_boundary_conditions.x = fields::HorizBoundaryConditionPair {
//         lower: fields::BoundaryCondition::NoPenetration,
//         upper: fields::BoundaryCondition::NoPenetration,
//     };

//     let velocity = fields::VolVectorField::new(&initial_dynamic_geometry, |_, _, _| {
//         Vector3::new(0., 0., 0.)
//     });
//     let implicit_solver = problem.make_implicit_solver();
//     // let implicit_solver = crate::implicit::ImplicitSolver {
//     //     linear_solver: crate::linalg::LinearSolver::Direct,
//     //     ignore_max_iters: true,
//     // };
//     physics::Solver::new(
//         problem,
//         implicit_solver,
//         initial_dynamic_geometry,
//         initial_height,
//         velocity,
//     )
// }
