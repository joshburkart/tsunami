use flow::{bases, float_consts::PI, physics, Float};
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
    pub fn integrate(&mut self) -> RectangularFields {
        self.solver.step();
        let fields_snapshot = self.solver.fields_snapshot();
        RectangularFields {
            height: fields_snapshot.fields.height_grid(),
            velocity: fields_snapshot.fields.velocity_grid(),
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

    // Want to generate a power so that a periodic "bump" is generated of width
    // `bump_size`. Start from FWHM definition, given circumference =
    // lengths[i]:
    //
    // ```
    // 1 / 2 = cos(pi * bump_size / lengths[i]) ^ (2n)
    // ```
    //
    // Solve for `n`:
    //
    // ```
    // n = log(1 / 2) / (2 * log(cos(pi * bump_size / lengths[i])))
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
        lunar_distance: 1.,
        tidal_prefactor: 0.,
        kinematic_viscosity,
        rotation_angular_speed: 0.,
        height_tolerances: physics::Tolerances {
            rel: 1e-4,
            abs: 1e-4,
        },
        velocity_tolerances: physics::Tolerances {
            rel: 1e-4,
            abs: 1e-4,
        },
        tracers_tolerances: physics::Tolerances {
            rel: 1e-4,
            abs: 1e-4,
        },
    };
    let solver = physics::Solver::new(problem, initial_fields);
    RectangularSolver { solver }
}
