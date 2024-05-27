use crate::{bases, physics, Float};

use crate::float_consts::PI;

pub fn bump_2d_spectral(
    num_points: usize,
    kinematic_viscosity: Float,
    base_height: Float,
    amplitude: Float,
) -> physics::Solver<bases::RectangularPeriodicBasis> {
    use bases::Basis;

    let lengths = [5., 15.];
    let bump_size = 0.3;

    // Want to generate a power so that a periodic "bump" is generated of width `bump_size`. Start from FWHM definition:
    //
    // ```
    // 1/2 = sin(pi * (1/2 - bump_size / lengths[i]))^(2n)
    // ```
    //
    // Solve for `n`:
    //
    // ```
    // n = log(1/2) / (2 * log(sin(pi * (1/2 - bump_size / lengths[i]))))
    // ```
    let pow = |bump_size: Float, length: Float| {
        (0.5 as Float).ln() / (PI * (0.5 - bump_size / length)).sin().ln() / 2.
    };
    let basis = std::sync::Arc::new(bases::RectangularPeriodicBasis::new(
        [num_points, num_points],
        lengths,
    ));
    let terrain_height = basis.scalar_to_spectral(&basis.make_scalar(|_, _| 0.));
    let mut initial_fields = physics::Fields::zeros(basis.clone());
    initial_fields.assign_height(&basis.scalar_to_spectral(&basis.make_scalar(|x, y| {
        base_height
            + amplitude
                * (PI * (x / lengths[0] - 0.21))
                    .sin()
                    .powi(2)
                    .powf(pow(bump_size, lengths[0]))
                * (PI * y / lengths[1])
                    .sin()
                    .powi(2)
                    .powf(pow(bump_size, lengths[1]))
    })));
    let problem = physics::Problem {
        basis,
        rain_rate: None,
        terrain_height,
        grav_accel: 9.8,
        kinematic_viscosity,
        rtol: 1e-3,
        atol: 1e-4,
    };
    physics::Solver::new(problem, initial_fields)
}
