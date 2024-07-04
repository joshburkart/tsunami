use crate::{bases, float_consts::PI, physics, Float};

pub fn bump_2d_spectral(
    num_points: usize,
    kinematic_viscosity: Float,
    base_height: Float,
    amplitude: Float,
) -> physics::Solver<bases::fourier::RectangularPeriodicBasis> {
    use bases::Basis;

    let lengths = [5., 15.];
    let bump_size = 0.3;
    let num_tracers = 100;

    // Want to generate a power so that a periodic "bump" is generated of width
    // `bump_size`. Start from FWHM definition:
    //
    // ```
    // 1 / 2 = sin(pi * (1 / 2 - bump_size / lengths[i])) ^ (2n)
    // ```
    //
    // Solve for `n`:
    //
    // ```
    // n = log(1 / 2) / (2 * log(sin(pi * (1 / 2 - bump_size / lengths[i]))))
    // ```
    let pow = |bump_size: Float, length: Float| {
        (0.5 as Float).ln() / (PI * (0.5 - bump_size / length)).sin().ln() / 2.
    };
    let basis = std::sync::Arc::new(bases::fourier::RectangularPeriodicBasis::new(
        [num_points, num_points],
        lengths,
    ));
    let terrain_height = basis.scalar_to_spectral(&basis.make_scalar(|_, _| 0.));
    let mut initial_fields = physics::Fields::zeros(basis.clone(), num_tracers);
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
        num_tracers,
        terrain_height,
        kinematic_viscosity,
        tidal_prefactor: 0.,
        lunar_distance: 1.,
        velocity_exaggeration_factor: 1e4,
        rotation_angular_speed: 0.,
        height_tolerances: Default::default(),
        velocity_tolerances: Default::default(),
        tracers_tolerances: Default::default(),
    };
    physics::Solver::new(problem, initial_fields)
}
