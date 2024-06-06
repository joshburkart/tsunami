use crate::render::{Renderable, SphereRenderable, TorusRenderable};
use flow::{float_consts, Float};
use ndarray as nd;

#[derive(Clone, Copy, PartialEq)]
pub enum GeometryType {
    Sphere,
    Torus,
}

pub struct Geometry(GeometryImpl);

enum GeometryImpl {
    Sphere(SphereGeometry),
    Torus(TorusGeometry),
}

impl Geometry {
    pub fn new(geometry_type: GeometryType, resolution_level: u32) -> Self {
        Self(match geometry_type {
            GeometryType::Sphere => GeometryImpl::Sphere(SphereGeometry::new(resolution_level)),
            GeometryType::Torus => GeometryImpl::Torus(TorusGeometry::new(resolution_level)),
        })
    }

    pub fn integrate(&mut self) -> Renderable {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => Renderable::Sphere(SphereRenderable {
                base_height: sphere.base_height,
                mu_grid: sphere.mu_grid.clone(),
                phi_grid: sphere.phi_grid.clone(),
                fields_snapshot: sphere.solver.integrate(),
            }),
            GeometryImpl::Torus(torus) => Renderable::Torus(TorusRenderable {
                base_height: torus.base_height,
                major_radius: torus.major_radius,
                minor_radius: torus.minor_radius,
                theta_grid: torus.theta_grid.clone(),
                phi_grid: torus.phi_grid.clone(),
                fields_snapshot: torus.solver.integrate(),
            }),
        }
    }
    pub fn set_kinematic_viscosity(&mut self, value: Float) {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => sphere.solver.problem_mut().kinematic_viscosity = value,
            GeometryImpl::Torus(torus) => torus.solver.problem_mut().kinematic_viscosity = value,
        }
    }
}

struct SphereGeometry {
    base_height: Float,

    solver: flow::physics::Solver<flow::bases::ylm::SphericalHarmonicBasis>,

    mu_grid: nd::Array1<Float>,
    phi_grid: nd::Array1<Float>,
}

impl SphereGeometry {
    pub fn new(resolution_level: u32) -> Self {
        use flow::bases::Basis;

        let (base_height, solver) = Self::make_solver(resolution_level);

        let [mu_grid, phi_grid] = solver.problem().basis.axes();

        Self {
            base_height,

            solver,

            mu_grid,
            phi_grid,
        }
    }

    pub fn make_solver(
        resolution_level: u32,
    ) -> (
        Float,
        flow::physics::Solver<flow::bases::ylm::SphericalHarmonicBasis>,
    ) {
        use flow::bases::Basis;
        use flow::float_consts::PI;

        let max_l = 2usize.pow(resolution_level);
        let base_height = 3.;
        let amplitude = 0.01;
        let bump_size = 0.01;
        let kinematic_viscosity = 1e-4; // TODO
        let grav_accel = 9.8;

        // TODO DRY
        // Want to generate a power so that a periodic "bump" is generated of width `bump_size`. Start
        // from FWHM definition:
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
        let pow = |bump_size: flow::Float, length: flow::Float| {
            (0.5 as flow::Float).ln() / (PI * (0.5 - bump_size / length)).sin().ln() / 2.
        };
        let basis = std::sync::Arc::new(flow::bases::ylm::SphericalHarmonicBasis::new(max_l));
        let terrain_height = basis.scalar_to_spectral(&basis.make_scalar(|_, _| 1.));
        let mut initial_fields = flow::physics::Fields::zeros(basis.clone());
        let initial_height_grid = basis.make_scalar(|mu, phi| {
            base_height
                + amplitude
                    * ((mu.acos() + float_consts::PI / 4.)
                        .sin()
                        .max(0.)
                        .powi(2)
                        .powf(pow(bump_size, 1.))
                        * phi.sin().max(0.).powi(2).powf(pow(bump_size, 1.))
                        + (mu.acos() - float_consts::PI / 4.)
                            .sin()
                            .max(0.)
                            .powi(2)
                            .powf(pow(bump_size, 1.))
                            * phi.cos().max(0.).powi(2).powf(pow(bump_size, 1.)))
        });
        initial_fields.assign_height(&basis.scalar_to_spectral(&initial_height_grid));
        let problem = flow::physics::Problem {
            basis,
            terrain_height,
            grav_accel,
            kinematic_viscosity,
            rtol: 1e-3,
            atol: 1e-7,
        };
        (
            base_height,
            flow::physics::Solver::new(problem, initial_fields),
        )
    }
}

struct TorusGeometry {
    base_height: Float,

    major_radius: Float,
    minor_radius: Float,

    theta_grid: nd::Array1<Float>,
    phi_grid: nd::Array1<Float>,

    solver: flow::physics::Solver<flow::bases::fourier::RectangularPeriodicBasis>,
}

impl TorusGeometry {
    pub fn new(resolution_level: u32) -> Self {
        use flow::bases::Basis;

        let (base_height, solver) = Self::make_solver(resolution_level);

        let axes = solver.problem().basis.axes();

        let major_radius = axes[0][1] * axes[0].len() as Float / float_consts::TAU;
        let minor_radius = axes[1][1] * axes[1].len() as Float / float_consts::TAU;
        let theta_grid = &axes[0] / major_radius;
        let phi_grid = &axes[1] / minor_radius;

        Self {
            base_height,

            major_radius,
            minor_radius,

            theta_grid,
            phi_grid,

            solver,
        }
    }

    fn make_solver(
        resolution_level: u32,
    ) -> (
        Float,
        flow::physics::Solver<flow::bases::fourier::RectangularPeriodicBasis>,
    ) {
        use flow::bases::Basis;
        use flow::float_consts::PI;

        let num_points = [
            2usize.pow(resolution_level + 1),
            2usize.pow(resolution_level),
        ];
        let lengths = [15., 5.];
        let base_height = 5.;
        let amplitude = base_height * 0.2;
        let bump_size = 0.15;
        let kinematic_viscosity = 1e-2;

        // Want to generate a power so that a periodic "bump" is generated of width `bump_size`. Start
        // from FWHM definition:
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
        let pow = |bump_size: flow::Float, length: flow::Float| {
            (0.5 as flow::Float).ln() / (PI * (0.5 - bump_size / length)).sin().ln() / 2.
        };
        let basis = std::sync::Arc::new(flow::bases::fourier::RectangularPeriodicBasis::new(
            num_points, lengths,
        ));
        let terrain_height = basis.scalar_to_spectral(&basis.make_scalar(|_, _| 0.));
        let mut initial_fields = flow::physics::Fields::zeros(basis.clone());
        initial_fields.assign_height(&basis.scalar_to_spectral(&basis.make_scalar(|x, y| {
            base_height
                + amplitude
                    * (PI * (x / lengths[0] - 0.11))
                        .sin()
                        .powi(2)
                        .powf(pow(bump_size, lengths[0]))
                    * (PI * y / lengths[1])
                        .sin()
                        .powi(2)
                        .powf(pow(bump_size, lengths[1]))
        })));
        let problem = flow::physics::Problem {
            basis,
            terrain_height,
            grav_accel: 9.8,
            kinematic_viscosity,
            rtol: 1e-2,
            atol: 1e-2,
        };
        (
            base_height,
            flow::physics::Solver::new(problem, initial_fields),
        )
    }
}
