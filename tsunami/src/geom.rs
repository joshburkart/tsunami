use flow::{float_consts, Float};
use ndarray as nd;

use crate::render::{Renderable, SphereRenderable, TorusRenderable};

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

    pub fn make_renderables(&self, num: usize) -> Vec<Renderable> {
        match &self.0 {
            GeometryImpl::Sphere(sphere) => sphere
                .make_renderables(num)
                .map(Renderable::Sphere)
                .collect(),
            GeometryImpl::Torus(torus) => {
                torus.make_renderables(num).map(Renderable::Torus).collect()
            }
        }
    }

    pub fn integrate(&mut self) {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => sphere.integrate(),
            GeometryImpl::Torus(torus) => torus.integrate(),
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
    curr_fields_snapshot: flow::physics::FieldsSnapshot<flow::bases::ylm::SphericalHarmonicBasis>,
    prev_fields_snapshot: flow::physics::FieldsSnapshot<flow::bases::ylm::SphericalHarmonicBasis>,

    mu_grid: nd::Array1<Float>,
    phi_grid: nd::Array1<Float>,
}

impl SphereGeometry {
    pub fn new(resolution_level: u32) -> Self {
        use flow::bases::Basis;

        let (base_height, solver) = Self::make_solver(resolution_level);
        let prev_fields_snapshot = solver.fields_snapshot();
        let curr_fields_snapshot = solver.fields_snapshot();

        let [mu_grid, phi_grid] = solver.problem().basis.axes();

        Self {
            base_height,

            solver,
            curr_fields_snapshot,
            prev_fields_snapshot,

            mu_grid,
            phi_grid,
        }
    }

    pub fn integrate(&mut self) {
        std::mem::swap(
            &mut self.prev_fields_snapshot,
            &mut self.curr_fields_snapshot,
        );
        self.solver.integrate();
        self.curr_fields_snapshot = self.solver.fields_snapshot();
    }

    pub fn make_renderables(&self, num: usize) -> impl Iterator<Item = SphereRenderable> + '_ {
        flow::physics::interp_between(&self.prev_fields_snapshot, &self.curr_fields_snapshot, num)
            .map(|fields_snapshot| SphereRenderable {
                base_height: self.base_height,
                mu_grid: self.mu_grid.clone(),
                phi_grid: self.phi_grid.clone(),
                height_array: fields_snapshot.fields.height_grid(),
            })
    }

    fn make_solver(
        resolution_level: u32,
    ) -> (
        Float,
        flow::physics::Solver<flow::bases::ylm::SphericalHarmonicBasis>,
    ) {
        use flow::{bases::Basis, float_consts::PI};

        let max_l = 2usize.pow(resolution_level);
        let base_height = 3.;
        let amplitude = 0.01;
        let bump_size = 0.01;
        let kinematic_viscosity = 1e-4; // TODO
        let grav_accel = 9.8;

        // TODO DRY
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
            rtol: 1e-5,
            atol: 1e-8,
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
    prev_fields_snapshot:
        flow::physics::FieldsSnapshot<flow::bases::fourier::RectangularPeriodicBasis>,
    curr_fields_snapshot:
        flow::physics::FieldsSnapshot<flow::bases::fourier::RectangularPeriodicBasis>,
}

impl TorusGeometry {
    pub fn new(resolution_level: u32) -> Self {
        use flow::bases::Basis;

        let (base_height, solver) = Self::make_solver(resolution_level);
        let prev_fields_snapshot = solver.fields_snapshot();
        let curr_fields_snapshot = prev_fields_snapshot.clone();

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
            prev_fields_snapshot,
            curr_fields_snapshot,
        }
    }

    pub fn integrate(&mut self) {
        std::mem::swap(
            &mut self.prev_fields_snapshot,
            &mut self.curr_fields_snapshot,
        );
        self.solver.integrate();
        self.curr_fields_snapshot = self.solver.fields_snapshot();
    }

    pub fn make_renderables(&self, num: usize) -> impl Iterator<Item = TorusRenderable> + '_ {
        flow::physics::interp_between(&self.prev_fields_snapshot, &self.curr_fields_snapshot, num)
            .map(|fields_snapshot| TorusRenderable {
                base_height: self.base_height,
                theta_grid: self.theta_grid.clone(),
                phi_grid: self.phi_grid.clone(),
                major_radius: self.major_radius,
                minor_radius: self.minor_radius,
                height_array: fields_snapshot.fields.height_grid(),
            })
    }

    fn make_solver(
        resolution_level: u32,
    ) -> (
        Float,
        flow::physics::Solver<flow::bases::fourier::RectangularPeriodicBasis>,
    ) {
        use flow::{bases::Basis, float_consts::PI};

        let num_points = [
            2usize.pow(resolution_level + 1),
            2usize.pow(resolution_level),
        ];
        let lengths = [15., 5.];
        let base_height = 5.;
        let amplitude = base_height * 0.2;
        let bump_size = 0.15;
        let kinematic_viscosity = 1e-2;

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
