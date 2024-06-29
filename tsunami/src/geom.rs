use flow::{float_consts, Float};
use ndarray as nd;

use crate::{
    render::{Renderable, SphereRenderable, TorusRenderable},
    EARTH_RADIUS_M, MOON_MASS_NONDIMEN, OCEAN_DEPTH_M,
};

const TRACER_STEP: usize = 3;
const TRACER_HISTORY_LENGTH: usize = TRACER_STEP.pow(3);

#[derive(Clone, Copy, Debug, PartialEq)]
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
            GeometryImpl::Sphere(sphere) => sphere.step(),
            GeometryImpl::Torus(torus) => torus.step(),
        }
    }

    pub fn trigger_earthquake(
        &mut self,
        position: three_d::Vec3,
        region_size_rad: Float,
        height_nondimen: Float,
    ) {
        use flow::bases::Basis;

        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => {
                use three_d::InnerSpace;

                let basis = sphere.solver.problem().basis.clone();
                sphere.solver.fields_mut(|mut fields| {
                    let mut new_height = fields.height_grid().to_owned();

                    // Want to generate a power so that a "bump" is generated of half-angle
                    // `half_angle_rad`. Start from FWHM definition:
                    //
                    // ```
                    // 1 / 2 = cos(half angle) ^ (2n)
                    // ```
                    //
                    // Solve for `n`:
                    //
                    // ```
                    // n = log(1 / 2) / (2 * log(cos(half angle)))
                    // ```
                    let half_angle_rad = region_size_rad / 2.;
                    let pow = ((0.5 as Float).ln() / half_angle_rad.cos().ln() / 2.) as f32;
                    log::info!("Setting off earthquake");
                    let click_direction = position.normalize();
                    new_height = &new_height
                        + &basis.make_scalar(|mu, phi| {
                            let cos_theta = mu as f32;
                            let sin_theta = (1. - cos_theta.powi(2)).sqrt() as f32;
                            let cos_phi = phi.cos() as f32;
                            let sin_phi = phi.sin() as f32;
                            let point_direction = three_d::Vec3::new(
                                sin_theta * cos_phi,
                                sin_theta * sin_phi,
                                cos_theta,
                            );
                            height_nondimen
                                * point_direction
                                    .dot(click_direction)
                                    .max(0.)
                                    .powi(2)
                                    .powf(pow) as Float
                        });

                    fields.assign_height(&basis.scalar_to_spectral(&new_height));
                });
            }
            _ => {}
        }
    }

    pub fn set_kinematic_viscosity(&mut self, value: Float) {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => sphere.solver.problem_mut().kinematic_viscosity = value,
            GeometryImpl::Torus(torus) => torus.solver.problem_mut().kinematic_viscosity = value,
        }
    }

    pub fn set_rotation_angular_speed(&mut self, value: Float) {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => {
                sphere.solver.problem_mut().rotation_angular_speed = value
            }
            GeometryImpl::Torus(_) => {}
        }
    }

    pub fn set_lunar_distance(&mut self, value: Float) {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => sphere.solver.problem_mut().lunar_distance = value,
            GeometryImpl::Torus(_) => {}
        }
    }

    pub fn set_velocity_exaggeration_factor(&mut self, value: Float) {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => {
                sphere.solver.problem_mut().velocity_exaggeration_factor = value
            }
            GeometryImpl::Torus(torus) => {
                torus.solver.problem_mut().velocity_exaggeration_factor = value
            }
        }
    }
}

pub struct SphereGeometry {
    step_index: usize,

    base_height: Float,

    solver: flow::physics::Solver<flow::bases::ylm::SphericalHarmonicBasis>,
    curr_fields_snapshot: flow::physics::FieldsSnapshot<flow::bases::ylm::SphericalHarmonicBasis>,
    prev_fields_snapshot: flow::physics::FieldsSnapshot<flow::bases::ylm::SphericalHarmonicBasis>,

    tracers_history: ringbuffer::AllocRingBuffer<nd::Array2<Float>>,

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
        let mu_grid = mu_grid.clone();
        let phi_grid = phi_grid.clone();

        let tracers_history = ringbuffer::AllocRingBuffer::new(TRACER_HISTORY_LENGTH);

        Self {
            step_index: 0,

            base_height,

            solver,
            curr_fields_snapshot,
            prev_fields_snapshot,
            tracers_history,

            mu_grid,
            phi_grid,
        }
    }

    pub fn step(&mut self) {
        use ringbuffer::RingBuffer;
        self.tracers_history
            .push(self.curr_fields_snapshot.fields.tracer_points().clone());

        std::mem::swap(
            &mut self.prev_fields_snapshot,
            &mut self.curr_fields_snapshot,
        );
        self.solver.step();
        self.curr_fields_snapshot = self.solver.fields_snapshot();

        self.step_index += 1;
    }

    pub fn make_renderables(&self, num: usize) -> impl Iterator<Item = SphereRenderable> + '_ {
        flow::physics::interp_between(&self.prev_fields_snapshot, &self.curr_fields_snapshot, num)
            .map(|fields_snapshot| {
                use flow::bases::Basis;
                use ringbuffer::RingBuffer;

                let height_grid = fields_snapshot.fields.height_grid();
                let tracer_points_history_mu_phi = self
                    .tracers_history
                    .iter()
                    .step_by(TRACER_STEP)
                    .cloned()
                    .collect::<Vec<_>>();
                let tracer_heights_history: Vec<_> = tracer_points_history_mu_phi
                    .iter()
                    .map(|tracer_points_mu_phi| {
                        self.solver
                            .problem()
                            .basis
                            .scalar_to_points(&height_grid, tracer_points_mu_phi.view())
                    })
                    .collect();

                SphereRenderable {
                    t: fields_snapshot.t,
                    base_height: self.base_height,
                    mu_grid: self.mu_grid.clone(),
                    phi_grid: self.phi_grid.clone(),
                    height_array: height_grid,
                    tracer_points_history_mu_phi,
                    tracer_heights_history,
                }
            })
    }

    fn make_solver(
        resolution_level: u32,
    ) -> (
        Float,
        flow::physics::Solver<flow::bases::ylm::SphericalHarmonicBasis>,
    ) {
        use flow::bases::Basis;

        let max_l = 2usize.pow(resolution_level);
        let base_height = 1.;

        // Dummy values, reset before taking a simulation step in `lib.rs`.
        let kinematic_viscosity = 0.;
        let rotation_angular_speed = 0.;
        let lunar_distance = 1e6;
        let velocity_exaggeration_factor = 1e4;

        let basis = std::sync::Arc::new(flow::bases::ylm::SphericalHarmonicBasis::new(max_l));
        let terrain_height = basis.scalar_to_spectral(&basis.make_scalar(|_, _| 1.));
        let mut initial_fields = flow::physics::Fields::zeros(basis.clone());
        let initial_height_grid = basis.make_scalar(|_, _| base_height);
        initial_fields.assign_height(&basis.scalar_to_spectral(&initial_height_grid));
        initial_fields.assign_tracers(basis.make_random_points().view());
        let problem = flow::physics::Problem {
            basis,
            terrain_height,
            kinematic_viscosity,
            rotation_angular_speed,
            tidal_prefactor: 4.
                * float_consts::PI
                * MOON_MASS_NONDIMEN
                * (EARTH_RADIUS_M / OCEAN_DEPTH_M),
            lunar_distance,
            velocity_exaggeration_factor,
            height_tolerances: flow::physics::Tolerances {
                rel: 1e-5,
                abs: 1e-10,
            },
            velocity_tolerances: flow::physics::Tolerances {
                rel: 1e-4,
                abs: 1e-4,
            },
            tracers_tolerances: flow::physics::Tolerances { rel: 1e0, abs: 1e0 },
        };
        (
            base_height,
            flow::physics::Solver::new(problem, initial_fields),
        )
    }
}

struct TorusGeometry {
    step_index: usize,

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

    tracers_history: ringbuffer::AllocRingBuffer<nd::Array2<Float>>,
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
        let theta_grid = axes[0] / major_radius;
        let phi_grid = axes[1] / minor_radius;

        let tracers_history = ringbuffer::AllocRingBuffer::new(TRACER_HISTORY_LENGTH);

        Self {
            step_index: 0,

            base_height,

            major_radius,
            minor_radius,

            theta_grid,
            phi_grid,

            solver,
            prev_fields_snapshot,
            curr_fields_snapshot,

            tracers_history,
        }
    }

    pub fn step(&mut self) {
        use ringbuffer::RingBuffer;
        self.tracers_history
            .push(self.curr_fields_snapshot.fields.tracer_points());

        std::mem::swap(
            &mut self.prev_fields_snapshot,
            &mut self.curr_fields_snapshot,
        );
        self.solver.step();
        self.curr_fields_snapshot = self.solver.fields_snapshot();

        self.step_index += 1;
    }

    pub fn make_renderables(&self, num: usize) -> impl Iterator<Item = TorusRenderable> + '_ {
        flow::physics::interp_between(&self.prev_fields_snapshot, &self.curr_fields_snapshot, num)
            .map(|fields_snapshot| {
                use flow::bases::Basis;
                use ringbuffer::RingBuffer;

                let height_grid = fields_snapshot.fields.height_grid();

                let mut tracer_points_history = self
                    .tracers_history
                    .iter()
                    .step_by(TRACER_STEP)
                    .cloned()
                    .collect::<Vec<_>>();
                let tracer_heights_history: Vec<_> = tracer_points_history
                    .iter()
                    .map(|tracer_points| {
                        self.solver
                            .problem()
                            .basis
                            .scalar_to_points(&height_grid, tracer_points.view())
                    })
                    .collect();
                for tracer_points in &mut tracer_points_history {
                    let mut tracer_points_theta = tracer_points.slice_mut(nd::s![0, ..]);
                    tracer_points_theta /= self.major_radius;
                    let mut tracer_points_phi = tracer_points.slice_mut(nd::s![1, ..]);
                    tracer_points_phi /= self.minor_radius;
                }

                TorusRenderable {
                    t: fields_snapshot.t,
                    base_height: self.base_height,
                    theta_grid: self.theta_grid.clone(),
                    phi_grid: self.phi_grid.clone(),
                    major_radius: self.major_radius,
                    minor_radius: self.minor_radius,
                    height_array: height_grid,
                    tracer_points_history_theta_phi: tracer_points_history,
                    tracer_heights_history,
                }
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
        let base_height = 1.;
        let amplitude = base_height * 0.2;
        let bump_size = 0.1;
        let kinematic_viscosity = 0.;
        let rotation_angular_speed = 0.;

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
                - amplitude
                    * (PI * (x / lengths[0] - 0.11))
                        .sin()
                        .powi(2)
                        .powf(pow(bump_size, lengths[0]))
                    * (PI * y / lengths[1])
                        .sin()
                        .powi(2)
                        .powf(pow(bump_size, lengths[1]))
        })));
        initial_fields.assign_tracers(basis.make_random_points().view());
        let problem = flow::physics::Problem {
            basis,
            terrain_height,
            kinematic_viscosity,
            rotation_angular_speed,
            lunar_distance: 1.,
            tidal_prefactor: 0.,
            velocity_exaggeration_factor: 1e4,
            height_tolerances: flow::physics::Tolerances {
                rel: 1e-3,
                abs: 1e-3,
            },
            velocity_tolerances: flow::physics::Tolerances {
                rel: 1e-3,
                abs: 1e-3,
            },
            tracers_tolerances: flow::physics::Tolerances { rel: 1e0, abs: 1e0 },
        };
        (
            base_height,
            flow::physics::Solver::new(problem, initial_fields),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_no_crash() {
        let mut sphere = SphereGeometry::new(6);
        let _: Vec<_> = sphere.make_renderables(3).collect();
        for _ in 0..20 {
            sphere.step();
        }
    }
}
