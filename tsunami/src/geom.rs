use flow::{float_consts, Float};
use three_d::{CpuMesh, Indices, Positions, Vector3};

use ndarray as nd;

#[derive(Clone, Copy, PartialEq)]
pub enum GeometryType {
    Sphere,
    Torus,
}

pub struct Geometry(GeometryImpl);

enum GeometryImpl {
    Sphere(Sphere),
    Torus(Torus),
}

impl Geometry {
    pub fn new(geometry_type: GeometryType, resolution_level: u32) -> Self {
        Self(match geometry_type {
            GeometryType::Sphere => GeometryImpl::Sphere(Sphere::new(resolution_level)),
            GeometryType::Torus => GeometryImpl::Torus(Torus::new(resolution_level)),
        })
    }

    pub fn integrate(&mut self, delta_t: Float) {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => sphere.solver.integrate(delta_t),
            GeometryImpl::Torus(torus) => torus.solver.integrate(delta_t),
        }
    }
    pub fn height_grid(&self) -> nd::Array2<Float> {
        match &self.0 {
            GeometryImpl::Sphere(sphere) => sphere.solver.fields().height_grid(),
            GeometryImpl::Torus(torus) => torus.solver.fields().height_grid(),
        }
    }
    pub fn set_kinematic_viscosity(&mut self, value: Float) {
        match &mut self.0 {
            GeometryImpl::Sphere(sphere) => sphere.solver.problem_mut().kinematic_viscosity = value,
            GeometryImpl::Torus(torus) => torus.solver.problem_mut().kinematic_viscosity = value,
        }
    }

    pub fn make_mesh(
        &self,
        height_array: &nd::Array2<Float>,
        height_exaggeration_factor: Float,
    ) -> CpuMesh {
        match &self.0 {
            GeometryImpl::Sphere(sphere) => {
                sphere.make_mesh(height_array, height_exaggeration_factor)
            }
            GeometryImpl::Torus(torus) => torus.make_mesh(height_array, height_exaggeration_factor),
        }
    }

    pub fn make_points(
        &self,
        height_array: &nd::Array2<Float>,
        height_exaggeration_factor: Float,
    ) -> Vec<Vector3<Float>> {
        match &self.0 {
            GeometryImpl::Sphere(sphere) => {
                sphere.make_points(height_array, height_exaggeration_factor)
            }
            GeometryImpl::Torus(torus) => {
                torus.make_points(height_array, height_exaggeration_factor)
            }
        }
    }
}

struct Sphere {
    base_height: Float,

    solver: flow::physics::Solver<flow::bases::ylm::SphericalHarmonicBasis>,

    mu_grid: nd::Array1<Float>,
    phi_grid: nd::Array1<Float>,
}

impl Sphere {
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

    pub fn make_mesh(
        &self,
        height_array: &nd::Array2<Float>,
        height_exaggeration_factor: Float,
    ) -> CpuMesh {
        let num_mu = self.mu_grid.len();
        let num_phi = self.phi_grid.len();

        let make_index = |i, j| (i * num_phi + (j % num_phi)) as u32;

        let mut points = self.make_points(height_array, height_exaggeration_factor);
        // Add bottom point.
        let bottom = points.len() as u32;
        points.push(self.make_point(
            -1.,
            0.,
            height_array.slice(nd::s![0, ..]).iter().sum::<Float>() / num_phi as Float,
            height_exaggeration_factor,
        ));
        // Add top point.
        let top = points.len() as u32;
        points.push(self.make_point(
            1.,
            0.,
            height_array.slice(nd::s![-1, ..]).iter().sum::<Float>() / num_phi as Float,
            height_exaggeration_factor,
        ));

        let mut indices = Vec::new();
        for i in 0..num_mu - 1 {
            for j in 0..num_phi {
                let ll = make_index(i, j);
                let lr = make_index(i + 1, j);
                let ul = make_index(i, j + 1);
                let ur = make_index(i + 1, j + 1);

                // Lower left triangle.
                indices.push(ll);
                indices.push(lr);
                indices.push(ul);
                // Upper right triangle.
                indices.push(lr);
                indices.push(ur);
                indices.push(ul);
            }
        }
        for j in 0..num_phi {
            // Add bottom cap.
            let ul = make_index(0, j);
            let ur = make_index(0, j + 1);
            indices.push(ul);
            indices.push(ur);
            indices.push(bottom);

            // Add top cap.
            let ll = make_index(num_mu - 1, j);
            let lr = make_index(num_mu - 1, j + 1);
            indices.push(lr);
            indices.push(ll);
            indices.push(top);
        }

        let mut mesh = CpuMesh {
            indices: Indices::U32(indices),
            positions: Positions::F32(points),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();
        mesh
    }

    pub fn make_points(
        &self,
        height_array: &nd::Array2<Float>,
        height_exaggeration_factor: Float,
    ) -> Vec<Vector3<Float>> {
        let mut points = Vec::with_capacity(self.mu_grid.len() * self.phi_grid.len());
        for (i, &mu) in self.mu_grid.iter().enumerate() {
            for (j, &phi) in self.phi_grid.iter().enumerate() {
                let height = height_array[[i, j]];
                points.push(self.make_point(mu, phi, height, height_exaggeration_factor));
            }
        }
        points
    }

    fn make_point(
        &self,
        mu: Float,
        phi: Float,
        height: Float,
        height_exaggeration_factor: Float,
    ) -> three_d::Vector3<Float> {
        let sin_theta = (1. - mu.powi(2)).sqrt();
        let radially_out = Vector3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), mu);
        radially_out * (1. + (height - self.base_height) * height_exaggeration_factor)
    }

    fn make_solver(
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
            rtol: 1e-2,
            atol: 1e-7,
        };
        (
            base_height,
            flow::physics::Solver::new(problem, initial_fields),
        )
    }
}

struct Torus {
    base_height: Float,

    major_radius: Float,
    minor_radius: Float,

    theta_grid: nd::Array1<Float>,
    phi_grid: nd::Array1<Float>,

    solver: flow::physics::Solver<flow::bases::fourier::RectangularPeriodicBasis>,
}

impl Torus {
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

    pub fn make_mesh(
        &self,
        height_array: &nd::Array2<Float>,
        height_exaggeration_factor: Float,
    ) -> CpuMesh {
        let num_theta = self.theta_grid.len();
        let num_phi = self.phi_grid.len();
        let num_cell_vertices = num_theta * num_phi;

        let make_index = |i, j| (i % num_theta) * num_phi + (j % num_phi);

        let mut points = self.make_points(height_array, height_exaggeration_factor);
        let height_flat = height_array.as_slice().unwrap();

        let mut indices = Vec::new();
        for i in 0..num_theta {
            for j in 0..num_phi {
                let ll = make_index(i, j);
                let lr = make_index(i + 1, j);
                let ul = make_index(i, j + 1);
                let ur = make_index(i + 1, j + 1);

                // Augment with linearly interpolated cell centers for improved visuals.
                let height_c =
                    0.25 * (height_flat[ll] + height_flat[lr] + height_flat[ul] + height_flat[ur]);
                points.push(self.make_point(
                    (self.theta_grid[i]
                        + if i + 1 < num_theta {
                            self.theta_grid[i + 1]
                        } else {
                            self.theta_grid[0] + float_consts::TAU
                        })
                        / 2.,
                    (self.phi_grid[j]
                        + if j + 1 < num_phi {
                            self.phi_grid[j + 1]
                        } else {
                            self.phi_grid[0] + float_consts::TAU
                        })
                        / 2.,
                    height_c,
                    height_exaggeration_factor,
                ));
                // Index of interpolated center point within flat `points` vector.
                let c = num_cell_vertices + make_index(i, j);

                // Bottom triangle.
                indices.push(ll as u32);
                indices.push(lr as u32);
                indices.push(c as u32);
                // Left triangle.
                indices.push(ll as u32);
                indices.push(c as u32);
                indices.push(ul as u32);
                // Top triangle.
                indices.push(ul as u32);
                indices.push(c as u32);
                indices.push(ur as u32);
                // Right triangle.
                indices.push(lr as u32);
                indices.push(ur as u32);
                indices.push(c as u32);
            }
        }

        let mut mesh = CpuMesh {
            indices: Indices::U32(indices),
            positions: Positions::F32(points),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();
        mesh
    }

    pub fn make_points(
        &self,
        height_array: &nd::Array2<Float>,
        height_exaggeration_factor: Float,
    ) -> Vec<Vector3<Float>> {
        let mut points = Vec::with_capacity(self.theta_grid.len() * self.phi_grid.len());
        for (i, &theta) in self.theta_grid.iter().enumerate() {
            for (j, &phi) in self.phi_grid.iter().enumerate() {
                let height = height_array[[i, j]];
                points.push(self.make_point(theta, phi, height, height_exaggeration_factor));
            }
        }
        points
    }

    fn make_point(
        &self,
        theta: Float,
        phi: Float,
        height: Float,
        height_exaggeration_factor: Float,
    ) -> three_d::Vector3<Float> {
        const SCALE: Float = 0.4;
        let radially_out = Vector3::new(-theta.cos(), theta.sin(), 0.);
        SCALE
            * (self.major_radius * radially_out
                + (self.minor_radius
                    + (height_exaggeration_factor / 100.) * (height - self.base_height))
                    * (-phi.cos() * radially_out + phi.sin() * Vector3::unit_z()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical() {
        let mut sphere = Sphere::new(5);
        let height_array = sphere.solver.fields().height_grid();
        sphere.make_mesh(&height_array, 2.);
        sphere.solver.integrate(0.01);
    }

    #[test]
    fn test_toroidal() {
        let mut torus = Torus::new(5);
        let height_array = torus.solver.fields().height_grid();
        torus.make_mesh(&height_array, 2.);
        torus.solver.integrate(0.01);
    }
}
