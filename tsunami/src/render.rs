use flow::{float_consts, Float};
use ndarray as nd;
use three_d::{CpuMesh, Indices, Positions, Vector3};

pub struct Rendering {
    pub points: Vec<Vector3<Float>>,
    pub mesh: CpuMesh,
}

#[derive(Clone)]
pub enum Renderable {
    Sphere(SphereRenderable),
    Torus(TorusRenderable),
}

impl Renderable {
    pub fn render(&self, height_exaggeration_factor: Float) -> Rendering {
        match self {
            Renderable::Sphere(sphere) => sphere.render(height_exaggeration_factor),
            Renderable::Torus(torus) => torus.render(height_exaggeration_factor),
        }
    }
}

pub fn interp_between(
    renderable_prev: &Renderable,
    renderable_next: &Renderable,
    t_between: Float,
) -> Renderable {
    match renderable_next {
        Renderable::Sphere(sphere_next) => {
            if let Renderable::Sphere(sphere_prev) = renderable_prev {
                if sphere_prev.mu_grid.len() == sphere_next.mu_grid.len()
                    && sphere_prev.phi_grid.len() == sphere_next.phi_grid.len()
                {
                    Renderable::Sphere(SphereRenderable {
                        base_height: sphere_next.base_height,
                        mu_grid: sphere_next.mu_grid.clone(),
                        phi_grid: sphere_next.phi_grid.clone(),
                        fields_snapshot: flow::physics::interp_between(
                            &sphere_prev.fields_snapshot,
                            &sphere_next.fields_snapshot,
                            t_between,
                        ),
                    })
                } else {
                    renderable_next.clone()
                }
            } else {
                renderable_next.clone()
            }
        }
        Renderable::Torus(torus_next) => {
            if let Renderable::Torus(torus_prev) = renderable_prev {
                if torus_prev.theta_grid.len() == torus_next.theta_grid.len()
                    && torus_prev.phi_grid.len() == torus_next.phi_grid.len()
                {
                    Renderable::Torus(TorusRenderable {
                        base_height: torus_next.base_height,
                        major_radius: torus_next.major_radius,
                        minor_radius: torus_prev.minor_radius,
                        theta_grid: torus_next.theta_grid.clone(),
                        phi_grid: torus_next.phi_grid.clone(),
                        fields_snapshot: flow::physics::interp_between(
                            &torus_prev.fields_snapshot,
                            &torus_next.fields_snapshot,
                            t_between,
                        ),
                    })
                } else {
                    renderable_next.clone()
                }
            } else {
                renderable_next.clone()
            }
        }
    }
}

#[derive(Clone)]
pub struct SphereRenderable {
    pub base_height: Float,

    pub mu_grid: nd::Array1<Float>,
    pub phi_grid: nd::Array1<Float>,

    pub fields_snapshot: flow::physics::FieldsSnapshot<flow::bases::ylm::SphericalHarmonicBasis>,
}

impl SphereRenderable {
    pub fn render(&self, height_exaggeration_factor: Float) -> Rendering {
        let num_mu = self.mu_grid.len();
        let num_phi = self.phi_grid.len();

        let make_index = |i, j| (i * num_phi + (j % num_phi)) as u32;

        let height_array = self.fields_snapshot.fields.height_grid();

        let mut points = self.make_points(&height_array, height_exaggeration_factor);
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
            positions: Positions::F32(points.clone()),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();

        Rendering { points, mesh }
    }

    fn make_points(
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
}

#[derive(Clone)]
pub struct TorusRenderable {
    pub base_height: Float,

    pub theta_grid: nd::Array1<Float>,
    pub phi_grid: nd::Array1<Float>,

    pub major_radius: Float,
    pub minor_radius: Float,

    pub fields_snapshot:
        flow::physics::FieldsSnapshot<flow::bases::fourier::RectangularPeriodicBasis>,
}

impl TorusRenderable {
    pub fn render(&self, height_exaggeration_factor: Float) -> Rendering {
        let num_theta = self.theta_grid.len();
        let num_phi = self.phi_grid.len();
        let num_cell_vertices = num_theta * num_phi;

        let make_index = |i, j| (i % num_theta) * num_phi + (j % num_phi);

        let height_array = self.fields_snapshot.fields.height_grid();

        let points = self.make_points(&height_array, height_exaggeration_factor);
        let mut augmented_points = points.clone();
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
                augmented_points.push(self.make_point(
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
                // Index of interpolated center point within flat `augmented_points` vector.
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
            positions: Positions::F32(augmented_points),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();

        Rendering { points, mesh }
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
}
