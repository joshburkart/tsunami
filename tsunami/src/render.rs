use flow::{float_consts, Float};
use ndarray as nd;
use three_d::{CpuMesh, Indices, Positions, Vector3};

#[derive(Clone)]
pub struct RenderingData {
    pub quadrature_points: Vec<Vector3<Float>>,
    pub tracer_points: TracerPoints,
    pub mesh: CpuMesh,
}

#[derive(Clone)]
pub enum Renderable {
    Sphere(SphereRenderable),
    Torus(TorusRenderable),
}

impl Renderable {
    pub fn make_rendering_data(&self, height_exaggeration_factor: Float) -> RenderingData {
        match self {
            Renderable::Sphere(sphere) => sphere.make_rendering_data(height_exaggeration_factor),
            Renderable::Torus(torus) => torus.make_rendering_data(height_exaggeration_factor),
        }
    }

    pub fn rotational_phase_rad(&self) -> Float {
        match self {
            Renderable::Sphere(sphere) => sphere.rotational_phase_rad,
            Renderable::Torus(torus) => torus.rotational_phase_rad,
        }
    }

    pub fn t_nondimen(&self) -> Float {
        match self {
            Renderable::Sphere(sphere) => sphere.t,
            Renderable::Torus(torus) => torus.t,
        }
    }
}

#[derive(Clone)]
pub struct SphereRenderable {
    pub base_height: Float,

    pub t: Float,
    pub rotational_phase_rad: Float,

    pub theta_grid: nd::Array1<Float>,
    pub phi_grid: nd::Array1<Float>,

    pub height_array: nd::Array2<Float>,

    pub tracer_points_history_theta_phi: Vec<nd::Array2<Float>>,
    pub tracer_heights_history: Vec<nd::Array1<Float>>,
}

impl SphereRenderable {
    pub fn make_rendering_data(&self, height_exaggeration_factor: Float) -> RenderingData {
        let num_theta = self.theta_grid.len();
        let num_phi = self.phi_grid.len();

        let make_index = |i, j| (i * num_phi + (j % num_phi)) as u32;

        let quadrature_points = self.make_quadrature_points(height_exaggeration_factor);

        let mut augmented_points = quadrature_points.clone();
        // Add top point.
        let top = augmented_points.len() as u32;
        augmented_points.push(self.make_point(
            0.,
            0.,
            self.height_array.slice(nd::s![0, ..]).iter().sum::<Float>() / num_phi as Float,
            height_exaggeration_factor,
        ));
        // Add bottom point.
        let bottom = augmented_points.len() as u32;
        augmented_points.push(
            self.make_point(
                float_consts::PI,
                0.,
                self.height_array
                    .slice(nd::s![-1, ..])
                    .iter()
                    .sum::<Float>()
                    / num_phi as Float,
                height_exaggeration_factor,
            ),
        );

        let mut indices = Vec::new();
        for i in 0..num_theta - 1 {
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
            // Add top cap.
            let ul = make_index(0, j);
            let ur = make_index(0, j + 1);
            indices.push(ul);
            indices.push(ur);
            indices.push(top);

            // Add bottom cap.
            let ll = make_index(num_theta - 1, j);
            let lr = make_index(num_theta - 1, j + 1);
            indices.push(lr);
            indices.push(ll);
            indices.push(bottom);
        }

        let mut mesh = CpuMesh {
            indices: Indices::U32(indices),
            positions: Positions::F64(augmented_points),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();

        RenderingData {
            quadrature_points,
            tracer_points: self.make_tracer_points(height_exaggeration_factor),
            mesh,
        }
    }

    fn make_quadrature_points(&self, height_exaggeration_factor: Float) -> Vec<Vector3<Float>> {
        let mut points = Vec::with_capacity(self.theta_grid.len() * self.phi_grid.len());
        for (i, &theta) in self.theta_grid.iter().enumerate() {
            for (j, &phi) in self.phi_grid.iter().enumerate() {
                let height = self.height_array[[i, j]];
                points.push(self.make_point(theta, phi, height, height_exaggeration_factor));
            }
        }
        points
    }

    fn make_tracer_points(&self, height_exaggeration_factor: Float) -> TracerPoints {
        let (points, positions) = self
            .tracer_points_history_theta_phi
            .iter()
            .enumerate()
            .zip(&self.tracer_heights_history)
            .map(|((position, tracer_points_theta_phi), tracer_heights)| {
                tracer_points_theta_phi
                    .axis_iter(nd::Axis(1))
                    .zip(tracer_heights.iter())
                    .map(move |(point_theta_phi, &height)| {
                        (
                            self.make_point(
                                point_theta_phi[[flow::bases::ylm::Component::Theta as usize]],
                                point_theta_phi[[flow::bases::ylm::Component::Phi as usize]],
                                height,
                                height_exaggeration_factor,
                            ),
                            position,
                        )
                    })
            })
            .flatten()
            .unzip();
        TracerPoints { points, positions }
    }

    fn make_point(
        &self,
        theta: Float,
        phi: Float,
        height: Float,
        height_exaggeration_factor: Float,
    ) -> three_d::Vector3<Float> {
        let (sin_theta, cos_theta) = theta.sin_cos();
        let radially_out = Vector3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
        radially_out * (1. + (height - self.base_height) * height_exaggeration_factor)
    }
}

#[derive(Clone)]
pub struct TorusRenderable {
    pub base_height: Float,

    pub t: Float,
    pub rotational_phase_rad: Float,

    pub theta_grid: nd::Array1<Float>,
    pub phi_grid: nd::Array1<Float>,

    pub major_radius: Float,
    pub minor_radius: Float,

    pub height_array: nd::Array2<Float>,

    pub tracer_points_history_theta_phi: Vec<nd::Array2<Float>>,
    pub tracer_heights_history: Vec<nd::Array1<Float>>,
}

impl TorusRenderable {
    pub fn make_rendering_data(&self, height_exaggeration_factor: Float) -> RenderingData {
        let num_theta = self.theta_grid.len();
        let num_phi = self.phi_grid.len();
        let num_cell_vertices = num_theta * num_phi;

        let make_index = |i, j| (i % num_theta) * num_phi + (j % num_phi);

        let quadrature_points = self.make_quadrature_points(height_exaggeration_factor);
        let mut augmented_points = quadrature_points.clone();
        let height_flat = self.height_array.as_slice().unwrap();

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
            positions: Positions::F64(augmented_points),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();

        RenderingData {
            quadrature_points,
            tracer_points: self.make_tracer_points(height_exaggeration_factor),
            mesh,
        }
    }

    fn make_quadrature_points(&self, height_exaggeration_factor: Float) -> Vec<Vector3<Float>> {
        let mut points = Vec::with_capacity(self.theta_grid.len() * self.phi_grid.len());
        for (i, &theta) in self.theta_grid.iter().enumerate() {
            for (j, &phi) in self.phi_grid.iter().enumerate() {
                let height = self.height_array[[i, j]];
                points.push(self.make_point(theta, phi, height, height_exaggeration_factor));
            }
        }
        points
    }

    fn make_tracer_points(&self, height_exaggeration_factor: Float) -> TracerPoints {
        let (points, positions) = self
            .tracer_points_history_theta_phi
            .iter()
            .enumerate()
            .zip(&self.tracer_heights_history)
            .map(|((position, tracer_points_theta_phi), tracer_heights)| {
                tracer_points_theta_phi
                    .axis_iter(nd::Axis(1))
                    .zip(tracer_heights.iter())
                    .map(move |(point_theta_phi, &height)| {
                        (
                            self.make_point(
                                point_theta_phi[[0]],
                                point_theta_phi[[1]],
                                height,
                                height_exaggeration_factor,
                            ),
                            position,
                        )
                    })
            })
            .flatten()
            .unzip();
        TracerPoints { points, positions }
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

#[derive(Clone)]
pub struct TracerPoints {
    pub points: Vec<Vector3<Float>>,
    pub positions: Vec<usize>,
}
