use three_d::*;
use wasm_bindgen::prelude::*;

use ndarray as nd;

use flow::float_consts;
use flow::Float;

#[derive(Clone)]
struct Parameters {
    pub kinematic_viscosity_rel_to_water: Float,
    pub height_exaggeration_factor: Float,
    pub realtime_ratio: Float,
    pub resolution_level: u32,
    pub show_point_cloud: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            kinematic_viscosity_rel_to_water: 1.,
            height_exaggeration_factor: 100.,
            realtime_ratio: 100.,
            resolution_level: 6,
            show_point_cloud: false,
        }
    }
}

#[wasm_bindgen(start)]
pub async fn run() {
    console_log::init().unwrap();

    let window = Window::new(WindowSettings {
        title: "Tsunami Simulator".to_string(),
        max_size: None,
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(7.0, 7.0, 7.0),
        vec3(0., 0., 0.),
        vec3(0.0, 0.0, 1.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(*camera.target(), 5., 15.);

    let mut params = Parameters::default();

    let mut torus = ToroidalGeometry::new(params.resolution_level);

    let mut gui = GUI::new(&context);
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "Roboto".to_owned(),
        egui::FontData::from_static(include_bytes!("../assets/Roboto-Light.ttf")),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "Roboto".to_owned());
    gui.context().set_fonts(fonts);

    let mut frame_time_history = egui::util::History::new(2..1000, 500.);

    let paths = [
        "right.png",
        "left.png",
        "top.png",
        "bottom.png",
        "front.png",
        "back.png",
    ];
    let mut loaded = three_d_asset::io::load_async(&paths).await.unwrap();
    let skybox = Skybox::new(
        &context,
        &loaded.deserialize(&paths[0]).unwrap(),
        &loaded.deserialize(&paths[1]).unwrap(),
        &loaded.deserialize(&paths[2]).unwrap(),
        &loaded.deserialize(&paths[3]).unwrap(),
        &loaded.deserialize(&paths[4]).unwrap(),
        &loaded.deserialize(&paths[5]).unwrap(),
    );

    let ambient =
        AmbientLight::new_with_environment(&context, 20.0, Srgba::WHITE, skybox.texture());
    let directional = DirectionalLight::new(&context, 5.0, Srgba::WHITE, &vec3(-1.0, -1.0, -1.0));

    let new_mesh = torus.make_mesh(params.height_exaggeration_factor);
    let mut mesh_model = Gm::new(
        Mesh::new(&context, &new_mesh.clone().into()),
        PhysicalMaterial::new(
            &context,
            &CpuMaterial {
                roughness: 0.2,
                metallic: 0.9,
                lighting_model: LightingModel::Cook(
                    NormalDistributionFunction::TrowbridgeReitzGGX,
                    GeometryFunction::SmithSchlickGGX,
                ),
                albedo: Srgba::new(0, 89, 179, u8::MAX),
                ..Default::default()
            },
        ),
    );

    let mut point_mesh = CpuMesh::sphere(10);
    point_mesh.transform(&Mat4::from_scale(0.008)).unwrap();
    let mut point_cloud_model = Gm {
        geometry: InstancedMesh::new(
            &context,
            &PointCloud {
                positions: new_mesh.positions,
                colors: None,
            }
            .into(),
            &point_mesh,
        ),
        material: ColorMaterial::default(),
    };

    let mut reset = false;

    window.render_loop(move |mut frame_input| {
        if reset {
            torus = ToroidalGeometry::new(params.resolution_level);
        }

        torus.solver.problem_mut().kinematic_viscosity =
            1e-3 * params.kinematic_viscosity_rel_to_water;
        torus.solver.integrate(3e-3 * params.realtime_ratio / 100.);

        {
            let new_mesh = torus.make_mesh(params.height_exaggeration_factor);
            mesh_model.geometry = Mesh::new(
                &context,
                &torus
                    .make_mesh(params.height_exaggeration_factor)
                    .clone()
                    .into(),
            );

            if params.show_point_cloud {
                point_cloud_model.geometry = InstancedMesh::new(
                    &context,
                    &PointCloud {
                        positions: new_mesh.positions,
                        colors: None,
                    }
                    .into(),
                    &point_mesh,
                );
                point_cloud_model.material.render_states.cull = Cull::None;
            } else {
                point_cloud_model.material.render_states.cull = Cull::FrontAndBack;
            }
        }

        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                use three_d::egui;
                egui::Window::new("Tsunami Simulator")
                    .vscroll(true)
                    .show(gui_context, |ui| {
                        egui::CollapsingHeader::new(egui::RichText::from("Settings").heading())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Physics");
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.kinematic_viscosity_rel_to_water,
                                        1e-1..=1e2,
                                    )
                                    .logarithmic(true)
                                    .text("kinematic viscosity")
                                    .suffix("× water"),
                                );
                                ui.add_space(15.);

                                ui.label("Visualization");
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.height_exaggeration_factor,
                                        1e0..=1e3,
                                    )
                                    .logarithmic(true)
                                    .text("height exaggeration")
                                    .suffix("×"),
                                );
                                ui.add(egui::Checkbox::new(
                                    &mut params.show_point_cloud,
                                    "show points",
                                ));
                                ui.add_space(15.);

                                ui.label("Performance");
                                reset = ui
                                    .add(
                                        egui::Slider::new(&mut params.resolution_level, 4..=7)
                                            .fixed_decimals(0)
                                            .step_by(1.)
                                            .prefix("2^")
                                            .text("num subdivisions"),
                                    )
                                    .changed();
                                ui.add(
                                    egui::Slider::new(&mut params.realtime_ratio, 1e0..=1e3)
                                        .logarithmic(true)
                                        .text("speed target")
                                        .suffix("× realtime"),
                                );
                                ui.label(format!(
                                    "{:.0} ms/frame",
                                    frame_time_history.average().unwrap_or(0.)
                                ));
                            });

                        ui.collapsing(egui::RichText::from("About").heading(), |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing.x = 0.0;
                                ui.label(
                                    "Solves the shallow water equations pseudospectrally. Torus \
                                    uses a rectangular domain with periodic boundary conditions. \
                                    Sphere uses spherical harmonics. Tech stack: Rust/WASM/WebGL/\
                                    egui/three-d. Written by Josh Burkart.",
                                );
                            });
                        });
                    });
            },
        );

        control.handle_events(&mut camera, &mut frame_input.events);
        frame_input
            .screen()
            .render(
                &camera,
                mesh_model
                    .into_iter()
                    .chain(&point_cloud_model)
                    .chain(&skybox),
                &[&ambient, &directional],
            )
            .write(|| gui.render())
            .unwrap();

        frame_time_history.add(
            frame_input.accumulated_time,
            frame_input.elapsed_time as f32,
        );

        FrameOutput::default()
    });
}

struct ToroidalGeometry {
    base_height: Float,

    major_radius: Float,
    minor_radius: Float,

    theta_grid: Vec<Float>,
    phi_grid: Vec<Float>,

    pub solver: flow::physics::Solver<flow::bases::RectangularPeriodicBasis>,
}

impl ToroidalGeometry {
    pub fn new(resolution_level: u32) -> Self {
        use flow::bases::Basis;

        let (base_height, solver) = Self::make_solver(resolution_level);

        let axes = solver.problem().basis.axes();

        let major_radius = axes[0][1] * axes[0].len() as Float / float_consts::TAU;
        let minor_radius = axes[1][1] * axes[1].len() as Float / float_consts::TAU;
        let theta_grid = (&axes[0] / major_radius).to_vec();
        let phi_grid = (&axes[1] / minor_radius).to_vec();

        Self {
            base_height,

            major_radius,
            minor_radius,

            theta_grid,
            phi_grid,

            solver,
        }
    }

    pub fn make_mesh(&self, height_exaggeration_factor: Float) -> CpuMesh {
        let num_theta = self.theta_grid.len();
        let num_phi = self.phi_grid.len();
        let num_cell_vertices = num_theta * num_phi;

        let make_index = |i, j| (i % num_theta) * num_phi + (j % num_phi);

        let height_array = self.solver.fields().height_grid();
        let mut points = self.make_points(&height_array, height_exaggeration_factor);
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
                indices.push(ll as u16);
                indices.push(lr as u16);
                indices.push(c as u16);
                // Left triangle.
                indices.push(ll as u16);
                indices.push(c as u16);
                indices.push(ul as u16);
                // Top triangle.
                indices.push(ul as u16);
                indices.push(c as u16);
                indices.push(ur as u16);
                // Right triangle.
                indices.push(lr as u16);
                indices.push(ur as u16);
                indices.push(c as u16);
            }
        }

        let mut mesh = CpuMesh {
            indices: Indices::U16(indices),
            positions: Positions::F32(points),
            ..Default::default()
        };
        mesh.compute_normals();
        mesh.validate().unwrap();
        mesh
    }

    fn make_points(
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
        let radially_out = Vector3::new(-theta.cos(), theta.sin(), 0.);
        self.major_radius * radially_out
            + (self.minor_radius
                + (height_exaggeration_factor / 100.) * (height - self.base_height))
                * (-phi.cos() * radially_out + phi.sin() * Vector3::unit_z())
    }

    fn make_solver(
        resolution_level: u32,
    ) -> (
        Float,
        flow::physics::Solver<flow::bases::RectangularPeriodicBasis>,
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
        let basis = std::sync::Arc::new(flow::bases::RectangularPeriodicBasis::new(
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
            rain_rate: None,
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
    fn test_toroidal() {
        let torus = ToroidalGeometry::new(5);
        torus.make_mesh(2.);
    }
}
