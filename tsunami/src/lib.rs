#![cfg_attr(feature = "wasm-bindgen-rayon", feature(stdarch_wasm_atomic_wait))]

use flow::Float;
use three_d::*;
use wasm_bindgen::prelude::*;

mod geom;
mod render;

#[cfg(feature = "wasm-bindgen-rayon")]
pub use wasm_bindgen_rayon::init_thread_pool;

const M_PER_MI: Float = 1_609.34;
const GRAV_ACCEL_M_PER_S2: Float = 9.8;
const EARTH_RADIUS_MI: Float = 3_958.8;
const EARTH_RADIUS_M: Float = EARTH_RADIUS_MI * M_PER_MI;
const OCEAN_DEPTH_M: Float = 3_682.;
const WATER_KINEMATIC_VISCOSITY_M2_PER_S: Float = 1e-6;
const MOON_MASS_NONDIMEN: Float = 7.342e22 / 5.972_168e24;
const LUNAR_DISTANCE_NONDIMEN: Float = 384_399e3 / EARTH_RADIUS_M;

const TITLE: &'static str = "Ocean Playground";

fn time_scale_s() -> Float {
    EARTH_RADIUS_M / (GRAV_ACCEL_M_PER_S2 * OCEAN_DEPTH_M).sqrt()
}

fn time_scale_hr() -> Float {
    time_scale_s() / 60. / 60.
}

fn water_kinematic_viscosity_nondimen() -> Float {
    WATER_KINEMATIC_VISCOSITY_M2_PER_S / EARTH_RADIUS_M.powi(2) * time_scale_s()
}

#[derive(Clone, Debug, derivative::Derivative)]
#[derivative(PartialEq)]
struct Parameters {
    pub geometry_type: geom::GeometryType,

    pub log10_kinematic_viscosity_rel_to_water: Float,
    pub rotation_period_hr: Float,
    pub lunar_distance_rel_to_actual: Float,
    pub earthquake_region_size_mi: Float,
    pub earthquake_height_m: Float,

    pub substeps_per_physics_step: usize,

    #[derivative(PartialEq = "ignore")]
    pub resolution_level: u32,

    #[derivative(PartialEq = "ignore")]
    pub height_exaggeration_factor: Float,
    #[derivative(PartialEq = "ignore")]
    pub velocity_exaggeration_factor: Float,
    #[derivative(PartialEq = "ignore")]
    pub show_points: ShowPoints,

    #[derivative(PartialEq = "ignore")]
    pub earthquake_position: Option<Vec3>,
    // Response set by the physics thread after an earthquake trigger has been read and handled.
    #[derivative(PartialEq = "ignore")]
    pub earthquake_triggered: bool,
}

#[derive(Clone)]
struct ParametersMessage {
    pub params: Parameters,
    pub geometry_version: usize,
}

impl Parameters {
    fn tides() -> Self {
        Self {
            geometry_type: geom::GeometryType::Sphere,
            resolution_level: 6,
            log10_kinematic_viscosity_rel_to_water: 0.,
            rotation_period_hr: 24.,
            lunar_distance_rel_to_actual: 1.,
            earthquake_region_size_mi: 300.,
            earthquake_height_m: -4.,
            substeps_per_physics_step: 1,
            height_exaggeration_factor: 500.,
            velocity_exaggeration_factor: 3e3,
            show_points: ShowPoints::Tracer,
            earthquake_position: None,
            earthquake_triggered: false,
        }
    }

    fn coriolis() -> Self {
        Self {
            lunar_distance_rel_to_actual: 10.,
            rotation_period_hr: 5.,
            earthquake_region_size_mi: 500.,
            earthquake_height_m: -3.5,
            earthquake_position: Some(vec3(1., 0., 1.)),
            ..Self::tides()
        }
    }

    fn torus() -> Self {
        Self {
            geometry_type: geom::GeometryType::Torus,
            velocity_exaggeration_factor: 100.,
            ..Self::tides()
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum ShowPoints {
    Quadrature,
    Tracer,
    None,
}

fn physics_loop(
    mut geometry: geom::Geometry,
    shared_params_message: std::sync::Arc<std::sync::RwLock<ParametersMessage>>,
    renderable_sender: std::sync::mpsc::SyncSender<render::Renderable>,
) {
    // Initialize.
    log::info!("Initializing physics thread");
    let mut params_message = { shared_params_message.read().unwrap().clone() };

    // Physics loop.
    loop {
        // Send along renderables to main loop.
        {
            for renderable in geometry
                .make_renderables(params_message.params.substeps_per_physics_step)
                .into_iter()
            {
                renderable_sender.send(renderable).unwrap();
            }
        }

        // Check for updated parameters.
        {
            let new_params_message = shared_params_message.read().unwrap();
            let new_params = &new_params_message.params;
            if params_message.geometry_version != new_params_message.geometry_version {
                geometry =
                    geom::Geometry::new(new_params.geometry_type, new_params.resolution_level);
            }
            params_message = new_params_message.clone();

            geometry.set_kinematic_viscosity(
                water_kinematic_viscosity_nondimen()
                    * (10 as Float)
                        .powf(params_message.params.log10_kinematic_viscosity_rel_to_water),
            );
            geometry.set_rotation_angular_speed(
                flow::float_consts::TAU / params_message.params.rotation_period_hr
                    * time_scale_hr(),
            );
            geometry.set_lunar_distance(
                params_message.params.lunar_distance_rel_to_actual * LUNAR_DISTANCE_NONDIMEN,
            );
            geometry.set_velocity_exaggeration_factor(
                params_message.params.velocity_exaggeration_factor,
            );
        }

        // Update parameters if needed.
        if let Some(earthquake_position) = params_message.params.earthquake_position {
            let mut shared_params_writer = shared_params_message.write().unwrap();
            shared_params_writer.params.earthquake_position = None;
            if !params_message.params.earthquake_triggered {
                geometry.trigger_earthquake(
                    earthquake_position,
                    params_message.params.earthquake_region_size_mi / EARTH_RADIUS_MI,
                    params_message.params.earthquake_height_m / OCEAN_DEPTH_M,
                );
                shared_params_writer.params.earthquake_triggered = true;
            }
        }

        // Take physics timestep.
        geometry.integrate();
    }
}

#[wasm_bindgen]
pub async fn run() {
    console_log::init().unwrap();

    log::info!(
        "Spawning threads, {} available in pool",
        rayon::current_num_threads()
    );

    let mut params = Parameters::tides();
    let mut geometry_version = 0;
    let geometry = geom::Geometry::new(params.geometry_type, params.resolution_level);
    let mut renderable = geometry.make_renderables(1).pop().unwrap();
    let (renderable_sender, renderable_reader) = std::sync::mpsc::sync_channel(50);
    let shared_params_message_write = std::sync::Arc::new(std::sync::RwLock::new(
        ParametersMessage {
            params: params.clone(),
            geometry_version,
        }
        .clone(),
    ));
    let shared_params_read = shared_params_message_write.clone();

    log::info!("Initializing rendering");

    let window = Window::new(WindowSettings {
        title: TITLE.to_string(),
        max_size: None,
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(1., 1., 1.) * 2.,
        vec3(0., 0., 0.),
        vec3(0.0, 0.0, 1.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(*camera.target(), 5. / 3., 5.);

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
    let mut commonmark_cache = egui_commonmark::CommonMarkCache::default();

    macro_rules! load_skybox {
        ($($path:literal),+) => {
            {
                let asset_bytes = [$(include_bytes!($path).to_vec()),+];
                let mut raw_assets = three_d_asset::io::RawAssets::new();
                for (path, bytes) in [$($path),+].iter().zip(asset_bytes.into_iter()) {
                    raw_assets.insert(path, bytes);
                }
                Skybox::new(
                    &context,
                    $(
                        &raw_assets.deserialize($path).unwrap()
                    ),+
                )
            }
        };
    }
    let skybox = load_skybox!(
        "../assets/right.jpg",
        "../assets/left.jpg",
        "../assets/top.jpg",
        "../assets/bottom.jpg",
        "../assets/front.jpg",
        "../assets/back.jpg"
    );

    let ambient =
        AmbientLight::new_with_environment(&context, 20.0, Srgba::WHITE, skybox.texture());
    let directional = DirectionalLight::new(&context, 5.0, Srgba::WHITE, &vec3(-1.0, -1.0, -1.0));

    let mut wall_time_per_render_sec = 0.;
    let mut sim_time_per_renderable = 0.;

    let rendering_data = renderable.make_rendering_data(params.height_exaggeration_factor);

    let mut mesh_model = Gm::new(
        Mesh::new(&context, &rendering_data.mesh.into()),
        PhysicalMaterial::new(
            &context,
            &CpuMaterial {
                roughness: 0.2,
                metallic: 0.9,
                lighting_model: LightingModel::Cook(
                    NormalDistributionFunction::TrowbridgeReitzGGX,
                    GeometryFunction::SmithSchlickGGX,
                ),
                albedo: Srgba::new(0, 102, 204, u8::MAX),
                ..Default::default()
            },
        ),
    );

    let mut point_mesh = CpuMesh::sphere(10);
    point_mesh.transform(&Mat4::from_scale(0.002)).unwrap();
    let mut quadrature_point_cloud_model = Gm {
        geometry: InstancedMesh::new(
            &context,
            &PointCloud {
                positions: Positions::F64(rendering_data.quadrature_points),
                colors: None,
            }
            .into(),
            &point_mesh,
        ),
        material: ColorMaterial::default(),
    };
    let mut tracer_point_cloud_model = Gm {
        geometry: InstancedMesh::new(
            &context,
            &PointCloud {
                positions: Positions::F64(rendering_data.tracer_points),
                colors: None,
            }
            .into(),
            &point_mesh,
        ),
        material: ColorMaterial::default(),
    };

    let mut tsunami_hint_circle_object = Gm {
        geometry: Mesh::new(&context, &point_mesh),
        material: ColorMaterial::new_transparent(
            &context,
            &CpuMaterial {
                roughness: 1.,
                albedo: Srgba::new(u8::MAX, u8::MAX, u8::MAX, u8::MAX / 3),
                transmission: 0.5,
                ..Default::default()
            },
        ),
    };

    let mut wall_time_of_last_renderable = web_time::Instant::now();
    let mut wall_time_per_renderable_sec = 0.5;
    let mut sim_time_of_last_renderable = 0.;

    let mut double_click_detector = DoubleClickDetector::default();

    // Remove loading screen.
    web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("loading")
        .unwrap()
        .remove();

    rayon::spawn(move || physics_loop(geometry, shared_params_read, renderable_sender));
    window.render_loop(move |mut frame_input| {
        if let Ok(mut shared_params_message) = shared_params_message_write.try_write() {
            if shared_params_message.params.earthquake_triggered && !params.earthquake_triggered {
                params.earthquake_triggered = false;
                params.earthquake_position = None;
            }
            *shared_params_message = ParametersMessage {
                params: params.clone(),
                geometry_version,
            };
        }

        if let Ok(new_renderable) = renderable_reader.try_recv() {
            renderable = new_renderable;
            if renderable.t_nondimen() < sim_time_of_last_renderable {
                sim_time_of_last_renderable = 0.;
            }
            let now = web_time::Instant::now();
            wall_time_per_renderable_sec = 0.98 * wall_time_per_renderable_sec
                + 0.02
                    * now
                        .duration_since(wall_time_of_last_renderable)
                        .as_secs_f64();
            sim_time_per_renderable = 0.9 * sim_time_per_renderable
                + 0.1 * (renderable.t_nondimen() - sim_time_of_last_renderable);

            wall_time_of_last_renderable = now;
            sim_time_of_last_renderable = renderable.t_nondimen();
        }
        let rendering_data = renderable.make_rendering_data(params.height_exaggeration_factor);

        {
            mesh_model.geometry = Mesh::new(&context, &rendering_data.mesh.into());

            if let ShowPoints::Quadrature = params.show_points {
                quadrature_point_cloud_model.geometry = InstancedMesh::new(
                    &context,
                    &PointCloud {
                        positions: Positions::F64(rendering_data.quadrature_points),
                        colors: None,
                    }
                    .into(),
                    &point_mesh,
                );
                quadrature_point_cloud_model.material.render_states.cull = Cull::None;
            } else {
                quadrature_point_cloud_model.material.render_states.cull = Cull::FrontAndBack;
            }

            if let ShowPoints::Tracer = params.show_points {
                tracer_point_cloud_model.geometry = InstancedMesh::new(
                    &context,
                    &PointCloud {
                        positions: Positions::F64(rendering_data.tracer_points),
                        colors: None,
                    }
                    .into(),
                    &point_mesh,
                );
                tracer_point_cloud_model.material.render_states.cull = Cull::None;
            } else {
                tracer_point_cloud_model.material.render_states.cull = Cull::FrontAndBack;
            }
        }

        for event in frame_input.events.iter() {
            match event {
                &Event::MousePress {
                    button: control::MouseButton::Left,
                    ..
                } => {
                    double_click_detector.process_press();
                }
                &Event::MouseRelease {
                    button: control::MouseButton::Left,
                    position: screen_position,
                    ..
                } => {
                    if let DoubleClickResult::Detected = double_click_detector.process_release() {
                        if let Some(space_position) =
                            pick(&context, &camera, screen_position, &mesh_model)
                        {
                            params.earthquake_position = Some(space_position);
                            params.earthquake_triggered = false;
                        }
                    }
                }
                &Event::MouseMotion {
                    position: screen_position,
                    ..
                } => {
                    if let (Some(space_position), geom::GeometryType::Sphere) = (
                        pick(&context, &camera, screen_position, &mesh_model),
                        params.geometry_type,
                    ) {
                        let mut tsunami_hint_circle = CpuMesh::circle(20);
                        let axis = space_position.cross(Vector3::unit_z()).normalize();
                        let angle = space_position.angle(Vector3::unit_z());
                        tsunami_hint_circle
                            .transform(&Matrix4::from_scale(
                                (params.earthquake_region_size_mi / EARTH_RADIUS_MI) as f32,
                            ))
                            .unwrap();
                        tsunami_hint_circle
                            .transform(&Matrix4::from_translation(
                                Vector3::unit_z().normalize_to(space_position.magnitude() * 1.01),
                            ))
                            .unwrap();
                        tsunami_hint_circle
                            .transform(&Matrix4::from_axis_angle(axis, -angle))
                            .unwrap();
                        tsunami_hint_circle_object.geometry =
                            Mesh::new(&context, &tsunami_hint_circle);
                        tsunami_hint_circle_object.material.render_states.cull = Cull::None;
                    } else {
                        tsunami_hint_circle_object.material.render_states.cull = Cull::FrontAndBack;
                    }
                }
                _ => {}
            }
        }

        // Set hint circle color to red if completing a double click.
        double_click_detector.process_idle();
        match double_click_detector.state {
            DoubleClickDetectorState::Idle
            | DoubleClickDetectorState::FirstPress(_)
            | DoubleClickDetectorState::FirstRelease(_) => {
                tsunami_hint_circle_object.material.color.g = u8::MAX;
                tsunami_hint_circle_object.material.color.b = u8::MAX;
            }
            DoubleClickDetectorState::SecondPress => {
                tsunami_hint_circle_object.material.color.g = 0;
                tsunami_hint_circle_object.material.color.b = 0;
            }
        }

        let mut geom_change = false;

        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                egui::Window::new(TITLE)
                    .vscroll(true)
                    .default_height(650.)
                    .show(gui_context, |ui| {
                        egui::CollapsingHeader::new(egui::RichText::from("Info").heading())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add_space(-10.);
                                egui_commonmark::CommonMarkViewer::new("info").show(
                                    ui,
                                    &mut commonmark_cache,
                                    INFO_MARKDOWN,
                                );
                                ui.add_space(-10.);
                            });

                        ui.collapsing(egui::RichText::from("Details").heading(), |ui| {
                            ui.add_space(-10.);
                            egui_commonmark::CommonMarkViewer::new("details").show(
                                ui,
                                &mut commonmark_cache,
                                DETAILS_MARKDOWN,
                            );
                            ui.add_space(-10.);
                        });

                        egui::CollapsingHeader::new(egui::RichText::from("Controls").heading())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("Presets").strong());
                                ui.horizontal(|ui| {
                                    for (selection_params, name) in [
                                        (Parameters::tides(), "Tides"),
                                        (Parameters::coriolis(), "Coriolis force"),
                                        (Parameters::torus(), "Torus world"),
                                    ] {
                                        let params_same = params == selection_params;
                                        if ui.selectable_label(params_same, name).clicked()
                                            && !params_same
                                        {
                                            params = selection_params;
                                            geometry_version += 1;
                                        }
                                    }
                                });
                                ui.add_space(10.);
                                ui.label(egui::RichText::new("Physics").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.log10_kinematic_viscosity_rel_to_water,
                                        (0.)..=(13.),
                                    )
                                    .text("viscosity")
                                    .prefix("10^")
                                    .suffix(" × water"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.rotation_period_hr,
                                        (5.)..=(100.),
                                    )
                                    .text("rotation period")
                                    .suffix(" hr"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.lunar_distance_rel_to_actual,
                                        (0.5)..=(10.),
                                    )
                                    .logarithmic(true)
                                    .text("lunar distance")
                                    .suffix(" × actual"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.earthquake_region_size_mi,
                                        1e2..=5e2,
                                    )
                                    .text("earthquake region size")
                                    .suffix(" mi"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut params.earthquake_height_m, -6e0..=6e0)
                                        .text("earthquake height")
                                        .suffix(" m"),
                                );
                                ui.horizontal(|ui| {
                                    geom_change |= ui.button("Restart").clicked();
                                    ui.label(format!(
                                        "{:.1} hr elapsed sim time",
                                        renderable.t_nondimen() * time_scale_hr()
                                    ));
                                });
                                ui.add_space(10.);

                                ui.label(egui::RichText::new("Visualization").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.height_exaggeration_factor,
                                        1e0..=1e4,
                                    )
                                    .logarithmic(true)
                                    .text("height exaggeration")
                                    .suffix("×"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.velocity_exaggeration_factor,
                                        1e0..=1e5,
                                    )
                                    .logarithmic(true)
                                    .text("velocity exaggeration")
                                    .suffix("×"),
                                );
                                ui.horizontal(|ui| {
                                    ui.radio_value(
                                        &mut params.show_points,
                                        ShowPoints::Quadrature,
                                        "quadrature",
                                    );
                                    ui.radio_value(
                                        &mut params.show_points,
                                        ShowPoints::Tracer,
                                        "tracer",
                                    );
                                    ui.radio_value(
                                        &mut params.show_points,
                                        ShowPoints::None,
                                        "none",
                                    );
                                    ui.label("show points");
                                });
                                ui.add_space(10.);

                                ui.label(egui::RichText::new("Performance").strong());
                                ui.horizontal(|ui| {
                                    for n in 5..=8 {
                                        geom_change |= ui
                                            .radio_value(
                                                &mut params.resolution_level,
                                                n,
                                                format!("{}", 2i32.pow(n)),
                                            )
                                            .changed();
                                    }
                                    ui.label("resolution");
                                });
                                // ui.add(
                                //     egui::Slider::new(
                                //         &mut params.substeps_per_physics_step,
                                //         1..=30,
                                //     )
                                //     .text("substeps per physics update"),
                                // );
                                ui.label(format!(
                                    "({:.0}, {:.0}) wall ms/(step, render)",
                                    wall_time_per_renderable_sec * 1000.,
                                    wall_time_per_render_sec * 1000.,
                                ));
                                ui.label(format!(
                                    "({:.0}, {:.1}) sim s/(step, wall ms)",
                                    sim_time_per_renderable * time_scale_s(),
                                    sim_time_per_renderable * time_scale_s()
                                        / (wall_time_per_renderable_sec * 1000.),
                                ));
                                ui.add_space(10.);

                                ui.label(egui::RichText::new("Geometry").strong());
                                ui.horizontal(|ui| {
                                    geom_change |= ui
                                        .radio_value(
                                            &mut params.geometry_type,
                                            geom::GeometryType::Sphere,
                                            "Sphere",
                                        )
                                        .changed();
                                    geom_change |= ui
                                        .radio_value(
                                            &mut params.geometry_type,
                                            geom::GeometryType::Torus,
                                            "Torus",
                                        )
                                        .changed();
                                });
                                ui.add_space(10.);
                            });
                    });

                gui_context.output(|output| {
                    if let Some(url) = &output.open_url {
                        web_sys::window()
                            .unwrap()
                            .open_with_url_and_target(&url.url, "_blank")
                            .unwrap();
                    }
                });
            },
        );

        if geom_change {
            geometry_version += 1;
        }

        control.handle_events(&mut camera, &mut frame_input.events);
        frame_input
            .screen()
            .render(
                &camera,
                std::iter::empty()
                    .chain(&mesh_model)
                    .chain(&quadrature_point_cloud_model)
                    .chain(&tracer_point_cloud_model)
                    .chain(&tsunami_hint_circle_object)
                    .chain(&skybox),
                &[&ambient, &directional],
            )
            .write(|| gui.render())
            .unwrap();

        // Exponential moving average.
        wall_time_per_render_sec =
            0.95 * wall_time_per_render_sec + 0.05 * frame_input.elapsed_time as Float / 1000.;

        FrameOutput::default()
    });
}

#[derive(Clone, Copy, Default)]
enum DoubleClickDetectorState {
    #[default]
    Idle,
    FirstPress(web_time::Instant),
    FirstRelease(web_time::Instant),
    SecondPress,
}

enum DoubleClickResult {
    Detected,
    NotDetected,
}

#[derive(Default)]
struct DoubleClickDetector {
    state: DoubleClickDetectorState,
}

impl DoubleClickDetector {
    const MAX_WAIT_TIME_MS: u128 = 300;

    pub fn process_idle(&mut self) {
        let now = web_time::Instant::now();
        self.state = match self.state {
            DoubleClickDetectorState::Idle => DoubleClickDetectorState::Idle,
            DoubleClickDetectorState::FirstPress(prev_time)
            | DoubleClickDetectorState::FirstRelease(prev_time) => {
                if Self::within_wait_time(prev_time, now) {
                    self.state
                } else {
                    DoubleClickDetectorState::Idle
                }
            }
            DoubleClickDetectorState::SecondPress => self.state,
        };
    }

    pub fn process_press(&mut self) {
        let now = web_time::Instant::now();
        self.state = match self.state {
            DoubleClickDetectorState::Idle => DoubleClickDetectorState::FirstPress(now),
            DoubleClickDetectorState::FirstPress(_) => DoubleClickDetectorState::Idle,
            DoubleClickDetectorState::FirstRelease(prev_time) => {
                if Self::within_wait_time(prev_time, now) {
                    DoubleClickDetectorState::SecondPress
                } else {
                    DoubleClickDetectorState::Idle
                }
            }
            DoubleClickDetectorState::SecondPress => DoubleClickDetectorState::Idle,
        };
    }

    pub fn process_release(&mut self) -> DoubleClickResult {
        let mut result = DoubleClickResult::NotDetected;
        let now = web_time::Instant::now();
        self.state = match self.state {
            DoubleClickDetectorState::Idle => DoubleClickDetectorState::Idle,
            DoubleClickDetectorState::FirstPress(prev_time) => {
                if Self::within_wait_time(prev_time, now) {
                    DoubleClickDetectorState::FirstRelease(now)
                } else {
                    DoubleClickDetectorState::Idle
                }
            }
            DoubleClickDetectorState::FirstRelease(_) => DoubleClickDetectorState::Idle,
            DoubleClickDetectorState::SecondPress => {
                result = DoubleClickResult::Detected;
                DoubleClickDetectorState::Idle
            }
        };
        result
    }

    fn within_wait_time(prev_time: web_time::Instant, now: web_time::Instant) -> bool {
        now.checked_duration_since(prev_time)
            .unwrap_or(web_time::Duration::ZERO)
            .as_millis()
            <= Self::MAX_WAIT_TIME_MS
    }
}

const INFO_MARKDOWN: &'static str = indoc::indoc! {"
    Interactive ocean simulations in a web browser!
    
    * **Double click** to set off an earthquake/tsunami
    * **Dotted lines** are “tracers” indicating local water motion
    * **Play around** with different controls below

    Alpha version — please **do not share yet**! Realistic terrain (continents/sea floor/etc.)
    coming soon...
"};

const DETAILS_MARKDOWN: &'static str = indoc::indoc! {"
    Solves the [shallow water equations](https://en.wikipedia.org/wiki/Shallow_water_equations)
    [pseudospectrally](https://en.wikipedia.org/wiki/Pseudo-spectral_method) using a
    [spherical harmonic](https://en.wikipedia.org/wiki/Spherical_harmonics) basis. (Toroidal
    geometry with more limited functionality also included for fun, which uses a rectangular domain
    with periodic boundary conditions and a Fourier basis.)
    
    Effects included:
    
    * Viscosity (negligible for tsunamis but can be artifically increased)
    * Coriolis force
    * Lunar tides (taking the moon as stationary for simplicity)
    * Advection of entrained “tracer” particles
    * Realistic terrain (continents/sea floor) (planned)

    Tracer motion is accelerated by a large factor for visualization. Nonlinear momentum advection
    term not yet implemented in spherical geometry.
    
    Tech stack: Rust/WebAssembly/WebGL/[`egui`](https://www.egui.rs/)/
    [`three-d`](https://github.com/asny/three-d).

    By Josh Burkart. View and contribute to the code on
    [GitHub](https://github.com/joshburkart/tsunami)!
"};
