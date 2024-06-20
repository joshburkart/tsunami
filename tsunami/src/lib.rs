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

const TITLE: &'static str = "Tsunami Playground";

fn time_scale_s() -> Float {
    EARTH_RADIUS_M / (GRAV_ACCEL_M_PER_S2 * OCEAN_DEPTH_M).sqrt()
}

fn time_scale_hr() -> Float {
    time_scale_s() / 60. / 60.
}

fn water_kinematic_viscosity_nondimen() -> Float {
    WATER_KINEMATIC_VISCOSITY_M2_PER_S / EARTH_RADIUS_M.powi(2) * time_scale_s()
}

#[derive(Clone)]
struct Parameters {
    pub geometry_type: geom::GeometryType,
    pub resolution_level: u32,

    pub log10_kinematic_viscosity_rel_to_water: Float,
    pub rotation_period_hr: Float,
    pub earthquake_region_size_mi: Float,
    pub earthquake_height_m: Float,

    pub substeps_per_physics_step: usize,

    pub height_exaggeration_factor: Float,
    pub show_point_cloud: bool,

    pub earthquake_position: Option<Vec3>,
    // Response set by the physics thread after an earthquake trigger has been read and handled.
    pub earthquake_triggered: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            geometry_type: geom::GeometryType::Sphere,
            resolution_level: 6,
            log10_kinematic_viscosity_rel_to_water: 0.,
            rotation_period_hr: 24.,
            earthquake_region_size_mi: 300.,
            earthquake_height_m: -1.,
            substeps_per_physics_step: 3,
            height_exaggeration_factor: 1000.,
            show_point_cloud: false,
            earthquake_position: Some(three_d::Vec3::new(0.5, -0.5, 0.5)),
            earthquake_triggered: false,
        }
    }
}

fn physics_loop(
    mut geometry: geom::Geometry,
    shared_params: std::sync::Arc<std::sync::RwLock<Parameters>>,
    renderable_sender: std::sync::mpsc::SyncSender<render::Renderable>,
) {
    // Initialize.
    log::info!("Initializing physics thread");
    let mut params = { shared_params.read().unwrap().clone() };

    // Physics loop.
    loop {
        // Send along renderables to main loop.
        {
            for renderable in geometry
                .make_renderables(params.substeps_per_physics_step)
                .into_iter()
            {
                renderable_sender.send(renderable).unwrap();
            }
        }

        // Check for updated parameters.
        {
            let new_params = shared_params.read().unwrap();
            if params.geometry_type != new_params.geometry_type
                || params.resolution_level != new_params.resolution_level
            {
                geometry =
                    geom::Geometry::new(new_params.geometry_type, new_params.resolution_level);
            }
            params = new_params.clone();

            geometry.set_kinematic_viscosity(
                water_kinematic_viscosity_nondimen()
                    * (10 as Float).powf(params.log10_kinematic_viscosity_rel_to_water),
            );
            geometry.set_rotation_angular_speed(
                flow::float_consts::TAU / params.rotation_period_hr * time_scale_hr(),
            );
        }

        // Update parameters if needed.
        if let Some(earthquake_position) = params.earthquake_position {
            let mut shared_params_writer = shared_params.write().unwrap();
            shared_params_writer.earthquake_position = None;
            if !params.earthquake_triggered {
                geometry.trigger_earthquake(
                    earthquake_position,
                    params.earthquake_region_size_mi / EARTH_RADIUS_MI,
                    params.earthquake_height_m / OCEAN_DEPTH_M,
                );
                shared_params_writer.earthquake_triggered = true;
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

    let mut params = Parameters::default();
    let geometry = geom::Geometry::new(params.geometry_type, params.resolution_level);
    let mut renderable = geometry.make_renderables(1).pop().unwrap();
    let (renderable_sender, renderable_reader) = std::sync::mpsc::sync_channel(50);
    let shared_params_write = std::sync::Arc::new(std::sync::RwLock::new(params.clone()));
    let shared_params_read = shared_params_write.clone();

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
        vec3(7.0, 7.0, 7.0) / 3.,
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

    let rendering = renderable.make_rendering_data(params.height_exaggeration_factor);

    let mut mesh_model = Gm::new(
        Mesh::new(&context, &rendering.mesh.into()),
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
    let mut point_cloud_model = Gm {
        geometry: InstancedMesh::new(
            &context,
            &PointCloud {
                positions: Positions::F64(rendering.points),
                colors: None,
            }
            .into(),
            &point_mesh,
        ),
        material: ColorMaterial::default(),
    };

    let mut wall_time_of_last_renderable = web_time::Instant::now();
    let mut wall_time_per_renderable_sec = 0.5;
    let mut sim_time_of_last_renderable = 0.;

    rayon::spawn(move || physics_loop(geometry, shared_params_read, renderable_sender));
    window.render_loop(move |mut frame_input| {
        if let Ok(mut shared_params) = shared_params_write.try_write() {
            if shared_params.earthquake_triggered && !params.earthquake_triggered {
                params.earthquake_triggered = false;
                params.earthquake_position = None;
            }
            *shared_params = params.clone();
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
                        .as_secs_f32();
            sim_time_per_renderable = 0.9 * sim_time_per_renderable
                + 0.1 * (renderable.t_nondimen() - sim_time_of_last_renderable);

            wall_time_of_last_renderable = now;
            sim_time_of_last_renderable = renderable.t_nondimen();
        }
        let rendering_data = renderable.make_rendering_data(params.height_exaggeration_factor);

        {
            mesh_model.geometry = Mesh::new(&context, &rendering_data.mesh.into());

            if params.show_point_cloud {
                point_cloud_model.geometry = InstancedMesh::new(
                    &context,
                    &PointCloud {
                        positions: Positions::F64(rendering_data.points),
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

        for event in frame_input.events.iter() {
            if let Event::MousePress {
                button: control::MouseButton::Right,
                position: screen_position,
                ..
            } = event
            {
                if let Some(space_position) = pick(&context, &camera, *screen_position, &mesh_model)
                {
                    params.earthquake_position = Some(space_position);
                    params.earthquake_triggered = false;
                    break;
                }
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
                    .show(gui_context, |ui| {
                        egui::CollapsingHeader::new(egui::RichText::from("Info").heading())
                            .default_open(true)
                            .show(ui, |ui| {
                                let markdown = indoc::indoc! {"
                                    Alpha version -- please **do not share yet**! Many rough
                                    edges, e.g. advection term not yet implemented in spherical
                                    geometry.
                                        
                                    **Right click** to set off a tsunami! Increase \"substeps per
                                    physics update\" below if the simulation is stuttering.
                                    
                                    Planned features: realistic terrain (continents/sea floor/
                                        etc.), tides, and more...
                                "};
                                ui.add_space(-10.);
                                egui_commonmark::CommonMarkViewer::new("info").show(
                                    ui,
                                    &mut commonmark_cache,
                                    markdown,
                                );
                                ui.add_space(-10.);
                            });

                        egui::CollapsingHeader::new(egui::RichText::from("Settings").heading())
                            .default_open(true)
                            .show(ui, |ui| {
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
                                        &mut params.earthquake_region_size_mi,
                                        1e2..=5e2,
                                    )
                                    .text("earthquake region size")
                                    .suffix(" mi"),
                                );
                                ui.add(
                                    egui::Slider::new(&mut params.earthquake_height_m, -5e0..=5e0)
                                        .text("earthquake height")
                                        .suffix(" m"),
                                );
                                ui.label(format!(
                                    "{:.1} hr elapsed sim time",
                                    renderable.t_nondimen() * time_scale_hr()
                                ));
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
                                ui.add(egui::Checkbox::new(
                                    &mut params.show_point_cloud,
                                    "show quadrature points",
                                ));
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
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.substeps_per_physics_step,
                                        1..=30,
                                    )
                                    .text("substeps per physics update"),
                                );
                                ui.label(format!(
                                    "{:.0} wall ms/substep, {:.0} wall ms/render, {:.0} sim \
                                     s/substep",
                                    wall_time_per_renderable_sec * 1000.,
                                    wall_time_per_render_sec * 1000.,
                                    sim_time_per_renderable * time_scale_s()
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
                            });

                        ui.collapsing(egui::RichText::from("Details").heading(), |ui| {
                            let markdown = indoc::indoc! {"
                                Solves the shallow water equations pseudospectrally using a
                                spherical harmonic basis. (Torus uses a rectangular domain with
                                periodic conditions and a Fourier basis.) Physical effects included:
                                
                                * Viscosity (negligible for tsunamis but can be artifically \
                                increased)
                                * Coriolis force
                                * Tides (Sun ignored, Moon taken to be stationary) (planned)
                                
                                Tech stack: Rust/WebAssembly/WebGL/[`egui`](https://www.egui.rs/)/
                                [`three-d`](https://github.com/asny/three-d).

                                By Josh Burkart. View and contribute to the code on
                                [GitHub](https://github.com/joshburkart/tsunami)!
                            "};
                            ui.add_space(-10.);
                            egui_commonmark::CommonMarkViewer::new("details").show(
                                ui,
                                &mut commonmark_cache,
                                markdown,
                            );
                            ui.add_space(-10.);
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

        // Exponential moving average.
        wall_time_per_render_sec =
            0.95 * wall_time_per_render_sec + 0.05 * frame_input.elapsed_time as Float / 1000.;

        FrameOutput::default()
    });
}
