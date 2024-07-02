#![cfg_attr(feature = "wasm-bindgen-rayon", feature(stdarch_wasm_atomic_wait))]

use flow::Float;
use three_d::*;
use wasm_bindgen::prelude::*;

pub mod geom;
mod param;
mod render;
mod util;

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
const TITLE_SIZE: f32 = 21.;
const HEADING_SIZE: f32 = 18.;

fn time_scale_s() -> Float {
    EARTH_RADIUS_M / (GRAV_ACCEL_M_PER_S2 * OCEAN_DEPTH_M).sqrt()
}

fn time_scale_hr() -> Float {
    time_scale_s() / 60. / 60.
}

fn water_kinematic_viscosity_nondimen() -> Float {
    WATER_KINEMATIC_VISCOSITY_M2_PER_S / EARTH_RADIUS_M.powi(2) * time_scale_s()
}

fn physics_loop(
    mut geometry: geom::Geometry,
    shared_params_message: std::sync::Arc<std::sync::RwLock<param::ParametersMessage>>,
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
                .make_renderables(params_message.params.performance.substeps_per_physics_step)
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
                geometry = geom::Geometry::new(
                    new_params.physics.geometry_type,
                    new_params.performance.resolution_level,
                );
            }
            params_message = new_params_message.clone();

            geometry.set_kinematic_viscosity(
                water_kinematic_viscosity_nondimen()
                    * (10 as Float).powf(
                        params_message
                            .params
                            .physics
                            .log10_kinematic_viscosity_rel_to_water,
                    ),
            );
            geometry.set_rotation_angular_speed(
                flow::float_consts::TAU / params_message.params.physics.rotation_period_hr
                    * time_scale_hr(),
            );
            geometry.set_lunar_distance(
                params_message.params.physics.lunar_distance_rel_to_actual
                    * LUNAR_DISTANCE_NONDIMEN,
            );
            geometry.set_velocity_exaggeration_factor(
                params_message
                    .params
                    .visualization
                    .velocity_exaggeration_factor,
            );
        }

        // Update parameters if needed.
        if let Some(earthquake_position) = params_message.params.earthquake_position {
            let mut shared_params_writer = shared_params_message.write().unwrap();
            shared_params_writer.params.earthquake_position = None;
            if !params_message.params.earthquake_triggered {
                geometry.trigger_earthquake(
                    earthquake_position,
                    params_message.params.physics.earthquake_region_size_mi / EARTH_RADIUS_MI,
                    params_message.params.physics.earthquake_height_m / OCEAN_DEPTH_M,
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

    let mut params = param::Parameters::from_preset(param::Preset::Whirlpool);
    let mut geometry_version = 0;
    let geometry = geom::Geometry::new(
        params.physics.geometry_type,
        params.performance.resolution_level,
    );
    let mut renderable = geometry.make_renderables(1).pop().unwrap();
    let (renderable_sender, renderable_reader) = std::sync::mpsc::sync_channel(5);
    let shared_params_message_write = std::sync::Arc::new(std::sync::RwLock::new(
        param::ParametersMessage {
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
        vec3(1., 1., 0.5).normalize() * 3.,
        vec3(0., 0., 0.),
        vec3(0.0, 0.0, 1.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(*camera.target(), 5. / 3., 5.);

    let mut gui = GUI::new(&context);
    let fonts = {
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "Roboto".to_owned(),
            egui::FontData::from_static(include_bytes!("../assets/Roboto-Light.ttf")),
        );
        fonts.font_data.insert(
            "NotoEmoji-Regular".to_owned(),
            egui::FontData::from_static(include_bytes!("../assets/NotoEmoji-Regular.ttf")),
        );
        *fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default() = vec!["Roboto".to_owned(), "NotoEmoji-Regular".to_owned()];
        fonts
    };
    gui.context().set_fonts(fonts.clone());
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
    let directional = DirectionalLight::new(&context, 5.0, Srgba::WHITE, &vec3(-1., 0., 0.));

    let mut performance_stats = param::PerformanceStats {
        wall_time_per_renderable_sec: 0.,
        wall_time_per_render_sec: 0.,
        sim_time_per_renderable: 0.,
    };

    let rendering_data =
        renderable.make_rendering_data(params.visualization.height_exaggeration_factor);

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

    let mut moon_sprite_object = {
        let mut raw_assets = three_d_asset::io::RawAssets::new();
        raw_assets.insert(
            "brightmoon.png",
            include_bytes!("../assets/brightmoon.png").to_vec(),
        );

        let mut moon_image: CpuTexture = raw_assets
            .deserialize("brightmoon.png")
            .inspect_err(|e| log::error!("{e:?}"))
            .unwrap();
        moon_image.data.to_linear_srgb();
        let moon_material = ColorMaterial {
            color: Srgba::WHITE,
            texture: Some(Texture2DRef::from_cpu_texture(&context, &moon_image)),
            is_transparent: true,
            render_states: RenderStates {
                write_mask: WriteMask::COLOR,
                blend: Blend::TRANSPARENCY,
                ..Default::default()
            },
        };
        Gm {
            geometry: Sprites::new(
                &context,
                &[vec3(
                    (LUNAR_DISTANCE_NONDIMEN * params.physics.lunar_distance_rel_to_actual) as f32,
                    0.,
                    0.,
                )],
                None,
            ),
            material: moon_material,
        }
    };

    let mut wall_time_of_last_renderable = web_time::Instant::now();
    let mut sim_time_of_last_renderable = 0.;

    let mut double_click_detector = util::DoubleClickDetector::default();

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
            *shared_params_message = param::ParametersMessage {
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

            performance_stats.update(param::PerformanceStats {
                wall_time_per_renderable_sec: now
                    .duration_since(wall_time_of_last_renderable)
                    .as_secs_f64(),
                wall_time_per_render_sec: frame_input.elapsed_time as Float / 1000.,
                sim_time_per_renderable: renderable.t_nondimen() - sim_time_of_last_renderable,
            });

            wall_time_of_last_renderable = now;
            sim_time_of_last_renderable = renderable.t_nondimen();
        }
        let rendering_data =
            renderable.make_rendering_data(params.visualization.height_exaggeration_factor);
        let rotation = if params.visualization.show_rotation {
            Matrix4::from_axis_angle(
                Vector3::unit_z(),
                -radians(renderable.rotational_phase_rad() as f32),
            )
        } else {
            Matrix4::identity()
        };

        {
            mesh_model.geometry = Mesh::new(&context, &rendering_data.mesh.into());
            mesh_model.set_transformation(rotation);

            moon_sprite_object.geometry = Sprites::new(
                &context,
                &[vec3(
                    (LUNAR_DISTANCE_NONDIMEN * params.physics.lunar_distance_rel_to_actual) as f32,
                    0.,
                    0.,
                )],
                None,
            );

            if let param::ShowPoints::Quadrature = params.visualization.show_points {
                quadrature_point_cloud_model.geometry = InstancedMesh::new(
                    &context,
                    &PointCloud {
                        positions: Positions::F32(
                            rendering_data
                                .quadrature_points
                                .into_iter()
                                .map(|point| {
                                    rotation.transform_vector(Vector3 {
                                        x: point.x as f32,
                                        y: point.y as f32,
                                        z: point.z as f32,
                                    })
                                })
                                .collect(),
                        ),
                        colors: None,
                    }
                    .into(),
                    &point_mesh,
                );
                quadrature_point_cloud_model.material.render_states.cull = Cull::None;
            } else {
                quadrature_point_cloud_model.material.render_states.cull = Cull::FrontAndBack;
            }

            if let param::ShowPoints::Tracer = params.visualization.show_points {
                tracer_point_cloud_model.geometry = InstancedMesh::new(
                    &context,
                    &PointCloud {
                        positions: Positions::F32(
                            rendering_data
                                .tracer_points
                                .into_iter()
                                .map(|point| {
                                    rotation.transform_vector(Vector3 {
                                        x: point.x as f32,
                                        y: point.y as f32,
                                        z: point.z as f32,
                                    })
                                })
                                .collect(),
                        ),
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
                    if let util::DoubleClickResult::Detected =
                        double_click_detector.process_release()
                    {
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
                        params.physics.geometry_type,
                    ) {
                        let mut tsunami_hint_circle = CpuMesh::circle(20);
                        let axis = space_position.cross(Vector3::unit_z()).normalize();
                        let angle = space_position.angle(Vector3::unit_z());
                        tsunami_hint_circle
                            .transform(&Matrix4::from_scale(
                                (params.physics.earthquake_region_size_mi / EARTH_RADIUS_MI / 2.)
                                    as f32,
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
        tsunami_hint_circle_object.set_transformation(rotation);

        // Set hint circle color to red if completing a double click.
        double_click_detector.process_idle();
        match double_click_detector.state {
            util::DoubleClickDetectorState::Idle
            | util::DoubleClickDetectorState::FirstPress(_)
            | util::DoubleClickDetectorState::FirstRelease(_) => {
                tsunami_hint_circle_object.material.color.g = u8::MAX;
                tsunami_hint_circle_object.material.color.b = u8::MAX;
            }
            util::DoubleClickDetectorState::SecondPress => {
                tsunami_hint_circle_object.material.color.g = 0;
                tsunami_hint_circle_object.material.color.b = 0;
            }
        }

        let mut geom_change = false;
        const UI_WIDTH: f32 = 333.;

        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                egui::SidePanel::left("left")
                    .exact_width(UI_WIDTH)
                    .resizable(false)
                    .show(gui_context, |ui| {
                        ui.add_space(10.);
                        ui.label(egui::RichText::new(TITLE).size(TITLE_SIZE).strong());
                        egui::CollapsingHeader::new(
                            egui::RichText::from("Basics").size(HEADING_SIZE),
                        )
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

                        egui::CollapsingHeader::new(egui::RichText::from("Details").heading())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add_space(-10.);
                                egui_commonmark::CommonMarkViewer::new("details").show(
                                    ui,
                                    &mut commonmark_cache,
                                    DETAILS_MARKDOWN,
                                );
                                ui.add_space(-10.);
                            });
                    });

                egui::SidePanel::right("right")
                    .exact_width(UI_WIDTH)
                    .resizable(false)
                    .show(gui_context, |ui| {
                        ui.add_space(10.);
                        params.generate_ui(ui, &renderable, performance_stats, &mut geom_change);
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

        // We rotated the physical objects above; now we need to rotate the camera as
        // well so that we are observing from a corotating frame.
        let rotated_camera = {
            let position = rotation.transform_vector(*camera.position());
            let target = rotation.transform_vector(*camera.target());
            let up = rotation.transform_vector(*camera.up());

            let mut rotated_camera = camera.clone();
            rotated_camera.set_view(position, target, up);
            rotated_camera
        };
        frame_input
            .screen()
            .render(
                &rotated_camera,
                std::iter::empty()
                    .chain(&mesh_model)
                    .chain(&quadrature_point_cloud_model)
                    .chain(&tracer_point_cloud_model)
                    .chain(&tsunami_hint_circle_object)
                    .chain(&moon_sprite_object)
                    .chain(&skybox),
                &[&ambient, &directional],
            )
            .write(|| gui.render())
            .unwrap();

        FrameOutput::default()
    });
}

const INFO_MARKDOWN: &'static str = indoc::indoc! {"
    Interactive ocean simulations in a web browser!
    
    * **Drag/scroll** to change the view
    * **Double click** to set off an earthquake/tsunami
    * **Dotted lines** are “tracers” indicating local water motion
    * **Play around** with different controls in the right panel

    Alpha version — please **do not share yet**!
"};

const DETAILS_MARKDOWN: &'static str = indoc::indoc! {"
    Solves the [shallow water equations](https://en.wikipedia.org/wiki/Shallow_water_equations)
    [pseudo](https://en.wikipedia.org/wiki/Pseudo-spectral_method)-\
    [spectrally](https://en.wikipedia.org/wiki/Spectral_method) using a
    [spherical harmonic](https://en.wikipedia.org/wiki/Spherical_harmonics) basis. (Toroidal
    geometry with more limited functionality also included for fun, which uses a rectangular domain
    with periodic boundary conditions and a Fourier basis.)
    
    Effects included:
    
    * [Viscosity](https://en.wikipedia.org/wiki/Viscosity) (negligible for tsunamis propagating in \
        deep water but can be artifically increased)
    * The [Coriolis force](https://en.wikipedia.org/wiki/Coriolis_force)
    * [Lunar tides](https://en.wikipedia.org/wiki/Tide) (taking the moon to be stationary for \
        simplicity)
    * [Advection](https://en.wikipedia.org/wiki/Advection) of entrained “tracer” particles
    * Realistic topography (continents/sea floor) *(planned)*

    Tech stack: Rust/WebAssembly/WebGL/[`egui`](https://www.egui.rs/)/
    [`three-d`](https://github.com/asny/three-d).

    By Josh Burkart. View and contribute to the code on
    [GitHub](https://github.com/joshburkart/tsunami)!
"};
