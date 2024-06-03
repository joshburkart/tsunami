use three_d::*;
use wasm_bindgen::prelude::*;

use flow::Float;

mod geom;

#[derive(Clone)]
struct Parameters {
    pub kinematic_viscosity_rel_to_water: Float,
    pub height_exaggeration_factor: Float,
    pub time_step: Float,
    pub resolution_level: u32,
    pub show_point_cloud: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            kinematic_viscosity_rel_to_water: 1.,
            height_exaggeration_factor: 40.,
            time_step: 100.,
            resolution_level: 6,
            show_point_cloud: true,
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
        vec3(7.0, 7.0, 7.0) / 3.,
        vec3(0., 0., 0.),
        vec3(0.0, 0.0, 1.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(*camera.target(), 5. / 3., 5.);

    let mut params = Parameters::default();

    let mut geometry_type = geom::GeometryType::Sphere;
    let mut geometry = geom::Geometry::new(geometry_type, params.resolution_level);

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

    let mut frame_time_history = egui::util::History::new(50..1000, 500.);

    let paths = [
        "right.jpg",
        "left.jpg",
        "top.jpg",
        "bottom.jpg",
        "front.jpg",
        "back.jpg",
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

    let height_array = geometry.height_grid();
    let new_mesh = geometry.make_mesh(&height_array, params.height_exaggeration_factor);
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
                positions: Positions::F64(
                    geometry.make_points(&height_array, params.height_exaggeration_factor),
                ),
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
            geometry = geom::Geometry::new(geometry_type, params.resolution_level);
            reset = false;
        }

        geometry.set_kinematic_viscosity(1e-3 * params.kinematic_viscosity_rel_to_water); // TODO
        geometry.integrate(3e-5 * params.time_step); // TODO

        {
            let height_array = geometry.height_grid();
            mesh_model.geometry = Mesh::new(
                &context,
                &geometry
                    .make_mesh(&&height_array, params.height_exaggeration_factor)
                    .clone()
                    .into(),
            );

            if params.show_point_cloud {
                point_cloud_model.geometry = InstancedMesh::new(
                    &context,
                    &PointCloud {
                        positions: Positions::F64(
                            geometry.make_points(&height_array, params.height_exaggeration_factor),
                        ),
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
                egui::Window::new("Tsunami Simulator")
                    .vscroll(true)
                    .show(gui_context, |ui| {
                        egui::CollapsingHeader::new(egui::RichText::from("Info").heading())
                            .default_open(true)
                            .show(ui, |ui| {
                                egui_commonmark::commonmark!(
                                    "info",
                                    ui,
                                    &mut commonmark_cache,
                                    "Alpha version -- please **do not share yet**! Many rough \
                                    edges, e.g. viscosity slider has nothing to do with water's \
                                    viscosity despite what it says, advection term not yet \
                                    implemented in spherical geometry, etc.\n\
                                    \n\
                                    Planned features: realistic terrain (continents/sea floor/\
                                    etc.), click to set off a tsunami, and more..."
                                );
                            });

                        egui::CollapsingHeader::new(egui::RichText::from("Settings").heading())
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("Geometry").strong());
                                ui.horizontal(|ui| {
                                    reset |= ui
                                        .radio_value(
                                            &mut geometry_type,
                                            geom::GeometryType::Sphere,
                                            "Sphere",
                                        )
                                        .changed()
                                        | ui.radio_value(
                                            &mut geometry_type,
                                            geom::GeometryType::Torus,
                                            "Torus",
                                        )
                                        .changed();
                                });
                                ui.add_space(10.);

                                ui.label(egui::RichText::new("Physics").strong());
                                ui.add(
                                    egui::Slider::new(
                                        &mut params.kinematic_viscosity_rel_to_water,
                                        1e-1..=1e2,
                                    )
                                    .logarithmic(true)
                                    .text("kinematic viscosity")
                                    .suffix("× water"),
                                );
                                ui.add_space(10.);

                                ui.label(egui::RichText::new("Visualization").strong());
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
                                ui.add_space(10.);

                                ui.label(egui::RichText::new("Performance").strong());
                                reset |= ui
                                    .add(
                                        egui::Slider::new(&mut params.resolution_level, 4..=7)
                                            .fixed_decimals(0)
                                            .step_by(1.)
                                            .prefix("2^")
                                            .text("num subdivisions"),
                                    )
                                    .changed();
                                ui.add(
                                    egui::Slider::new(&mut params.time_step, 1e0..=1e3)
                                        .logarithmic(true)
                                        .text("time step")
                                        .suffix(" s"),
                                );
                                ui.label(format!(
                                    "{:.0} ms/time step",
                                    frame_time_history.average().unwrap_or(0.)
                                ));
                            });

                        ui.collapsing(egui::RichText::from("Details").heading(), |ui| {
                            egui_commonmark::commonmark!(
                                "details",
                                ui,
                                &mut commonmark_cache,
                                "Solves the shallow water equations pseudospectrally. Torus \
                                uses a rectangular domain with periodic boundary conditions and a \
                                Fourier basis. Sphere uses a spherical harmonic basis.\n\
                                \n\
                                Tech stack: Rust/Wasm/WebGL/[`egui`](https://www.egui.rs/)/\
                                [`three-d`](https://github.com/asny/three-d).\n\
                                \n\
                                By Josh Burkart: [repo](https://gitlab.com/joshburkart/flow)."
                            );
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
