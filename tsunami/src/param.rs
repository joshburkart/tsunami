use three_d::{egui, vec3, Vec3};

use crate::{geom, render, Float};

#[derive(Clone, Debug)]
pub struct Parameters {
    pub physics: PhysicsParameters,
    pub visualization: VisualizationParameters,
    pub performance: PerformanceParameters,

    pub earthquake_position: Option<Vec3>,
    // Response set by the physics thread after an earthquake trigger has been read and handled.
    pub earthquake_triggered: bool,
}

impl Parameters {
    pub fn generate_ui(
        &mut self,
        ui: &mut egui::Ui,
        renderable: &render::Renderable,
        performance_stats: PerformanceStats,
        geom_change: &mut bool,
    ) {
        egui::CollapsingHeader::new(egui::RichText::from("Controls").size(crate::HEADING_SIZE))
            .default_open(true)
            .show(ui, |ui| {
                ui.add_space(5.);

                Preset::generate_ui(ui, self, geom_change);
                self.physics.generate_ui(ui, &renderable, geom_change);
                self.visualization.generate_ui(ui);
                self.performance
                    .generate_ui(ui, performance_stats, geom_change);
            });
    }
}

#[derive(Copy, Clone, strum::EnumIter)]
pub enum Preset {
    Vortices,
    Tides,
    Torus,
}

impl Preset {
    fn generate_ui(ui: &mut egui::Ui, params: &mut Parameters, geom_change: &mut bool) {
        ui.label(egui::RichText::new("Presets").strong());
        ui.horizontal(|ui| {
            use strum::IntoEnumIterator;
            for preset in Self::iter() {
                let selection_params = Parameters::from_preset(preset);
                let params_same = params.physics == selection_params.physics;
                if ui
                    .selectable_label(params_same, preset.to_string())
                    .clicked()
                    && !params_same
                {
                    *params = selection_params;
                    *geom_change = true;
                }
            }
        });
        ui.add_space(10.);
    }
}

impl std::fmt::Display for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Preset::Vortices => "Vortices",
            Preset::Tides => "Tides",
            Preset::Torus => "Torus",
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PhysicsParameters {
    pub log10_kinematic_viscosity_rel_to_water: Float,
    pub rotation_period_hr: Float,
    pub lunar_distance_rel_to_actual: Float,
    pub earthquake_region_size_mi: Float,
    pub earthquake_height_m: Float,
    pub geometry_type: geom::GeometryType,
}

impl PhysicsParameters {
    fn generate_ui(
        &mut self,
        ui: &mut three_d::egui::Ui,
        renderable: &render::Renderable,
        geom_change: &mut bool,
    ) {
        ui.label(egui::RichText::new("Physics").strong());
        ui.add(
            egui::Slider::new(
                &mut self.log10_kinematic_viscosity_rel_to_water,
                (0.)..=(13.),
            )
            .text("viscosity")
            .prefix("10^")
            .suffix(" × water"),
        );
        ui.add(
            egui::Slider::new(&mut self.rotation_period_hr, (5.)..=(100.))
                .text("rotation period")
                .suffix(" hr"),
        );
        ui.add(
            egui::Slider::new(&mut self.lunar_distance_rel_to_actual, (0.5)..=(10.))
                .logarithmic(true)
                .text("lunar distance")
                .suffix(" × actual"),
        );
        ui.add(
            egui::Slider::new(&mut self.earthquake_region_size_mi, 2e2..=1e3)
                .text("earthquake region size")
                .suffix(" mi"),
        );
        ui.add(
            egui::Slider::new(&mut self.earthquake_height_m, -6e0..=6e0)
                .text("earthquake height")
                .suffix(" m"),
        );
        ui.horizontal(|ui| {
            *geom_change |= ui
                .radio_value(
                    &mut self.geometry_type,
                    geom::GeometryType::Sphere,
                    "Sphere",
                )
                .changed();
            *geom_change |= ui
                .radio_value(&mut self.geometry_type, geom::GeometryType::Torus, "Torus")
                .changed();
            ui.label("geometry");
        });
        ui.horizontal(|ui| {
            *geom_change |= ui.button("Restart").clicked();
            ui.label(format!(
                "{:.1} hr elapsed sim time",
                renderable.t_nondimen() * crate::time_scale_hr()
            ));
        });
        ui.add_space(10.);
    }
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            log10_kinematic_viscosity_rel_to_water: 0.,
            rotation_period_hr: 24.,
            lunar_distance_rel_to_actual: 1.,
            earthquake_region_size_mi: 600.,
            earthquake_height_m: -4.,
            geometry_type: geom::GeometryType::Sphere,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VisualizationParameters {
    pub height_exaggeration_factor: Float,
    pub velocity_exaggeration_factor: Float,
    pub show_features: ShowFeatures,
    pub show_rotation: ShowRotation,
}

impl VisualizationParameters {
    fn generate_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Visualization").strong());
        ui.add(
            egui::Slider::new(&mut self.height_exaggeration_factor, 1e0..=1e4)
                .logarithmic(true)
                .text("height exaggeration")
                .suffix("×"),
        );
        ui.add(
            egui::Slider::new(&mut self.velocity_exaggeration_factor, 1e0..=1e5)
                .logarithmic(true)
                .text("velocity exaggeration")
                .suffix("×"),
        );
        self.show_features.generate_ui(ui);
        self.show_rotation.generate_ui(ui);
        ui.add_space(10.);
    }
}

impl Default for VisualizationParameters {
    fn default() -> Self {
        Self {
            height_exaggeration_factor: 500.,
            velocity_exaggeration_factor: 1.5e3,
            show_features: ShowFeatures::Tracers,
            show_rotation: ShowRotation::None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PerformanceParameters {
    pub substeps_per_physics_step: usize,
    pub resolution_level: u32,
    pub log2_num_tracers: u32,
}

impl PerformanceParameters {
    fn generate_ui(
        &mut self,
        ui: &mut egui::Ui,
        performance_stats: PerformanceStats,
        geom_change: &mut bool,
    ) {
        ui.label(egui::RichText::new("Performance").strong());
        ui.horizontal(|ui| {
            for n in 5..=8 {
                *geom_change |= ui
                    .radio_value(&mut self.resolution_level, n, format!("{}", 2i32.pow(n)))
                    .changed();
            }
            ui.label("resolution");
        });
        ui.horizontal(|ui| {
            for n in 12..=15 {
                *geom_change |= ui
                    .radio_value(
                        &mut self.log2_num_tracers,
                        n,
                        format!("{}k", 2i32.pow(n) / 1000),
                    )
                    .changed();
            }
            ui.label("num tracers");
        });
        // ui.add(
        //     egui::Slider::new(
        //         &mut params.substeps_per_physics_step,
        //         1..=30,
        //     )
        //     .text("substeps per physics update"),
        // );
        performance_stats.generate_ui(ui);

        ui.add_space(10.);
    }
}

impl Default for PerformanceParameters {
    fn default() -> Self {
        Self {
            resolution_level: 6,
            log2_num_tracers: 12,
            substeps_per_physics_step: 1,
        }
    }
}

impl Parameters {
    pub fn from_preset(preset: Preset) -> Self {
        match preset {
            Preset::Vortices => Self {
                physics: PhysicsParameters {
                    lunar_distance_rel_to_actual: 10.,
                    rotation_period_hr: 5.,
                    earthquake_region_size_mi: 1000.,
                    earthquake_height_m: -3.5,
                    ..Default::default()
                },
                visualization: VisualizationParameters {
                    velocity_exaggeration_factor: 8e3,
                    ..Default::default()
                },
                performance: PerformanceParameters {
                    log2_num_tracers: 13,
                    ..Default::default()
                },
                earthquake_position: Some(vec3(0.5, 1., 1.)),
                ..Self::default()
            },
            Preset::Tides => Self {
                visualization: VisualizationParameters {
                    show_rotation: ShowRotation::Inertial,
                    height_exaggeration_factor: 1500.,
                    ..Default::default()
                },
                ..Default::default()
            },
            Preset::Torus => Self {
                physics: PhysicsParameters {
                    geometry_type: geom::GeometryType::Torus,
                    ..Default::default()
                },
                visualization: VisualizationParameters {
                    velocity_exaggeration_factor: 100.,
                    ..Default::default()
                },
                performance: PerformanceParameters {
                    resolution_level: 7,
                    ..Default::default()
                },
                ..Self::default()
            },
        }
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            physics: Default::default(),
            visualization: Default::default(),
            performance: Default::default(),
            earthquake_position: None,
            earthquake_triggered: false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, strum::EnumIter)]
pub enum ShowFeatures {
    Quadrature,
    Tracers,
    None,
}

impl ShowFeatures {
    fn generate_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            use strum::IntoEnumIterator;

            for value in Self::iter() {
                ui.radio_value(self, value, value.to_string());
            }
            ui.label("show");
        });
    }
}

impl std::fmt::Display for ShowFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ShowFeatures::Quadrature => "quadrature points",
            ShowFeatures::Tracers => "path lines",
            ShowFeatures::None => "none",
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, strum::EnumIter)]
pub enum ShowRotation {
    Corotating,
    Inertial,
    None,
}

impl ShowRotation {
    fn generate_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            use strum::IntoEnumIterator;

            for value in Self::iter() {
                ui.radio_value(self, value, value.to_string());
            }
            ui.label("show rotation 🤢");
        });
    }
}

impl std::fmt::Display for ShowRotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ShowRotation::Corotating => "corotating",
            ShowRotation::Inertial => "inertial",
            ShowRotation::None => "none",
        })
    }
}

#[derive(Clone, Copy, Default)]
pub struct PerformanceStats {
    pub wall_time_per_renderable_sec: Float,
    pub wall_time_per_render_sec: Float,
    pub sim_time_per_renderable: Float,
}

impl PerformanceStats {
    pub fn update(&mut self, instantaneous: Self) {
        // Exponential moving averages.
        const CURRENT_WEIGHT: Float = 0.98;
        self.wall_time_per_renderable_sec = CURRENT_WEIGHT * self.wall_time_per_renderable_sec
            + (1. - CURRENT_WEIGHT) * instantaneous.wall_time_per_renderable_sec;
        self.sim_time_per_renderable = CURRENT_WEIGHT * self.sim_time_per_renderable
            + (1. - CURRENT_WEIGHT) * instantaneous.sim_time_per_renderable;
        self.wall_time_per_render_sec = CURRENT_WEIGHT * self.wall_time_per_render_sec
            + (1. - CURRENT_WEIGHT) * instantaneous.wall_time_per_render_sec;
    }

    fn generate_ui(&self, ui: &mut egui::Ui) {
        ui.label(format!(
            "({:.0}, {:.0}) wall ms/(step, render)",
            self.wall_time_per_renderable_sec * 1000.,
            self.wall_time_per_render_sec * 1000.,
        ));
        ui.label(format!(
            "({:.0}, {:.1}) sim s/(step, wall ms)",
            self.sim_time_per_renderable * crate::time_scale_s(),
            self.sim_time_per_renderable * crate::time_scale_s()
                / (self.wall_time_per_renderable_sec * 1000.),
        ));
    }
}

#[derive(Clone)]
pub struct ParametersMessage {
    pub params: Parameters,
    pub geometry_version: usize,
}
