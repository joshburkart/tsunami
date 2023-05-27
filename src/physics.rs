use pyo3::prelude::*;

use crate::{geom, indexing, Float};

#[pyclass]
#[derive(Clone)]
pub struct Fields {
    pub height: geom::HorizScalarField,
    pub velocity: geom::VectorField,
    pub pressure: geom::ScalarField,
}

#[pyclass]
pub struct Solver {
    dynamic_geometry: Option<geom::DynamicGeometry>,
    rain_rate: geom::HorizScalarField,
    fields: Fields,
}
impl Solver {
    pub fn new(
        initial_dynamic_geometry: geom::DynamicGeometry,
        initial_fields: Fields,
        rain_rate: geom::HorizScalarField,
    ) -> Self {
        Self {
            dynamic_geometry: Some(initial_dynamic_geometry),
            rain_rate,
            fields: initial_fields,
        }
    }

    pub fn step(&mut self, dt: Float) {
        let new_height = perform_height_update(
            dt,
            self.dynamic_geometry.as_ref().unwrap(),
            &self.rain_rate,
            &mut self.fields,
        );

        self.dynamic_geometry = Some(geom::DynamicGeometry::new(
            self.dynamic_geometry.take().unwrap().into_static_geometry(),
            &new_height,
        ));

        // Interpolate velocity to new height map.
        // let new_velocity =
        //     interpolate_onto(&z_lattice.centers, &fields.velocity,
        // &new_z_axis.centers);

        // Compute new pressure field.
        // let new_pressure = compute_pressure(&self.grid, z_lattice,
        // &new_velocity);

        // Perform velocity update.
        // TODO
    }
}
#[pymethods]
impl Solver {
    #[getter]
    pub fn grid(&self) -> geom::Grid {
        self.dynamic_geometry.as_ref().unwrap().grid().clone()
    }

    #[getter]
    pub fn z_lattice<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.dynamic_geometry.as_ref().unwrap().z_lattice(py)
    }

    #[pyo3(name = "step")]
    pub fn step_py(&mut self, dt: Float) {
        self.step(dt)
    }
}

// /// Linearly interpolate a velocity vector field array `v` with vertical
// coordinate array `z` onto a /// new vertical coordinate array `new_z`. Use
// constant extrapolation if a value of `new_z` falls /// outside the range of
// `z`. fn interpolate_onto(z: &geom::ScalarField, v: &geom::VectorField, new_z:
// &geom::ScalarField) -> geom::VectorField {     let mut new_v =
// geom::VectorField::zeros((v.dim().0, v.dim().1, new_z.dim().2, v.dim().3));
//     let dim = z.dim();
//     for i in 0..dim.0 {
//         for j in 0..dim.1 {
//             let zij = z.slice(s![i, j, ..]);
//             let vij = v.slice(s![i, j, .., ..]);
//             let new_zij = new_z.slice(s![i, j, ..]);
//             let mut new_vij = new_v.slice_mut(s![i, j, .., ..]);

//             let mut k = 0usize;
//             for (new_k, &new_zijk) in new_zij.iter().enumerate() {
//                 while k < dim.2 && zij[k] < new_zijk {
//                     k += 1;
//                 }
//                 new_vij.slice_mut(s![new_k, ..]).assign(&if k == 0 {
//                     vij.slice(s![0, ..]).into_owned()
//                 } else if k == dim.2 {
//                     vij.slice(s![dim.2 - 1, ..]).into_owned()
//                 } else {
//                     //    <---------delta_zijk--------->
//                     //    |          |                 |
//                     // zij_left  new_zijk          zij_right
//                     let zij_left = zij[k - 1];
//                     let zij_right = zij[k];
//                     let delta_zijk = zij_right - zij_left;
//                     let alpha = if delta_zijk == 0. {
//                         0.
//                     } else {
//                         (new_zijk - zij_left) / delta_zijk
//                     };
//                     (1. - alpha) * &vij.slice(s![k - 1, ..]) + alpha *
// &vij.slice(s![k, ..])                 });
//             }
//         }
//     }
//     new_v
// }

fn perform_height_update(
    dt: Float,
    dynamic_geometry: &geom::DynamicGeometry,
    rain_rate: &geom::HorizScalarField,
    fields: &Fields,
) -> geom::HorizScalarField {
    let mut new_height = fields.height.clone();
    let cell_footprint_indexing = dynamic_geometry.grid().cell_footprint_indexing();
    for cell_footprint_index in indexing::iter_indices(cell_footprint_indexing) {
        let mut total_mass_flux = 0.;
        let cell_footprint_edges =
            cell_footprint_indexing.compute_footprint_edges(cell_footprint_index);
        for cell_index in dynamic_geometry
            .grid()
            .cell_indexing()
            .column(cell_footprint_index)
        {
            println!(" Cell index {cell_index:?}");
            for cell_footprint_edge in cell_footprint_edges {
                let vertical_face =
                    dynamic_geometry.compute_vertical_face(cell_index, cell_footprint_edge);
                println!("  Vertical face {vertical_face:?}");
                let velocity_at_face = fields.velocity.interpolate_to_face(&vertical_face);
                println!(
                    "  Flux {:.3e}",
                    velocity_at_face.dot(&vertical_face.outward_normal) * vertical_face.area
                );
                total_mass_flux +=
                    velocity_at_face.dot(&vertical_face.outward_normal) * vertical_face.area;
            }
        }
        println!("Cell footprint {cell_footprint_index:?}, mass flux: {total_mass_flux:.3e}");
        *new_height.center_mut(cell_footprint_index) -=
            dt * total_mass_flux / dynamic_geometry.grid().footprint_area();
        *new_height.center_mut(cell_footprint_index) += dt * rain_rate.center(cell_footprint_index);
    }
    new_height
}

// fn compute_pressure(grid: &Grid, z_lattice: &ZLattice, velocity:
// &geom::VectorField) -> geom::ScalarField {     // let cg_solver =
// argmin::solver::conjugategradient::ConjugateGradient::new(b)     todo!()
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector3;

    #[test]
    fn test_perform_height_update() {
        let x_axis = geom::Axis::new(-1., 1., 3);
        let y_axis = geom::Axis::new(10., 11., 4);
        let grid = geom::Grid::new(x_axis, y_axis, 5);
        let static_geometry = geom::StaticGeometry::new(grid, &|_, _| 0.);

        let mut height = geom::HorizScalarField::new(static_geometry.grid(), |_, _| 0.);
        for cell_footprint_index in
            indexing::iter_indices(static_geometry.grid().cell_footprint_indexing())
        {
            *height.center_mut(cell_footprint_index) = 3.;
        }
        let dynamic_geometry = geom::DynamicGeometry::new(static_geometry, &height);

        let fields = Fields {
            height: height.clone(),
            velocity: geom::VectorField::new(&dynamic_geometry, |_, _, _| Vector3::zeros()),
            pressure: geom::ScalarField::new(&dynamic_geometry, |_, _, _| 0.),
        };

        // No rain.
        {
            let rain_rate = geom::HorizScalarField::new(dynamic_geometry.grid(), |_, _| 0.);

            let new_height = perform_height_update(1., &dynamic_geometry, &rain_rate, &fields);

            approx::assert_abs_diff_eq!(new_height, height);
        }
        // Some rain.
        {
            let rain_rate = geom::HorizScalarField::new(dynamic_geometry.grid(), |_, _| 1.5e-2);

            let new_height = perform_height_update(1., &dynamic_geometry, &rain_rate, &fields);

            approx::assert_abs_diff_eq!(new_height, height + 1.5e-2);
        }
    }
}
