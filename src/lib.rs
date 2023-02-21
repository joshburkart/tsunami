#![deny(clippy::all)]

pub mod indexing;

use ndarray::{self as nd, s};

pub type Float = f64;

pub type Array1 = nd::Array1<Float>;
pub type Array2 = nd::Array2<Float>;
pub type Array3 = nd::Array3<Float>;

pub type CoordArray = nd::Array1<Float>;

pub type VectorField = nd::Array4<Float>;

pub type HorizVectorField = nd::Array3<Float>;

pub struct Grid {
    x_axis: Axis,
    y_axis: Axis,

    vertex_indexing: indexing::VertexIndexing,
}
impl Grid {
    pub fn new(x_axis: Axis, y_axis: Axis, num_z_points: usize) -> Self {
        let vertex_footprint_indexing = indexing::VertexFootprintIndexing {
            num_x_points: x_axis.vertices().len(),
            num_y_points: y_axis.vertices().len(),
        };
        let cell_footprint_indexing = indexing::CellFootprintIndexing {
            vertex_footprint_indexing,
        };
        let cell_indexing = indexing::CellIndexing {
            cell_footprint_indexing,
            num_z_cells_per_column: num_z_points - 1,
        };
        let vertex_indexing = indexing::VertexIndexing { cell_indexing };
        Self {
            x_axis,
            y_axis,
            vertex_indexing,
        }
    }

    pub fn x_axis(&self) -> &Axis {
        &self.x_axis
    }
    pub fn y_axis(&self) -> &Axis {
        &self.y_axis
    }

    pub fn cell_footprint_centroid(
        &self,
        cell_footprint_index: &indexing::CellFootprintIndex,
    ) -> [Float; 2] {
        let indexing::CellFootprintIndex {
            x: i,
            y: j,
            triangle,
        } = *cell_footprint_index;
        match triangle {
            indexing::Triangle::UpperLeft => [
                (2. / 3.) * self.x_axis.vertices[[i]] + (1. / 3.) * self.x_axis.vertices[[i + 1]],
                (2. / 3.) * self.y_axis.vertices[[j]] + (1. / 3.) * self.y_axis.vertices[[j + 1]],
            ],
            indexing::Triangle::LowerRight => [
                (1. / 3.) * self.x_axis.vertices[[i]] + (2. / 3.) * self.x_axis.vertices[[i + 1]],
                (1. / 3.) * self.y_axis.vertices[[j]] + (2. / 3.) * self.y_axis.vertices[[j + 1]],
            ],
        }
    }

    pub fn vertex_indexing(&self) -> &indexing::VertexIndexing {
        &self.vertex_indexing
    }
    pub fn cell_indexing(&self) -> &indexing::CellIndexing {
        &self.vertex_indexing.cell_indexing
    }
    pub fn vertex_footprint_indexing(&self) -> &indexing::VertexFootprintIndexing {
        &self
            .vertex_indexing
            .cell_indexing
            .cell_footprint_indexing
            .vertex_footprint_indexing
    }
    pub fn cell_footprint_indexing(&self) -> &indexing::CellFootprintIndexing {
        &self.vertex_indexing.cell_indexing.cell_footprint_indexing
    }
}

/// A fixed $x$ or $y$ axis.
pub struct Axis {
    vertices: CoordArray,
    centers: CoordArray,
    spacing: Float,
}
impl Axis {
    pub fn new(min: Float, max: Float, num_cells: usize) -> Self {
        let vertices = CoordArray::linspace(min, max, num_cells + 1);
        let centers = (&vertices.slice(s![..-1]) + &vertices.slice(s![1..])) / 2.;
        let spacing = vertices[1] - vertices[0];
        Self {
            vertices,
            centers,
            spacing,
        }
    }

    pub fn vertices(&self) -> &CoordArray {
        &self.vertices
    }
    pub fn centers(&self) -> &CoordArray {
        &self.centers
    }
    pub fn spacing(&self) -> Float {
        self.spacing
    }
}

/// A scalar field.
///
/// Defines the values of the field at cell centers.
pub struct ScalarField {
    /// Underlying array.
    ///
    /// Axes:
    ///     0: x index
    ///     1: y index
    ///     2: z index
    ///     3: Which truncated triangular prism (UL/LR)
    centers: nd::Array4<Float>,
}
impl ScalarField {
    pub fn center(&self, index: indexing::CellIndex) -> Float {
        self.centers[[
            index.footprint.x,
            index.footprint.y,
            index.z,
            index.footprint.triangle as usize,
        ]]
    }
}

pub struct Height {
    /// Axes:
    ///     0: x index
    ///     1: y index
    ///     2: Which triangle (UL/LR)
    centers: nd::Array3<Float>,
}
impl Height {
    pub fn zeros(cell_footprint_indexing: &indexing::CellFootprintIndexing) -> Self {
        Self {
            centers: nd::Array3::zeros((
                cell_footprint_indexing
                    .vertex_footprint_indexing
                    .num_x_points,
                cell_footprint_indexing
                    .vertex_footprint_indexing
                    .num_y_points,
                2,
            )),
        }
    }
    pub fn center(&self, cell_footprint_index: indexing::CellFootprintIndex) -> Float {
        self.centers[[
            cell_footprint_index.x,
            cell_footprint_index.y,
            cell_footprint_index.triangle as usize,
        ]]
    }
    pub fn center_mut(&mut self, cell_footprint_index: indexing::CellFootprintIndex) -> &mut Float {
        &mut self.centers[[
            cell_footprint_index.x,
            cell_footprint_index.y,
            cell_footprint_index.triangle as usize,
        ]]
    }
}

pub struct Terrain {
    /// Axes:
    ///     0: x index
    ///     1: y index
    ///     2: Which triangle (UL/LR)
    centers: nd::Array3<Float>,
    /// Axes:
    ///     0: x index
    ///     1: y index
    vertices: nd::Array2<Float>,
}
impl Terrain {
    pub fn new(vertices: nd::Array2<Float>) -> Self {
        let mut centers = nd::Array::zeros((vertices.dim().0 - 1, vertices.dim().1 - 1, 2));
        for i in 0..vertices.dim().0 - 1 {
            for j in 0..vertices.dim().1 - 1 {
                // Take vertex means for triangle centers.
                centers[[i, j, indexing::Triangle::UpperLeft as usize]] =
                    (vertices[[i, j]] + vertices[[i + 1, j + 1]] + vertices[[i, j + 1]]) / 3.;
                centers[[i, j, indexing::Triangle::LowerRight as usize]] =
                    (vertices[[i, j]] + vertices[[i + 1, j]] + vertices[[i + 1, j + 1]]) / 3.;
            }
        }
        Self { centers, vertices }
    }
}

pub struct ZLattice {
    lattice: nd::Array3<Float>,
}
impl ZLattice {
    pub fn compute(grid: &Grid, terrain: &Terrain, height: &Height) -> Self {
        let num_z_points = grid.vertex_indexing().num_z_points();
        let mut lattice = Array3::zeros((
            grid.vertex_indexing.num_x_points(),
            grid.vertex_indexing.num_y_points(),
            num_z_points,
        ));
        let vertex_footprint_indexing = grid.vertex_footprint_indexing();
        for vertex_footprint_index in indexing::iter_indices(vertex_footprint_indexing) {
            let height = mean(
                vertex_footprint_index
                    .cells(vertex_footprint_indexing)
                    .map(|cell_footprint_index| height.center(cell_footprint_index)),
            )
            .unwrap();
            let indexing::VertexFootprintIndex { x: i, y: j } = vertex_footprint_index;
            for k in 0..num_z_points {
                lattice[[i, j, k]] =
                    terrain.vertices[[i, j]] + (k as Float / (num_z_points as Float - 1.)) * height;
            }
        }
        Self { lattice }
    }
    // pub fn compute_lp(
    //     grid: &Grid,
    //     terrain: &Terrain,
    //     height: &Height,
    // ) -> Result<Self, highs::HighsModelStatus> {
    //     let mut problem = highs::RowProblem::new();

    //     // Define variables to be solved for.
    //     let vertex_variables = (0..grid.vertex_footprint_indexing().len())
    //         .map(|flat_index| (flat_index, problem.add_column(0., (0.)..Float::INFINITY)))
    //         .collect::<Vec<_>>();

    //     // Define the principal constraints, which specify that the average of the heights at each
    //     // cell footprint's three vertices be equal to the height at the cell footprint center.
    //     for cell_footprint_index in indexing::iter_indices(grid.cell_footprint_indexing()) {
    //         let center_height = height.center(cell_footprint_index);
    //         problem.add_row(
    //             center_height..center_height,
    //             &cell_footprint_index
    //                 .vertices()
    //                 .into_iter()
    //                 .map(|footprint_vertex_index| {
    //                     (
    //                         vertex_variables[grid
    //                             .vertex_footprint_indexing()
    //                             .flatten(footprint_vertex_index)]
    //                         .1,
    //                         1. / 3.,
    //                     )
    //                 })
    //                 .collect::<Vec<_>>(),
    //         );
    //     }

    //     // Determine how to compute the mesh Laplacian at a single vertex. TODO: Account for mesh
    //     // spacing.
    //     let compute_laplacian =
    //         |vertex_footprint_index: indexing::VertexFootprintIndex| -> Vec<(highs::Col, Float)> {
    //             let neighbors = grid
    //                 .vertex_footprint_indexing()
    //                 .neighbors(vertex_footprint_index);
    //             let mut num_neighbors = 0;
    //             let mut linear_expr = neighbors
    //                 .map(|neighbor_vertex_footprint_index| {
    //                     num_neighbors += 1;
    //                     (
    //                         vertex_variables[grid
    //                             .vertex_footprint_indexing()
    //                             .flatten(neighbor_vertex_footprint_index)]
    //                         .1,
    //                         -1.,
    //                     )
    //                 })
    //                 .collect::<Vec<_>>();
    //             linear_expr.push((
    //                 vertex_variables[grid
    //                     .vertex_footprint_indexing()
    //                     .flatten(vertex_footprint_index)]
    //                 .1,
    //                 num_neighbors as Float,
    //             ));
    //             linear_expr
    //         };

    //     // Add slack variables and associated constraints to effect an absolute value objective
    //     // function on the mesh Laplacian.
    //     for vertex_footprint_index in indexing::iter_indices(grid.vertex_footprint_indexing()) {
    //         let slack_variable = problem.add_column(1., Float::NEG_INFINITY..Float::INFINITY);
    //         let laplacian = compute_laplacian(vertex_footprint_index);

    //         // L + s >= 0  ->  s >= -L
    //         let mut slack_constraint = laplacian.clone();
    //         slack_constraint.push((slack_variable, 1.));
    //         problem.add_row((0.)..Float::INFINITY, slack_constraint);

    //         // L - s <= 0  ->  s >= L
    //         let mut slack_constraint = laplacian;
    //         slack_constraint.push((slack_variable, -1.));
    //         problem.add_row(Float::NEG_INFINITY..0., slack_constraint);
    //     }

    //     // Solve.
    //     let solution = problem.optimise(highs::Sense::Minimise).solve();
    //     if solution.status() != highs::HighsModelStatus::Optimal {
    //         return Err(solution.status());
    //     }
    //     let solution = solution.get_solution();

    //     // Unpack the solution into a lattice. Evenly distribute z points across each column of
    //     // cells.
    //     let num_z_points = grid.vertex_indexing().num_z_points();
    //     let mut lattice = Array3::zeros((
    //         grid.vertex_indexing.num_x_points(),
    //         grid.vertex_indexing.num_y_points(),
    //         num_z_points,
    //     ));
    //     for (vertex_footprint_flat_index, variable) in vertex_variables.iter().enumerate() {
    //         let indexing::VertexFootprintIndex { x: i, y: j } = grid
    //             .vertex_footprint_indexing()
    //             .unflatten(vertex_footprint_flat_index);
    //         let height = solution.columns()[variable.0];
    //         for k in 0..num_z_points {
    //             lattice[[i, j, k]] =
    //                 terrain.vertices[[i, j]] + (k as Float / (num_z_points as Float - 1.)) * height;
    //         }
    //     }
    //     Ok(Self { lattice })
    // }
}

fn mean(iterator: impl Iterator<Item = Float>) -> Option<Float> {
    let (count, sum) = iterator.fold((0, 0.), |acc, value| (acc.0 + 1, acc.1 + value));
    if count > 0 {
        Some(sum / count as Float)
    } else {
        None
    }
}

// pub struct Mesh {
//     pub grid: Grid,
//     pub z_axis: ZLattice,
// }
// impl Mesh {
//     pub fn grid(&self) -> &Grid {
//         &self.grid
//     }
//     pub fn z_axis(&self) -> &ZLattice {
//         &self.z_axis
//     }
// }

// pub struct Fields {
//     pub height: HorizScalarField,
//     pub velocity: VectorField,
//     pub pressure: ScalarField,
// }
// impl Fields {
//     pub fn compute_horiz_velocity_column_int(&self, z_axis: &ZLattice) -> HorizVectorField {
//         (&z_axis.spacing().slice(s![.., .., nd::NewAxis, nd::NewAxis]) * &self.velocity)
//             .sum_axis(nd::Axis(2)) // Column integral.
//             .slice(s![.., .., ..-1]) // Drop z component.
//             .into_owned()
//     }
// }

// struct Solver {
//     grid: Grid,
// }
// impl Solver {
//     pub fn step(
//         &self,
//         dt: Float,
//         z_axis: &ZLattice,
//         fields: &Fields,
//         terrain: &HorizScalarField,
//         rain_rate: &HorizScalarField,
//     ) -> (Fields, ZLattice) {
//         let new_height = perform_height_update(dt, &self.grid, z_axis, fields, rain_rate);

//         let new_z_axis = ZLattice::new(&self.grid, terrain, &fields.height);

//         // Interpolate velocity to new height map.
//         let new_velocity = interpolate_onto(&z_axis.centers, &fields.velocity, &new_z_axis.centers);

//         // Compute new pressure field.
//         let new_pressure = compute_pressure(&self.grid, z_axis, &new_velocity);

//         // Perform velocity update.
//         // TODO

//         (
//             Fields {
//                 height: new_height,
//                 velocity: new_velocity,
//                 pressure: new_pressure,
//             },
//             new_z_axis,
//         )
//     }
// }

// /// Linearly interpolate a velocity vector field array `v` with vertical coordinate array `z` onto a
// /// new vertical coordinate array `new_z`. Use constant extrapolation if a value of `new_z` falls
// /// outside the range of `z`.
// fn interpolate_onto(z: &ScalarField, v: &VectorField, new_z: &ScalarField) -> VectorField {
//     let mut new_v = VectorField::zeros((v.dim().0, v.dim().1, new_z.dim().2, v.dim().3));
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
//                     (1. - alpha) * &vij.slice(s![k - 1, ..]) + alpha * &vij.slice(s![k, ..])
//                 });
//             }
//         }
//     }
//     new_v
// }

// fn perform_height_update(
//     dt: Float,
//     grid: &Grid,
//     z_axis: &ZLattice,
//     fields: &Fields,
//     rain_rate: &HorizScalarField,
// ) -> HorizScalarField {
//     #![allow(non_snake_case)]
//     let U = fields.compute_horiz_velocity_column_int(&z_axis);
//     let U_x = U.slice(s![.., .., 0]);
//     let U_y = U.slice(s![.., .., 1]);
//     let U_x_faces = (&U_x.slice(s![1.., ..]) + &U_x.slice(s![..-1, ..])) / 2.;
//     let U_y_faces = (&U_y.slice(s![.., 1..]) + &U_y.slice(s![.., ..-1])) / 2.;
//     let U_x_diffs: HorizScalarField =
//         (&U_x_faces.slice(s![1.., ..]) - &U_x_faces.slice(s![..-1, ..])) / grid.x_axis.spacing();
//     let U_y_diffs: HorizScalarField =
//         (&U_y_faces.slice(s![.., 1..]) - &U_y_faces.slice(s![.., ..-1])) / grid.y_axis.spacing();

//     let mut new_height = fields.height.clone();
//     let mut new_height_inner = new_height.slice_mut(s![1..-1, 1..-1]);
//     new_height_inner += &(-dt * (U_x_diffs + U_y_diffs));

//     new_height += &(dt * rain_rate);

//     new_height
// }

// fn compute_pressure(grid: &Grid, z_axis: &ZLattice, velocity: &VectorField) -> ScalarField {
//     // let cg_solver = argmin::solver::conjugategradient::ConjugateGradient::new(b)
//     todo!()
// }

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_construct_z_lattice_flat() {
        let grid = Grid::new(Axis::new(0., 1., 29), Axis::new(0., 1., 32), 11);
        let terrain = Terrain::new(nd::Array2::zeros([
            grid.x_axis().vertices().len(),
            grid.y_axis().vertices().len(),
        ]));
        let mut height = Height::zeros(grid.cell_footprint_indexing());
        for cell_footprint_index in indexing::iter_indices(grid.cell_footprint_indexing()) {
            *height.center_mut(cell_footprint_index) = 7.3;
        }

        let z_lattice = ZLattice::compute(&grid, &terrain, &height);

        assert_relative_eq!(
            z_lattice.lattice,
            nd::Array1::linspace(0., 7.3, grid.vertex_indexing().num_z_points())
                .slice(nd::s![nd::NewAxis, nd::NewAxis, ..])
                .broadcast(z_lattice.lattice.dim())
                .unwrap(),
            epsilon = 1e-6,
        );
    }

    #[test]
    fn test_construct_z_lattice_grade() {
        let grid = Grid::new(Axis::new(0., 1., 60), Axis::new(0., 1., 31), 11);
        let terrain = Terrain::new(nd::Array2::zeros([
            grid.x_axis().vertices().len(),
            grid.y_axis().vertices().len(),
        ]));
        let mut height = Height::zeros(grid.cell_footprint_indexing());
        for cell_footprint_index in indexing::iter_indices(grid.cell_footprint_indexing()) {
            let centroid = grid.cell_footprint_centroid(&cell_footprint_index);
            *height.center_mut(cell_footprint_index) = 1. + centroid[0] + 2. * centroid[1];
        }

        let z_lattice = ZLattice::compute(&grid, &terrain, &height);

        let height_expected = &grid.x_axis().vertices().slice(s![.., nd::NewAxis])
            + 2. * &grid.y_axis().vertices().slice(s![nd::NewAxis, ..])
            + 1.;
        let z_lattice_expected = &height_expected.slice(s![.., .., nd::NewAxis])
            * &nd::Array::linspace(0., 1., grid.vertex_indexing().num_z_points()).slice(s![
                nd::NewAxis,
                nd::NewAxis,
                ..
            ]);
        // Ensure the expected array matches within a tolerance.
        assert_relative_eq!(
            z_lattice.lattice.slice(s![.., .., -1]),
            height_expected,
            epsilon = 0.2,
        );
        assert_relative_eq!(z_lattice.lattice, z_lattice_expected, epsilon = 0.2,);
        // Ensure that the "inner" portions of the actual and expected arrays match to a much
        // tighter tolerance.
        assert_relative_eq!(
            z_lattice.lattice.slice(s![1..-1, 1..-1, -1]),
            &height_expected.slice(s![1..-1, 1..-1]),
            epsilon = 1e-10
        );
        assert_relative_eq!(
            z_lattice.lattice.slice(s![1..-1, 1..-1, ..]),
            z_lattice_expected.slice(s![1..-1, 1..-1, ..]),
            epsilon = 1e-10,
        );
    }

    #[test]
    fn test_mean() {
        assert_relative_eq!(mean([0., 5., 10.].into_iter()).unwrap(), 5.);
    }

    // #[test]
    // fn test_interpolate_onto() {
    //     let z = nd::array![[[0., 1., 3.]]];
    //     let v = nd::array![[[10., 20., -20.]]]
    //         .slice(s![.., .., .., nd::NewAxis])
    //         .into_owned();

    //     for val in [-100., -1., 0., 1e-15] {
    //         let z_new = nd::array![[[val]]];
    //         let y_new = interpolate_onto(&z, &v, &z_new);
    //         assert_eq!(y_new.dim(), (1, 1, 1, 1));
    //         assert_relative_eq!(
    //             y_new,
    //             nd::array![[[10.]]].slice(s![.., .., .., nd::NewAxis]),
    //         );
    //     }
    //     for val in [3. - 1e-15, 3., 4., 1e4] {
    //         let z_new = nd::array![[[val]]];
    //         let y_new = interpolate_onto(&z, &v, &z_new);
    //         assert_eq!(y_new.dim(), (1, 1, 1, 1));
    //         assert_relative_eq!(
    //             y_new,
    //             nd::array![[[-20.]]].slice(s![.., .., .., nd::NewAxis]),
    //         );
    //     }
    //     {
    //         let z_new = nd::array![[[-1., 0., 0.1, 0.5, 1. - 1e-15, 1., 1. + 1e-15, 2., 3.1]]];
    //         let y_new = interpolate_onto(&z, &v, &z_new);
    //         assert_eq!(y_new.dim(), (1, 1, 9, 1));
    //         assert_relative_eq!(
    //             y_new,
    //             nd::array![[[10., 10., 11., 15., 20., 20., 20., 0., -20.]]].slice(s![
    //                 ..,
    //                 ..,
    //                 ..,
    //                 nd::NewAxis
    //             ]),
    //         );
    //     }
    // }
}
