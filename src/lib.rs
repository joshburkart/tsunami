#![allow(dead_code)]

use ndarray::{self as nd, s};
use strum::EnumCount;

pub type Float = f64;

pub type Array1 = nd::Array1<Float>;
pub type Array2 = nd::Array2<Float>;
pub type Array3 = nd::Array3<Float>;

pub type CoordArray = nd::Array1<Float>;

pub type VectorField = nd::Array4<Float>;

pub type HorizVectorField = nd::Array3<Float>;

trait Indexing {
    type Index: Copy;

    fn len(&self) -> usize;

    fn flatten(&self, index: Self::Index) -> usize;
    fn unflatten(&self, flat_index: usize) -> Self::Index;
}

fn iter_indices<I: Indexing>(indexing: &I) -> impl Iterator<Item = I::Index> + '_ {
    (0..indexing.len()).map(|flat_index| indexing.unflatten(flat_index))
}

#[derive(Clone, Copy)]
pub struct VertexFootprintIndexing {
    num_x_points: usize,
    num_y_points: usize,
}
impl VertexFootprintIndexing {
    pub fn neighbors(
        &self,
        center: VertexFootprintIndex,
    ) -> impl Iterator<Item = VertexFootprintIndex> {
        let min_x = center.x.checked_sub(1).unwrap_or(1);
        let max_x = (center.x + 2).min(self.num_x_points - 1);
        let min_y = center.y.checked_sub(1).unwrap_or(1);
        let max_y = (center.y + 2).min(self.num_y_points - 1);

        (min_x..max_x)
            .step_by(2)
            .map(move |x| VertexFootprintIndex { x, y: center.y })
            .chain(
                (min_y..max_y)
                    .step_by(2)
                    .map(move |y| VertexFootprintIndex { x: center.x, y }),
            )
    }
}
#[derive(Copy, Clone)]
pub struct VertexFootprintIndex {
    pub x: usize,
    pub y: usize,
}
impl Indexing for VertexFootprintIndexing {
    type Index = VertexFootprintIndex;

    fn len(&self) -> usize {
        self.num_x_points * self.num_y_points
    }

    fn flatten(&self, index: Self::Index) -> usize {
        self.num_y_points * index.x + index.y
    }
    fn unflatten(&self, flat_index: usize) -> Self::Index {
        VertexFootprintIndex {
            x: flat_index / self.num_y_points,
            y: flat_index % self.num_y_points,
        }
    }
}

#[derive(Clone, Copy)]
pub struct VertexIndexing {
    pub cell_indexing: CellIndexing,
}
impl VertexIndexing {
    pub fn num_x_points(&self) -> usize {
        self.cell_indexing
            .cell_footprint_indexing
            .vertex_footprint_indexing
            .num_x_points
    }
    pub fn num_y_points(&self) -> usize {
        self.cell_indexing
            .cell_footprint_indexing
            .vertex_footprint_indexing
            .num_y_points
    }
    pub fn num_z_points(&self) -> usize {
        self.cell_indexing.num_z_cells_per_column + 1
    }
}
#[derive(Copy, Clone)]
pub struct VertexIndex {
    pub footprint: VertexFootprintIndex,
    pub z: usize,
}
impl Indexing for VertexIndexing {
    type Index = VertexIndex;

    fn len(&self) -> usize {
        self.cell_indexing
            .cell_footprint_indexing
            .vertex_footprint_indexing
            .num_x_points
            * self
                .cell_indexing
                .cell_footprint_indexing
                .vertex_footprint_indexing
                .num_y_points
            * self.num_z_points()
    }

    fn flatten(&self, index: Self::Index) -> usize {
        self.cell_indexing
            .cell_footprint_indexing
            .vertex_footprint_indexing
            .flatten(index.footprint)
            * self.num_z_points()
            + index.z
    }
    fn unflatten(&self, flat_index: usize) -> Self::Index {
        VertexIndex {
            footprint: self
                .cell_indexing
                .cell_footprint_indexing
                .vertex_footprint_indexing
                .unflatten(flat_index / self.num_z_points()),
            z: flat_index % self.num_z_points(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct CellFootprintIndexing {
    pub vertex_footprint_indexing: VertexFootprintIndexing,
}
#[derive(Copy, Clone)]
pub struct CellFootprintIndex {
    pub x: usize,
    pub y: usize,
    pub triangle: Triangle,
}
impl Indexing for CellFootprintIndexing {
    type Index = CellFootprintIndex;

    fn len(&self) -> usize {
        (self.vertex_footprint_indexing.num_x_points - 1)
            * (self.vertex_footprint_indexing.num_y_points - 1)
            * Triangle::COUNT
    }

    fn flatten(&self, index: Self::Index) -> usize {
        (self.vertex_footprint_indexing.num_y_points - 1) * Triangle::COUNT * index.x
            + Triangle::COUNT * index.y
            + index.triangle as usize
    }
    fn unflatten(&self, flat_index: usize) -> Self::Index {
        let triangle = match flat_index % Triangle::COUNT {
            x if x == Triangle::LowerRight as usize => Triangle::LowerRight,
            x if x == Triangle::UpperLeft as usize => Triangle::UpperLeft,
            _ => unreachable!(),
        };
        let y = (flat_index / Triangle::COUNT) % (self.vertex_footprint_indexing.num_y_points - 1);
        let x = flat_index / Triangle::COUNT / (self.vertex_footprint_indexing.num_y_points - 1);
        CellFootprintIndex { x, y, triangle }
    }
}

#[derive(Clone, Copy)]
pub struct CellIndexing {
    pub cell_footprint_indexing: CellFootprintIndexing,
    pub num_z_cells_per_column: usize,
}
#[derive(Copy, Clone)]
pub struct CellIndex {
    pub footprint: CellFootprintIndex,
    pub z: usize,
}
impl Indexing for CellIndexing {
    type Index = CellIndex;

    fn len(&self) -> usize {
        self.cell_footprint_indexing.len() * self.num_z_cells_per_column
    }

    fn flatten(&self, index: Self::Index) -> usize {
        self.cell_footprint_indexing.flatten(index.footprint) * self.num_z_cells_per_column
            + index.z
    }
    fn unflatten(&self, flat_index: usize) -> Self::Index {
        CellIndex {
            footprint: self
                .cell_footprint_indexing
                .unflatten(flat_index / self.num_z_cells_per_column),
            z: flat_index % self.num_z_cells_per_column,
        }
    }
}

pub struct Grid {
    x_axis: Axis,
    y_axis: Axis,

    vertex_indexing: VertexIndexing,
}
impl Grid {
    pub fn new(x_axis: Axis, y_axis: Axis, num_z_points: usize) -> Self {
        let vertex_footprint_indexing = VertexFootprintIndexing {
            num_x_points: x_axis.vertices().len(),
            num_y_points: y_axis.vertices().len(),
        };
        let cell_footprint_indexing = CellFootprintIndexing {
            vertex_footprint_indexing,
        };
        let cell_indexing = CellIndexing {
            cell_footprint_indexing,
            num_z_cells_per_column: num_z_points - 1,
        };
        let vertex_indexing = VertexIndexing { cell_indexing };
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

    pub fn vertex_indexing(&self) -> &VertexIndexing {
        &self.vertex_indexing
    }
    pub fn cell_indexing(&self) -> &CellIndexing {
        &self.vertex_indexing.cell_indexing
    }
    pub fn vertex_footprint_indexing(&self) -> &VertexFootprintIndexing {
        &self
            .vertex_indexing
            .cell_indexing
            .cell_footprint_indexing
            .vertex_footprint_indexing
    }
    pub fn cell_footprint_indexing(&self) -> &CellFootprintIndexing {
        &self.vertex_indexing.cell_indexing.cell_footprint_indexing
    }

    pub fn vertex_footprint_indices(
        &self,
        cell_footprint_index: CellFootprintIndex,
    ) -> [VertexFootprintIndex; 3] {
        match cell_footprint_index.triangle {
            Triangle::UpperLeft => [
                VertexFootprintIndex {
                    x: cell_footprint_index.x,
                    y: cell_footprint_index.y,
                },
                VertexFootprintIndex {
                    x: cell_footprint_index.x + 1,
                    y: cell_footprint_index.y + 1,
                },
                VertexFootprintIndex {
                    x: cell_footprint_index.x,
                    y: cell_footprint_index.y + 1,
                },
            ],
            Triangle::LowerRight => [
                VertexFootprintIndex {
                    x: cell_footprint_index.x,
                    y: cell_footprint_index.y,
                },
                VertexFootprintIndex {
                    x: cell_footprint_index.x + 1,
                    y: cell_footprint_index.y,
                },
                VertexFootprintIndex {
                    x: cell_footprint_index.x + 1,
                    y: cell_footprint_index.y + 1,
                },
            ],
        }
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

#[derive(Copy, Clone, strum_macros::EnumIter, strum_macros::EnumCount)]
pub enum Triangle {
    UpperLeft = 0,
    LowerRight = 1,
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

pub struct Height {
    /// Axes:
    ///     0: x index
    ///     1: y index
    ///     2: Which triangle (UL/LR)
    centers: nd::Array3<Float>,
}
impl Height {
    pub fn zeros(cell_footprint_indexing: &CellFootprintIndexing) -> Self {
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
    pub fn center(&self, cell_footprint_index: CellFootprintIndex) -> Float {
        self.centers[[
            cell_footprint_index.x,
            cell_footprint_index.y,
            cell_footprint_index.triangle as usize,
        ]]
    }
    pub fn center_mut(&mut self, cell_footprint_index: CellFootprintIndex) -> &mut Float {
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
                centers[[i, j, Triangle::UpperLeft as usize]] =
                    (vertices[[i, j]] + vertices[[i + 1, j + 1]] + vertices[[i, j + 1]]) / 3.;
                centers[[i, j, Triangle::LowerRight as usize]] =
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
    pub fn compute(grid: &Grid, terrain: &Terrain, height: &Height) -> Result<Self, minilp::Error> {
        let mut problem = minilp::Problem::new(minilp::OptimizationDirection::Minimize);

        // Define variables to be solved for.
        let vertex_variables = (0..grid.vertex_footprint_indexing().len())
            .map(|_| problem.add_var(0., (0., Float::INFINITY)))
            .collect::<Vec<_>>();

        // Define the principal constraints, which specify that the average of the heights at each
        // cell footprint's three vertices be equal to the height at the cell footprint center.
        for cell_footprint_index in iter_indices(grid.cell_footprint_indexing()) {
            problem.add_constraint(
                grid.vertex_footprint_indices(cell_footprint_index)
                    .into_iter()
                    .map(|footprint_vertex_index| {
                        (
                            vertex_variables[grid
                                .vertex_footprint_indexing()
                                .flatten(footprint_vertex_index)],
                            1. / 3.,
                        )
                    })
                    .collect::<minilp::LinearExpr>(),
                minilp::ComparisonOp::Eq,
                height.center(cell_footprint_index),
            );
        }

        // Determine how to compute the mesh Laplacian at a single vertex. TODO: Account for mesh
        // spacing.
        let compute_laplacian =
            |vertex_footprint_index: VertexFootprintIndex| -> minilp::LinearExpr {
                let neighbors = grid
                    .vertex_footprint_indexing()
                    .neighbors(vertex_footprint_index);
                let mut num_neighbors = 0;
                let mut linear_expr: minilp::LinearExpr = neighbors
                    .map(|neighbor_vertex_footprint_index| {
                        num_neighbors += 1;
                        (
                            vertex_variables[grid
                                .vertex_footprint_indexing()
                                .flatten(neighbor_vertex_footprint_index)],
                            -1.,
                        )
                    })
                    .collect();
                linear_expr.add(
                    vertex_variables[grid
                        .vertex_footprint_indexing()
                        .flatten(vertex_footprint_index)],
                    num_neighbors as Float,
                );
                linear_expr
            };

        // Add slack variables and associated constraints to effect an absolute value objective
        // function on the mesh Laplacian.
        for vertex_footprint_index in iter_indices(grid.vertex_footprint_indexing()) {
            let slack_variable = problem.add_var(1., (Float::NEG_INFINITY, Float::INFINITY));
            let laplacian = compute_laplacian(vertex_footprint_index);

            // L + s >= 0  ->  s >= -L
            let mut slack_constraint = laplacian.clone();
            slack_constraint.add(slack_variable, 1.);
            problem.add_constraint(slack_constraint, minilp::ComparisonOp::Ge, 0.);

            // L - s <= 0  ->  s >= L
            let mut slack_constraint = laplacian;
            slack_constraint.add(slack_variable, -1.);
            problem.add_constraint(slack_constraint, minilp::ComparisonOp::Le, 0.);
        }

        // Solve.
        let solution = problem.solve()?;

        // Unpack the solution into a lattice. Evenly distribute z points across each column of
        // cells.
        let num_z_points = grid.vertex_indexing().num_z_points();
        let mut lattice = Array3::zeros((
            grid.vertex_indexing.num_x_points(),
            grid.vertex_indexing.num_y_points(),
            num_z_points,
        ));
        for (vertex_footprint_flat_index, variable) in vertex_variables.iter().enumerate() {
            let VertexFootprintIndex { x: i, y: j } = grid
                .vertex_footprint_indexing()
                .unflatten(vertex_footprint_flat_index);
            let height = *solution.var_value(*variable);
            for k in 0..num_z_points {
                lattice[[i, j, k]] =
                    terrain.vertices[[i, j]] + (k as Float / (num_z_points as Float - 1.)) * height;
            }
        }
        Ok(Self { lattice })
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
        for cell_footprint_index in iter_indices(grid.cell_footprint_indexing()) {
            *height.center_mut(cell_footprint_index) = 7.3;
        }

        let z_lattice = ZLattice::compute(&grid, &terrain, &height).unwrap();

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
    fn test_indexing() {
        let vertex_footprint_indexing = VertexFootprintIndexing {
            num_x_points: 5,
            num_y_points: 27,
        };
        test_indexing_impl(&vertex_footprint_indexing);

        let cell_footprint_indexing = CellFootprintIndexing {
            vertex_footprint_indexing,
        };
        test_indexing_impl(&cell_footprint_indexing);

        let cell_indexing = CellIndexing {
            cell_footprint_indexing,
            num_z_cells_per_column: 19,
        };
        test_indexing_impl(&cell_indexing);

        let vertex_indexing = VertexIndexing { cell_indexing };
        test_indexing_impl(&vertex_indexing);
    }

    fn test_indexing_impl<I: Indexing>(indexing: &I) {
        for (expected_flat_index, index) in iter_indices(indexing).enumerate() {
            let flat_index = indexing.flatten(index);
            assert_eq!(expected_flat_index, flat_index);
        }
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
