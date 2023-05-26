use nalgebra as na;
use ndarray::{self as nd, s};

use crate::{indexing, Array2, Array3, Float, UnitVector3, Vector3};

pub type CoordArray = nd::Array1<Float>;

pub struct Grid {
    x_axis: Axis,
    y_axis: Axis,

    down_right_anti_diagonal: UnitVector3,

    vertex_indexing: indexing::VertexIndexing,
}
impl Grid {
    pub fn new(x_axis: Axis, y_axis: Axis, num_z_points: usize) -> Self {
        let vertex_footprint_indexing = indexing::VertexFootprintIndexing::new(
            x_axis.vertices().len(),
            y_axis.vertices().len(),
        );
        let cell_footprint_indexing =
            indexing::CellFootprintIndexing::new(vertex_footprint_indexing);
        let cell_indexing = indexing::CellIndexing::new(cell_footprint_indexing, num_z_points - 1);
        let vertex_indexing = indexing::VertexIndexing::new(cell_indexing);
        let down_right_anti_diagonal =
            UnitVector3::new_normalize(Vector3::new(y_axis.spacing(), -x_axis.spacing(), 0.));
        Self {
            x_axis,
            y_axis,
            vertex_indexing,
            down_right_anti_diagonal,
        }
    }

    pub fn x_axis(&self) -> &Axis {
        &self.x_axis
    }
    pub fn y_axis(&self) -> &Axis {
        &self.y_axis
    }

    pub fn down_right_anti_diagonal(&self) -> UnitVector3 {
        self.down_right_anti_diagonal
    }

    pub fn vertex_indexing(&self) -> &indexing::VertexIndexing {
        &self.vertex_indexing
    }
    pub fn cell_indexing(&self) -> &indexing::CellIndexing {
        self.vertex_indexing.cell_indexing()
    }
    pub fn vertex_footprint_indexing(&self) -> &indexing::VertexFootprintIndexing {
        self.vertex_indexing()
            .cell_indexing()
            .cell_footprint_indexing()
            .vertex_footprint_indexing()
    }
    pub fn cell_footprint_indexing(&self) -> &indexing::CellFootprintIndexing {
        self.vertex_indexing()
            .cell_indexing()
            .cell_footprint_indexing()
    }

    pub fn footprint_area(&self) -> Float {
        self.x_axis.spacing * self.y_axis.spacing / 2.
    }

    fn make_terrain<F: Fn(Float, Float) -> Float>(&self, f: F) -> Terrain {
        use indexing::Index;
        use indexing::Indexing;

        let mut centers = Array3::zeros(self.cell_footprint_indexing().shape());
        let mut vertices = Array2::zeros(self.vertex_footprint_indexing().shape());
        for cell_footprint_index in indexing::iter_indices(self.cell_footprint_indexing()) {
            let centroid = self.compute_cell_footprint_centroid(cell_footprint_index);
            centers[cell_footprint_index.to_array_index()] = f(centroid[0], centroid[1]);
        }
        for vertex_footprint_index in indexing::iter_indices(self.vertex_footprint_indexing()) {
            vertices[vertex_footprint_index.to_array_index()] = f(
                self.x_axis.vertices()[vertex_footprint_index.x],
                self.y_axis.vertices()[vertex_footprint_index.y],
            );
        }
        Terrain { centers, vertices }
    }

    pub fn compute_cell_footprint_centroid(
        &self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> [Float; 2] {
        let indexing::CellFootprintIndex {
            x: i,
            y: j,
            triangle,
        } = cell_footprint_index;
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
}

pub struct StaticGeometry {
    grid: Grid,
    terrain: Terrain,
}
impl StaticGeometry {
    pub fn new<F: Fn(Float, Float) -> Float>(grid: Grid, terrain_maker: F) -> Self {
        let terrain = grid.make_terrain(terrain_maker);
        Self { grid, terrain }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }
    pub fn terrain(&self) -> &Terrain {
        &self.terrain
    }
}

/// A fixed $x$ or $y$ axis
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

/// A lattice of z coordinate points
///
/// Each column of z points starts from the terrain and ends at the top of the height field.
struct ZLattice {
    /// Indexed by [`indexing::VertexIndexing`]
    lattice: Array3,
}
impl ZLattice {
    pub fn new(grid: &Grid, terrain: &Terrain, height: &HorizScalarField) -> Self {
        use indexing::{Index, Indexing};

        let vertex_indexing = grid.vertex_indexing();
        let num_z_points = vertex_indexing.num_z_points();
        let mut lattice = Array3::zeros(vertex_indexing.shape());
        let vertex_footprint_indexing = grid.vertex_footprint_indexing();
        for vertex_footprint_index in indexing::iter_indices(vertex_footprint_indexing) {
            let terrain_height = terrain.vertices[vertex_footprint_index.to_array_index()];
            let height = mean(
                vertex_footprint_index
                    .adjacent_cells(vertex_footprint_indexing)
                    .map(|cell_footprint_index| height.center(cell_footprint_index)),
            )
            .unwrap();
            for vertex_index in vertex_indexing.column(vertex_footprint_index) {
                lattice[vertex_index.to_array_index()] = terrain_height
                    + (vertex_index.z as Float / (num_z_points as Float - 1.)) * height
            }
        }
        Self { lattice }
    }
}

pub struct DynamicGeometry<'a> {
    static_geometry: &'a StaticGeometry,
    z_lattice: ZLattice,
}
impl<'a> DynamicGeometry<'a> {
    pub fn new(static_geometry: &'a StaticGeometry, height: &HorizScalarField) -> Self {
        assert_eq!(
            [
                static_geometry.grid().x_axis.centers.len(),
                static_geometry.grid().y_axis.centers.len(),
                2
            ],
            height.centers.shape()
        );
        let z_lattice = ZLattice::new(static_geometry.grid(), static_geometry.terrain(), &height);
        Self {
            static_geometry,
            z_lattice,
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.static_geometry.grid
    }
    pub fn terrain(&self) -> &Terrain {
        &self.static_geometry.terrain
    }

    pub fn compute_vertical_face(
        &self,
        cell_index: indexing::CellIndex,
        cell_footprint_edge: indexing::CellFootprintEdge,
    ) -> CellVerticalFace {
        use indexing::Index;

        // --------
        // |     /|
        // |    / |
        // |   /  |
        // |  /   |
        // | /    |
        // |/     |
        // --------
        let indexing::CellIndex { z, .. } = cell_index;
        let indexing::CellFootprintEdge {
            x,
            y,
            triangle_edge,
            ..
        } = cell_footprint_edge;
        let (vertex_footprint_1, vertex_footprint_2, horiz_length, outward_normal) = {
            let dx = self.static_geometry.grid.x_axis.spacing;
            let dy = self.static_geometry.grid.y_axis.spacing;
            let ddiag = (dx.powi(2) + dy.powi(2)).sqrt();

            let down_right = self.static_geometry.grid.down_right_anti_diagonal();

            match triangle_edge {
                indexing::TriangleEdge::UpperLeft(upper_left_edge) => match upper_left_edge {
                    indexing::UpperLeftTriangleEdge::Up => (
                        indexing::VertexFootprintIndex { x, y: y + 1 },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        dx,
                        na::Vector3::y_axis(),
                    ),
                    indexing::UpperLeftTriangleEdge::Left => (
                        indexing::VertexFootprintIndex { x, y },
                        indexing::VertexFootprintIndex { x: x + 1, y },
                        dy,
                        -na::Vector3::x_axis(),
                    ),
                    indexing::UpperLeftTriangleEdge::DownRight => (
                        indexing::VertexFootprintIndex { x, y },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        ddiag,
                        down_right,
                    ),
                },
                indexing::TriangleEdge::LowerRight(lower_right_edge) => match lower_right_edge {
                    indexing::LowerRightTriangleEdge::Down => (
                        indexing::VertexFootprintIndex { x, y },
                        indexing::VertexFootprintIndex { x: x + 1, y },
                        dx,
                        -na::Vector3::y_axis(),
                    ),
                    indexing::LowerRightTriangleEdge::Right => (
                        indexing::VertexFootprintIndex { x: x + 1, y },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        dy,
                        na::Vector3::x_axis(),
                    ),
                    indexing::LowerRightTriangleEdge::UpLeft => (
                        indexing::VertexFootprintIndex { x, y },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        ddiag,
                        -down_right,
                    ),
                },
            }
        };
        let area = {
            let vert_length_1 = self.z_lattice.lattice[indexing::VertexIndex {
                footprint: vertex_footprint_1,
                z: z + 1,
            }
            .to_array_index()]
                - self.z_lattice.lattice[indexing::VertexIndex {
                    footprint: vertex_footprint_1,
                    z,
                }
                .to_array_index()];
            let vert_length_2 = self.z_lattice.lattice[indexing::VertexIndex {
                footprint: vertex_footprint_2,
                z: z + 1,
            }
            .to_array_index()]
                - self.z_lattice.lattice[indexing::VertexIndex {
                    footprint: vertex_footprint_2,
                    z,
                }
                .to_array_index()];
            0.5 * horiz_length * (vert_length_1 + vert_length_2)
        };

        CellVerticalFace {
            area,
            outward_normal,
            cell_index,
            neighbor: cell_footprint_edge.neighbor,
        }
    }
}

pub struct CellVerticalFace {
    pub cell_index: indexing::CellIndex,
    pub area: Float,
    pub outward_normal: UnitVector3,
    pub neighbor: indexing::CellFootprintNeighbor,
}

/// A scalar field
///
/// Defines the values of the field at cell centers.
pub struct ScalarField {
    /// Indexed by [`indexing::CellIndexing`]
    centers: nd::Array4<Float>,
}
impl ScalarField {
    pub fn zeros(grid: &Grid) -> Self {
        use indexing::Indexing;

        Self {
            centers: nd::Array4::default(grid.cell_indexing().shape()),
        }
    }

    pub fn center(&self, index: indexing::CellIndex) -> Float {
        use indexing::Index;

        self.centers[index.to_array_index()]
    }
}

/// A vector field
///
/// Defines the values of the field at cell centers.
pub struct VectorField {
    /// Indexed by [`indexing::CellIndexing`]
    centers: nd::Array4<Vector3>,
}
impl VectorField {
    pub fn zeros(grid: &Grid) -> Self {
        use indexing::Indexing;

        Self {
            centers: nd::Array4::default(grid.cell_indexing().shape()),
        }
    }

    pub fn center(&self, index: indexing::CellIndex) -> Vector3 {
        use indexing::Index;

        self.centers[index.to_array_index()]
    }

    pub fn interpolate_to_face(&self, vertical_face: &CellVerticalFace) -> Vector3 {
        match vertical_face.neighbor {
            indexing::CellFootprintNeighbor::CellFootprint(cell_footprint_index) => {
                let neighbor_cell_index = indexing::CellIndex {
                    footprint: cell_footprint_index,
                    z: vertical_face.cell_index.z,
                };
                0.5 * (self.center(vertical_face.cell_index) + self.center(neighbor_cell_index))
            }
            indexing::CellFootprintNeighbor::Boundary(_) => self.center(vertical_face.cell_index),
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct HorizScalarField {
    /// Indexed by [`indexing::CellFootprintIndexing`]
    centers: Array3,
}
impl HorizScalarField {
    pub fn new(centers: Array3) -> Self {
        Self { centers }
    }
    pub fn zeros(cell_footprint_indexing: &indexing::CellFootprintIndexing) -> Self {
        use indexing::Indexing;

        Self {
            centers: nd::Array3::zeros(cell_footprint_indexing.shape()),
        }
    }

    pub fn center(&self, cell_footprint_index: indexing::CellFootprintIndex) -> Float {
        use indexing::Index;

        self.centers[cell_footprint_index.to_array_index()]
    }
    pub fn center_mut(&mut self, cell_footprint_index: indexing::CellFootprintIndex) -> &mut Float {
        use indexing::Index;

        &mut self.centers[cell_footprint_index.to_array_index()]
    }
}
impl std::ops::Add<Float> for HorizScalarField {
    type Output = Self;

    fn add(mut self, rhs: Float) -> Self::Output {
        self += rhs;
        self
    }
}
impl std::ops::Add<Float> for &HorizScalarField {
    type Output = HorizScalarField;

    fn add(self, rhs: Float) -> Self::Output {
        let mut new = self.clone();
        new += rhs;
        new
    }
}
impl std::ops::AddAssign<Float> for HorizScalarField {
    fn add_assign(&mut self, rhs: Float) {
        self.centers += rhs;
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for HorizScalarField {
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.centers.abs_diff_eq(&other.centers, epsilon)
    }
}

pub struct Terrain {
    /// Indexed by [`indexing::CellFootprintIndexing`]
    centers: Array3,
    /// Indexed by [`indexing::VertexFootprintIndexing`]
    vertices: Array2,
}

/// Compute the mean of an iterator over numbers
fn mean(iterator: impl Iterator<Item = Float>) -> Option<Float> {
    let (count, sum) = iterator.fold((0, 0.), |acc, value| (acc.0 + 1, acc.1 + value));
    if count > 0 {
        Some(sum / count as Float)
    } else {
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn test_construct_z_lattice_flat() {
        let grid = Grid::new(Axis::new(0., 1., 29), Axis::new(0., 1., 32), 11);
        let terrain = grid.make_terrain(|_, _| 0.);
        let mut height = HorizScalarField::zeros(grid.cell_footprint_indexing());
        for cell_footprint_index in indexing::iter_indices(grid.cell_footprint_indexing()) {
            *height.center_mut(cell_footprint_index) = 7.3;
        }

        let z_lattice = ZLattice::new(&grid, &terrain, &height);

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
        let terrain = grid.make_terrain(|_, _| 0.);
        let mut height = HorizScalarField::zeros(grid.cell_footprint_indexing());
        for cell_footprint_index in indexing::iter_indices(grid.cell_footprint_indexing()) {
            let centroid = grid.compute_cell_footprint_centroid(cell_footprint_index);
            *height.center_mut(cell_footprint_index) = 1. + centroid[0] + 2. * centroid[1];
        }

        let z_lattice = ZLattice::new(&grid, &terrain, &height);

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
