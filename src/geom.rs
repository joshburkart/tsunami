use nalgebra as na;
use ndarray::{self as nd, s};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::{indexing, Array1, Array2, Array3, Float, UnitVector2, UnitVector3, Vector2, Vector3};

#[pyclass]
#[derive(Clone)]
pub struct Grid {
    x_axis: Axis,
    y_axis: Axis,

    down_right_anti_diagonal: UnitVector2,

    vertex_indexing: indexing::VertexIndexing,
}
impl Grid {
    pub fn new(x_axis: Axis, y_axis: Axis, num_z_cells: usize) -> Self {
        let vertex_footprint_indexing = indexing::VertexFootprintIndexing::new(
            x_axis.vertices().len(),
            y_axis.vertices().len(),
        );
        let cell_footprint_indexing =
            indexing::CellFootprintIndexing::new(vertex_footprint_indexing);
        let cell_indexing = indexing::CellIndexing::new(cell_footprint_indexing, num_z_cells);
        let vertex_indexing = indexing::VertexIndexing::new(cell_indexing);
        let down_right_anti_diagonal =
            UnitVector2::new_normalize(Vector2::new(y_axis.spacing(), -x_axis.spacing()));
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

    pub fn down_right_anti_diagonal(&self) -> UnitVector2 {
        self.down_right_anti_diagonal
    }

    pub fn footprint_area(&self) -> Float {
        self.x_axis.spacing * self.y_axis.spacing / 2.
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

    pub fn make_cell_footprint_array<F: Fn(Float, Float) -> Float>(&self, f: F) -> Array3 {
        use indexing::{Index, Indexing};

        let mut centers = Array3::zeros(self.cell_footprint_indexing().shape());
        for cell_footprint_index in indexing::iter_indices(self.cell_footprint_indexing()) {
            let centroid = self.compute_cell_footprint_centroid(cell_footprint_index);
            centers[cell_footprint_index.to_array_index()] = f(centroid[0], centroid[1]);
        }
        centers
    }

    pub fn make_vertex_footprint_array<F: Fn(Float, Float) -> Float>(&self, f: F) -> Array2 {
        use indexing::{Index, Indexing};

        let mut vertices = Array2::zeros(self.vertex_footprint_indexing().shape());
        for vertex_footprint_index in indexing::iter_indices(self.vertex_footprint_indexing()) {
            vertices[vertex_footprint_index.to_array_index()] = f(
                self.x_axis.vertices()[vertex_footprint_index.x],
                self.y_axis.vertices()[vertex_footprint_index.y],
            );
        }
        vertices
    }

    pub fn compute_cell_footprint_edge(
        &self,
        cell_footprint_pair: indexing::CellFootprintPair,
    ) -> CellFootprintEdge {
        // --------
        // |     /|
        // |    / |
        // |   /  |
        // |  /   |
        // | /    |
        // |/     |
        // --------
        let indexing::CellFootprintPair {
            x,
            y,
            triangle_edge,
            ..
        } = cell_footprint_pair;
        let (vertex_footprint_1, vertex_footprint_2, length, outward_normal) = {
            let dx = self.x_axis.spacing;
            let dy = self.y_axis.spacing;
            let ddiag = (dx.powi(2) + dy.powi(2)).sqrt();

            let down_right = self.down_right_anti_diagonal();

            match triangle_edge {
                indexing::TriangleEdge::UpperLeft(upper_left_edge) => match upper_left_edge {
                    indexing::UpperLeftTriangleEdge::Up => (
                        indexing::VertexFootprintIndex { x, y: y + 1 },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        dx,
                        Vector2::y_axis(),
                    ),
                    indexing::UpperLeftTriangleEdge::Left => (
                        indexing::VertexFootprintIndex { x, y },
                        indexing::VertexFootprintIndex { x, y: y + 1 },
                        dy,
                        -Vector2::x_axis(),
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
                        -Vector2::y_axis(),
                    ),
                    indexing::LowerRightTriangleEdge::Right => (
                        indexing::VertexFootprintIndex { x: x + 1, y },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        dy,
                        Vector2::x_axis(),
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

        CellFootprintEdge {
            outward_normal,
            length,
            cell_footprint_pair,
            vertex_footprint_1,
            vertex_footprint_2,
        }
    }

    fn make_terrain<F: Fn(Float, Float) -> Float>(&self, f: &F) -> Terrain {
        Terrain {
            centers: self.make_cell_footprint_array(f),
            vertices: self.make_vertex_footprint_array(f),
        }
    }

    fn compute_cell_footprint_centroid(
        &self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> na::Point2<Float> {
        let indexing::CellFootprintIndex { x, y, triangle } = cell_footprint_index;
        match triangle {
            indexing::Triangle::UpperLeft => na::Point2::new(
                (2. / 3.) * self.x_axis.vertices[[x]] + (1. / 3.) * self.x_axis.vertices[[x + 1]],
                (2. / 3.) * self.y_axis.vertices[[y]] + (1. / 3.) * self.y_axis.vertices[[y + 1]],
            ),
            indexing::Triangle::LowerRight => na::Point2::new(
                (1. / 3.) * self.x_axis.vertices[[x]] + (2. / 3.) * self.x_axis.vertices[[x + 1]],
                (1. / 3.) * self.y_axis.vertices[[y]] + (2. / 3.) * self.y_axis.vertices[[y + 1]],
            ),
        }
    }
}
#[pymethods]
impl Grid {
    #[getter]
    #[pyo3(name = "x_axis")]
    pub fn x_axis_py(&self) -> Axis {
        self.x_axis.clone()
    }

    #[getter]
    #[pyo3(name = "y_axis")]
    pub fn y_axis_py(&self) -> Axis {
        self.y_axis.clone()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct StaticGeometry {
    grid: Grid,
    terrain: Terrain,
}
impl StaticGeometry {
    pub fn new<F: Fn(Float, Float) -> Float>(grid: Grid, terrain_func: &F) -> Self {
        let terrain = grid.make_terrain(terrain_func);
        Self { grid, terrain }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn terrain(&self) -> &Terrain {
        &self.terrain
    }
}
#[pymethods]
impl StaticGeometry {
    #[pyo3(name = "grid")]
    pub fn grid_py(&self) -> Grid {
        self.grid.clone()
    }

    #[pyo3(name = "terrain")]
    pub fn terrain_py(&self) -> Terrain {
        self.terrain.clone()
    }
}

/// A fixed $x$ or $y$ axis
#[pyclass]
#[derive(Clone)]
pub struct Axis {
    vertices: Array1,
    centers: Array1,
    spacing: Float,
}
impl Axis {
    pub fn new(min: Float, max: Float, num_cells: usize) -> Self {
        let vertices = Array1::linspace(min, max, num_cells + 1);
        let centers = (&vertices.slice(s![..-1]) + &vertices.slice(s![1..])) / 2.;
        let spacing = vertices[1] - vertices[0];
        Self {
            vertices,
            centers,
            spacing,
        }
    }

    pub fn vertices(&self) -> &Array1 {
        &self.vertices
    }

    pub fn centers(&self) -> &Array1 {
        &self.centers
    }

    pub fn spacing(&self) -> Float {
        self.spacing
    }
}
#[pymethods]
impl Axis {
    #[getter]
    #[pyo3(name = "vertices")]
    pub fn vertices_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray1<Float> {
        self.vertices.clone().into_pyarray(py)
    }

    #[getter]
    #[pyo3(name = "centers")]
    pub fn centers_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray1<Float> {
        self.centers.clone().into_pyarray(py)
    }
}

/// A lattice of z coordinate points
///
/// Each column of z points starts from the terrain and ends at the top of the
/// height field.
#[pyclass]
#[derive(Clone)]
pub struct ZLattice {
    /// Indexed by [`indexing::VertexIndexing`]
    lattice: Array3,
    // Indexed by [`indexing::VertexFootprintIndexing`]
    spacings: Array2,
}
impl ZLattice {
    pub fn new(grid: &Grid, terrain: &Terrain, height: &HeightField) -> Self {
        use indexing::{Index, Indexing};

        let vertex_indexing = grid.vertex_indexing();
        let mut lattice = Array3::zeros(vertex_indexing.shape());
        let mut spacings = Array2::zeros(grid.vertex_footprint_indexing().shape());
        let vertex_footprint_indexing = grid.vertex_footprint_indexing();
        for vertex_footprint_index in indexing::iter_indices(vertex_footprint_indexing) {
            let terrain_height = terrain.vertices[vertex_footprint_index.to_array_index()];
            let height = mean(
                vertex_footprint_index
                    .adjacent_cells(vertex_footprint_indexing)
                    .map(|cell_footprint_index| height.center(cell_footprint_index)),
            )
            .unwrap();
            let spacing = height / grid.cell_indexing().num_z_cells() as Float;
            spacings[vertex_footprint_index.to_array_index()] = spacing;
            for vertex_index in vertex_indexing.column(vertex_footprint_index) {
                lattice[vertex_index.to_array_index()] =
                    terrain_height + spacing * vertex_index.z as Float;
            }
        }
        Self { lattice, spacings }
    }

    pub fn z_spacing(&self, vertex_footprint_index: indexing::VertexFootprintIndex) -> Float {
        use indexing::Index;

        self.spacings[vertex_footprint_index.to_array_index()]
    }

    pub fn z_spacing_array(&self) -> &Array2 {
        &self.spacings
    }

    pub fn vertex_value(&self, vertex_index: indexing::VertexIndex) -> Float {
        use indexing::Index;

        self.lattice[vertex_index.to_array_index()]
    }

    pub fn z_lattice<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.lattice.clone().into_pyarray(py)
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DynamicGeometry {
    static_geometry: StaticGeometry,
    z_lattice: ZLattice,
}
impl DynamicGeometry {
    pub fn new(static_geometry: StaticGeometry, height: &HeightField) -> Self {
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

    pub fn z_lattice(&self) -> &ZLattice {
        &self.z_lattice
    }

    pub fn terrain(&self) -> &Terrain {
        &self.static_geometry.terrain
    }

    pub fn into_static_geometry(self) -> StaticGeometry {
        self.static_geometry
    }

    pub fn compute_vertical_face(
        &self,
        cell_index: indexing::CellIndex,
        cell_footprint_pair: indexing::CellFootprintPair,
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
        let cell_footprint_edge = self
            .static_geometry
            .grid()
            .compute_cell_footprint_edge(cell_footprint_pair);
        let area = {
            let vert_length_1 = self.z_lattice.lattice[indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprint_1,
                z: z + 1,
            }
            .to_array_index()]
                - self.z_lattice.lattice[indexing::VertexIndex {
                    footprint: cell_footprint_edge.vertex_footprint_1,
                    z,
                }
                .to_array_index()];
            let vert_length_2 = self.z_lattice.lattice[indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprint_2,
                z: z + 1,
            }
            .to_array_index()]
                - self.z_lattice.lattice[indexing::VertexIndex {
                    footprint: cell_footprint_edge.vertex_footprint_2,
                    z,
                }
                .to_array_index()];
            0.5 * cell_footprint_edge.length * (vert_length_1 + vert_length_2)
        };

        CellVerticalFace {
            area,
            cell_index,
            cell_footprint_edge,
        }
    }

    pub fn make_cell_array<V: Default, F: Fn(Float, Float, Float) -> V>(
        &self,
        f: F,
    ) -> nd::Array4<V> {
        use indexing::{Index, Indexing};

        let mut cells = nd::Array4::<V>::default(self.static_geometry.grid.cell_indexing().shape());
        for cell_index in indexing::iter_indices(self.static_geometry.grid.cell_indexing()) {
            let lattice_index_1 = [cell_index.footprint.x, cell_index.footprint.y, cell_index.z];
            let lattice_index_2 = [
                cell_index.footprint.x,
                cell_index.footprint.y,
                cell_index.z + 1,
            ];
            cells[cell_index.to_array_index()] = f(
                self.static_geometry.grid.x_axis.vertices()[cell_index.footprint.x],
                self.static_geometry.grid.y_axis.vertices()[cell_index.footprint.y],
                0.5 * (self.z_lattice.lattice[lattice_index_1]
                    + self.z_lattice.lattice[lattice_index_2]),
            );
        }
        cells
    }
}
#[pymethods]
impl DynamicGeometry {
    #[pyo3(name = "z_lattice")]
    pub fn z_lattice_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.z_lattice.z_lattice(py)
    }
}

#[derive(Debug)]
pub struct CellFootprintEdge {
    pub cell_footprint_pair: indexing::CellFootprintPair,
    pub outward_normal: UnitVector2,
    pub length: Float,
    pub vertex_footprint_1: indexing::VertexFootprintIndex,
    pub vertex_footprint_2: indexing::VertexFootprintIndex,
}

#[derive(Debug)]
pub struct CellVerticalFace {
    pub cell_index: indexing::CellIndex,
    pub area: Float,
    pub cell_footprint_edge: CellFootprintEdge,
}

#[derive(Clone)]
pub struct VelocityField {
    /// Indexed by [`indexing::CellIndexing`]
    cells: nd::Array4<Vector3>,
}
impl VelocityField {
    pub fn new<F: Fn(Float, Float, Float) -> Vector3>(
        dynamic_geometry: &DynamicGeometry,
        f: F,
    ) -> Self {
        Self {
            cells: dynamic_geometry.make_cell_array(f),
        }
    }

    pub fn cell_value(&self, index: indexing::CellIndex) -> Vector3 {
        use indexing::Index;

        self.cells[index.to_array_index()]
    }

    pub fn column_average(&self) -> nd::Array3<na::Vector2<Float>> {
        self.cells
            .mapv(|velocity| na::Vector2::new(velocity.x, velocity.y))
            .sum_axis(nd::Axis(2))
            / self.cells.shape()[2] as Float
    }

    pub fn interpolate_to_face(&self, vertical_face: &CellVerticalFace) -> Vector3 {
        match vertical_face
            .cell_footprint_edge
            .cell_footprint_pair
            .neighbor
        {
            indexing::CellFootprintNeighbor::CellFootprint(cell_footprint_index) => {
                let neighbor_cell_index = indexing::CellIndex {
                    footprint: cell_footprint_index,
                    z: vertical_face.cell_index.z,
                };
                0.5 * (self.cell_value(vertical_face.cell_index)
                    + self.cell_value(neighbor_cell_index))
            }
            indexing::CellFootprintNeighbor::Boundary(_) => {
                self.cell_value(vertical_face.cell_index)
            }
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PressureField {
    /// Indexed by [`indexing::VertexIndexing`]
    vertices: Array3,
}
impl PressureField {
    pub fn zeros(dynamic_geometry: &DynamicGeometry) -> Self {
        use indexing::Indexing;

        Self {
            vertices: Array3::zeros(dynamic_geometry.grid().vertex_indexing().shape()),
        }
    }

    pub fn vertex_value(&self, vertex: indexing::VertexIndex) -> Float {
        use indexing::Index;

        self.vertices[vertex.to_array_index()]
    }

    pub fn vertex_value_mut(&mut self, vertex: indexing::VertexIndex) -> &mut Float {
        use indexing::Index;

        &mut self.vertices[vertex.to_array_index()]
    }

    // TODO: This is the partial at constant w, not at constant z.
    pub fn forward_deriv_x(
        &self,
        dynamic_geometry: &DynamicGeometry,
        vertex: indexing::VertexIndex,
    ) -> Float {
        let x_spacing = dynamic_geometry.grid().x_axis().spacing();
        (self.vertex_value(vertex.increment_x(1)) - self.vertex_value(vertex)) / x_spacing
    }

    // TODO: This is the partial at constant w, not at constant z.
    pub fn forward_deriv_y(
        &self,
        dynamic_geometry: &DynamicGeometry,
        vertex: indexing::VertexIndex,
    ) -> Float {
        let y_spacing = dynamic_geometry.grid().y_axis().spacing();
        (self.vertex_value(vertex.increment_y(1)) - self.vertex_value(vertex)) / y_spacing
    }

    pub fn forward_deriv_z(
        &self,
        dynamic_geometry: &DynamicGeometry,
        vertex: indexing::VertexIndex,
    ) -> Float {
        let z_spacing = dynamic_geometry.z_lattice().z_spacing(vertex.footprint);
        (self.vertex_value(vertex.increment_z(1)) - self.vertex_value(vertex)) / z_spacing
    }

    pub fn laplacian(&self, dynamic_geometry: &DynamicGeometry) -> Self {
        let x_spacing = dynamic_geometry.grid().x_axis().spacing();
        let y_spacing = dynamic_geometry.grid().y_axis().spacing();

        let w = Array1::linspace(
            0.,
            1.,
            dynamic_geometry.grid().vertex_indexing().num_z_points(),
        );
        let w_spacing = w[1] - w[0];
        // let z_spacing_array = dynamic_geometry.z_lattice().z_spacing_array();

        let partial_xx = (-2. * &self.vertices.slice(s![1..-1, 1..-1, 1..-1])
            + &self.vertices.slice(s![2.., 1..-1, 1..-1])
            + &self.vertices.slice(s![..-2, 1..-1, 1..-1]))
            / x_spacing.powi(2);
        let partial_yy = (-2. * &self.vertices.slice(s![1..-1, 1..-1, 1..-1])
            + &self.vertices.slice(s![1..-1, 2.., 1..-1])
            + &self.vertices.slice(s![1..-1, ..-2, 1..-1]))
            / y_spacing.powi(2);
        let partial_ww = (-2. * &self.vertices.slice(s![1..-1, 1..-1, 1..-1])
            + &self.vertices.slice(s![1..-1, 1..-1, 2..])
            + &self.vertices.slice(s![1..-1, 1..-1, ..-2]))
            / w_spacing.powi(2);

        todo!()
    }
}
impl argmin_math::ArgminDot<PressureField, Float> for PressureField {
    fn dot(&self, other: &PressureField) -> Float {
        (&self.vertices * &other.vertices).sum()
    }
}
impl argmin_math::ArgminSub<PressureField, PressureField> for PressureField {
    fn sub(&self, other: &PressureField) -> PressureField {
        PressureField {
            vertices: &self.vertices - &other.vertices,
        }
    }
}
impl argmin_math::ArgminAdd<PressureField, PressureField> for PressureField {
    fn add(&self, other: &PressureField) -> PressureField {
        PressureField {
            vertices: &self.vertices + &other.vertices,
        }
    }
}
impl argmin_math::ArgminConj for PressureField {
    fn conj(&self) -> Self {
        self.clone()
    }
}
impl argmin_math::ArgminMul<Float, PressureField> for PressureField {
    fn mul(&self, other: &Float) -> PressureField {
        Self {
            vertices: *other * &self.vertices,
        }
    }
}
impl argmin_math::ArgminMul<PressureField, PressureField> for Float {
    fn mul(&self, other: &PressureField) -> PressureField {
        PressureField {
            vertices: *self * &other.vertices,
        }
    }
}
#[pymethods]
impl PressureField {
    #[pyo3(name = "pressure")]
    pub fn pressure_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.vertices.clone().into_pyarray(py)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct HeightField {
    /// Indexed by [`indexing::CellFootprintIndexing`]
    centers: Array3,
}
impl HeightField {
    pub fn new<F: Fn(Float, Float) -> Float>(grid: &Grid, f: F) -> Self {
        Self {
            centers: grid.make_cell_footprint_array(f),
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

    // TODO: Remove otherwise what's the point of this wrapper struct...
    pub fn centers(&self) -> &Array3 {
        &self.centers
    }
}
impl std::ops::Add for HeightField {
    type Output = Self;

    fn add(mut self, rhs: HeightField) -> Self::Output {
        self += rhs;
        self
    }
}
impl std::ops::AddAssign for HeightField {
    fn add_assign(&mut self, rhs: HeightField) {
        self.centers += &rhs.centers;
    }
}
impl std::ops::Add<Float> for HeightField {
    type Output = Self;

    fn add(mut self, rhs: Float) -> Self::Output {
        self += rhs;
        self
    }
}
impl std::ops::Add<Float> for &HeightField {
    type Output = HeightField;

    fn add(self, rhs: Float) -> Self::Output {
        let mut new = self.clone();
        new += rhs;
        new
    }
}
impl std::ops::Add<HeightField> for Float {
    type Output = HeightField;

    fn add(self, mut rhs: HeightField) -> Self::Output {
        rhs += self;
        rhs
    }
}
impl std::ops::Add<&HeightField> for Float {
    type Output = HeightField;

    fn add(self, rhs: &HeightField) -> Self::Output {
        let mut new = rhs.clone();
        new += self;
        new
    }
}
impl std::ops::AddAssign<Float> for HeightField {
    fn add_assign(&mut self, rhs: Float) {
        self.centers += rhs;
    }
}
impl std::ops::Mul<HeightField> for Float {
    type Output = HeightField;

    fn mul(self, rhs: HeightField) -> Self::Output {
        HeightField {
            centers: self * rhs.centers,
        }
    }
}
impl std::ops::Mul<&HeightField> for Float {
    type Output = HeightField;

    fn mul(self, rhs: &HeightField) -> Self::Output {
        HeightField {
            centers: self * &rhs.centers,
        }
    }
}
#[cfg(test)]
impl approx::AbsDiffEq for HeightField {
    type Epsilon = Float;

    fn default_epsilon() -> Self::Epsilon {
        1e-5
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.centers.abs_diff_eq(&other.centers, epsilon)
    }
}

#[pyclass]
#[derive(Clone)]
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
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_construct_z_lattice_zero_height() {
        let grid = Grid::new(Axis::new(0., 1., 29), Axis::new(0., 1., 32), 10);
        let terrain = grid.make_terrain(&|x, y| 10. * x * y);
        let height = HeightField::new(&grid, |_, _| 0.);

        let z_lattice = ZLattice::new(&grid, &terrain, &height);

        let heights = (&z_lattice.lattice.slice(s![.., .., 1..])
            - &z_lattice.lattice.slice(s![.., .., ..-1]))
            .into_owned();
        let expected_heights = 0. * &heights;
        assert_relative_eq!(heights, expected_heights, epsilon = 1e-6,);
    }

    #[test]
    fn test_construct_z_lattice_flat() {
        let grid = Grid::new(Axis::new(0., 1., 29), Axis::new(0., 1., 32), 10);
        let terrain = grid.make_terrain(&|_, _| 0.);
        let mut height = HeightField::new(&grid, |_, _| 0.);
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
        let grid = Grid::new(Axis::new(0., 1., 60), Axis::new(0., 1., 31), 10);
        let terrain = grid.make_terrain(&|_, _| 0.);
        let mut height = HeightField::new(&grid, |_, _| 0.);
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
        // Ensure that the "inner" portions of the actual and expected arrays match to a
        // much tighter tolerance.
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
    //         let z_new = nd::array![[[-1., 0., 0.1, 0.5, 1. - 1e-15, 1., 1. +
    // 1e-15, 2., 3.1]]];         let y_new = interpolate_onto(&z, &v,
    // &z_new);         assert_eq!(y_new.dim(), (1, 1, 9, 1));
    //         assert_relative_eq!(
    //             y_new,
    //             nd::array![[[10., 10., 11., 15., 20., 20., 20., 0.,
    // -20.]]].slice(s![                 ..,
    //                 ..,
    //                 ..,
    //                 nd::NewAxis
    //             ]),
    //         );
    //     }
    // }
}
