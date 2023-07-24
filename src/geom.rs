use nalgebra as na;
use ndarray::{self as nd, s};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::{
    fields,
    indexing::{self, Index, Indexing, IntoIndexIterator},
    Array1, Array2, Array3, Float, Point2, Point3, UnitVector2, UnitVector3, Vector2, Vector3,
};

pub const MIN_VOLUME: Float = 1e-10;

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

    pub fn make_cell_footprint_array<V: fields::Value, F: Fn(Float, Float) -> V>(
        &self,
        f: F,
    ) -> nd::Array3<V> {
        let mut centers = nd::Array3::zeros(self.cell_footprint_indexing().shape());
        for cell_footprint_index in self.cell_footprint_indexing().iter() {
            let centroid = self.compute_cell_footprint_centroid(cell_footprint_index);
            centers[cell_footprint_index.to_array_index()] = f(centroid[0], centroid[1]);
        }
        centers
    }

    pub fn make_vertex_footprint_array<F: Fn(Float, Float) -> Float>(&self, f: F) -> Array2 {
        let mut vertices = Array2::zeros(self.vertex_footprint_indexing().shape());
        for vertex_footprint_index in self.vertex_footprint_indexing().iter() {
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
            footprint: indexing::CellFootprintIndex { x, y, .. },
            triangle_side: triangle_edge,
            ..
        } = cell_footprint_pair;
        let (vertex_footprint_1, vertex_footprint_2, length, outward_normal) = {
            let dx = self.x_axis.spacing;
            let dy = self.y_axis.spacing;
            let ddiag = (dx.powi(2) + dy.powi(2)).sqrt();

            let down_right = self.down_right_anti_diagonal();

            match triangle_edge {
                indexing::TriangleSide::UpperLeft(upper_left_edge) => match upper_left_edge {
                    indexing::UpperLeftTriangleSide::Up => (
                        indexing::VertexFootprintIndex { x, y: y + 1 },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        dx,
                        Vector2::y_axis(),
                    ),
                    indexing::UpperLeftTriangleSide::Left => (
                        indexing::VertexFootprintIndex { x, y },
                        indexing::VertexFootprintIndex { x, y: y + 1 },
                        dy,
                        -Vector2::x_axis(),
                    ),
                    indexing::UpperLeftTriangleSide::DownRight => (
                        indexing::VertexFootprintIndex { x, y },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        ddiag,
                        down_right,
                    ),
                },
                indexing::TriangleSide::LowerRight(lower_right_edge) => match lower_right_edge {
                    indexing::LowerRightTriangleSide::Down => (
                        indexing::VertexFootprintIndex { x, y },
                        indexing::VertexFootprintIndex { x: x + 1, y },
                        dx,
                        -Vector2::y_axis(),
                    ),
                    indexing::LowerRightTriangleSide::Right => (
                        indexing::VertexFootprintIndex { x: x + 1, y },
                        indexing::VertexFootprintIndex { x: x + 1, y: y + 1 },
                        dy,
                        Vector2::x_axis(),
                    ),
                    indexing::LowerRightTriangleSide::UpLeft => (
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
            vertex_footprints: [vertex_footprint_1, vertex_footprint_2],
        }
    }

    fn compute_cell_footprint_centroid(
        &self,
        cell_footprint_index: indexing::CellFootprintIndex,
    ) -> Point2 {
        let indexing::CellFootprintIndex { x, y, triangle } = cell_footprint_index;
        match triangle {
            indexing::Triangle::UpperLeft => na::Point2::new(
                (2. / 3.) * self.x_axis.vertices[[x]] + (1. / 3.) * self.x_axis.vertices[[x + 1]],
                (1. / 3.) * self.y_axis.vertices[[y]] + (2. / 3.) * self.y_axis.vertices[[y + 1]],
            ),
            indexing::Triangle::LowerRight => na::Point2::new(
                (1. / 3.) * self.x_axis.vertices[[x]] + (2. / 3.) * self.x_axis.vertices[[x + 1]],
                (2. / 3.) * self.y_axis.vertices[[y]] + (1. / 3.) * self.y_axis.vertices[[y + 1]],
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
    terrain: fields::Terrain,
}
impl StaticGeometry {
    pub fn new<F: Fn(Float, Float) -> Float>(grid: Grid, terrain_func: F) -> Self {
        let terrain = fields::Terrain::new(&grid, terrain_func);
        Self { grid, terrain }
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    pub fn terrain(&self) -> &fields::Terrain {
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
    pub fn terrain_py(&self) -> fields::Terrain {
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
}
impl ZLattice {
    pub fn new_from_volume(
        grid: &Grid,
        terrain: &fields::Terrain,
        volume: &fields::VolScalarField,
    ) -> Self {
        let vertex_indexing = grid.vertex_indexing();
        let vertex_footprint_indexing = grid.vertex_footprint_indexing();

        let mut lattice = Array3::zeros(vertex_indexing.shape());
        for vertex_footprint_index in vertex_footprint_indexing.iter() {
            let terrain_height = terrain.vertex_value(vertex_footprint_index);
            for vertex_index in vertex_indexing.column(vertex_footprint_index) {
                if vertex_index.z == 0 {
                    lattice[vertex_index.to_array_index()] = terrain_height;
                } else {
                    let spacing = mean(
                        vertex_footprint_index
                            .adjacent_cells(vertex_footprint_indexing)
                            .map(|cell_footprint_index| {
                                let cell_index = indexing::CellIndex {
                                    footprint: cell_footprint_index,
                                    z: vertex_index.z - 1,
                                };
                                volume.cell_value(cell_index).max(MIN_VOLUME)
                                    / grid.footprint_area()
                            }),
                    )
                    .unwrap();
                    lattice[vertex_index.to_array_index()] = lattice[indexing::VertexIndex {
                        footprint: vertex_footprint_index,
                        z: vertex_index.z - 1,
                    }
                    .to_array_index()]
                        + spacing;
                }
            }
        }
        Self { lattice }
    }
    pub fn new_from_height(
        grid: &Grid,
        terrain: &fields::Terrain,
        height: &fields::AreaScalarField,
    ) -> Self {
        let mut volume = fields::VolScalarField::zeros(grid.cell_indexing());
        for cell_footprint_index in grid.cell_footprint_indexing().iter() {
            let column_cell_volume = height.cell_footprint_value(cell_footprint_index)
                * grid.footprint_area()
                / grid.cell_indexing().num_z_cells() as Float;
            for cell_index in grid.cell_indexing().column(cell_footprint_index) {
                *volume.cell_value_mut(cell_index) = column_cell_volume;
            }
        }
        Self::new_from_volume(grid, terrain, &volume)
    }

    pub fn vertex_value(&self, vertex_index: indexing::VertexIndex) -> Float {
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
    cells: nd::Array<Cell, <indexing::CellIndex as indexing::Index>::ArrayIndex>,
}
impl DynamicGeometry {
    pub fn new_from_volume(
        static_geometry: StaticGeometry,
        volume: &fields::VolScalarField,
    ) -> Self {
        let z_lattice =
            ZLattice::new_from_volume(static_geometry.grid(), static_geometry.terrain(), volume);
        let mut cells = nd::Array::uninit(static_geometry.grid().cell_indexing().shape());
        for (cell_index, cell) in Self::iter_cells(&static_geometry, &z_lattice) {
            cells[cell_index.to_array_index()] = std::mem::MaybeUninit::new(cell);
        }
        Self {
            static_geometry,
            z_lattice,
            cells: unsafe { cells.assume_init() },
        }
    }

    pub fn new_from_height(
        static_geometry: StaticGeometry,
        height: &fields::AreaScalarField,
    ) -> Self {
        let z_lattice =
            ZLattice::new_from_height(static_geometry.grid(), static_geometry.terrain(), height);
        let mut cells = nd::Array::uninit(static_geometry.grid().cell_indexing().shape());
        for (cell_index, cell) in Self::iter_cells(&static_geometry, &z_lattice) {
            cells[cell_index.to_array_index()] = std::mem::MaybeUninit::new(cell);
        }
        Self {
            static_geometry,
            z_lattice,
            cells: unsafe { cells.assume_init() },
        }
    }

    pub fn grid(&self) -> &Grid {
        &self.static_geometry.grid
    }

    pub fn z_lattice(&self) -> &ZLattice {
        &self.z_lattice
    }

    pub fn terrain(&self) -> &fields::Terrain {
        &self.static_geometry.terrain
    }

    pub fn into_static_geometry(self) -> StaticGeometry {
        self.static_geometry
    }

    pub fn cell(&self, cell_index: indexing::CellIndex) -> &Cell {
        &self.cells[cell_index.to_array_index()]
    }

    pub fn make_cell_array<V, F: Fn(Float, Float, Float) -> V>(&self, f: F) -> nd::Array4<V> {
        let mut cells = nd::Array4::<V>::uninit(self.grid().cell_indexing().shape());
        for cell_index in self.grid().cell_indexing().iter() {
            let centroid = self.cell(cell_index).centroid;
            cells[cell_index.to_array_index()] =
                std::mem::MaybeUninit::new(f(centroid.x, centroid.y, centroid.z));
        }
        unsafe { cells.assume_init() }
    }

    pub fn make_vertex_array<V: Default, F: Fn(Float, Float, Float) -> V>(
        &self,
        f: F,
    ) -> nd::Array3<V> {
        let mut vertices = nd::Array3::<V>::uninit(self.grid().vertex_indexing().shape());
        for indexing::VertexIndex {
            footprint: indexing::VertexFootprintIndex { x, y },
            z,
        } in self.grid().vertex_indexing().iter()
        {
            vertices[[x, y, z]] = std::mem::MaybeUninit::new(f(
                self.grid().x_axis.vertices[x],
                self.grid().y_axis.vertices[y],
                self.z_lattice.lattice[[x, y, z]],
            ));
        }
        unsafe { vertices.assume_init() }
    }

    fn iter_cells<'a>(
        static_geometry: &'a StaticGeometry,
        z_lattice: &'a ZLattice,
    ) -> impl Iterator<Item = (indexing::CellIndex, Cell)> + 'a {
        let grid = static_geometry.grid();
        let cell_footprint_indexing = grid.cell_footprint_indexing();
        let cell_indexing = grid.cell_indexing();
        cell_footprint_indexing
            .iter()
            .map(move |cell_footprint_index| {
                let cell_footprint_pairs =
                    cell_footprint_indexing.compute_footprint_pairs(cell_footprint_index);
                let cell_footprint_centroid =
                    grid.compute_cell_footprint_centroid(cell_footprint_index);
                cell_indexing
                    .column(cell_footprint_index)
                    .map(move |cell_index| {
                        (
                            cell_index,
                            Cell {
                                index: cell_index,
                                volume: Self::compute_volume(
                                    static_geometry,
                                    z_lattice,
                                    cell_index,
                                ),
                                centroid: Self::compute_cell_centroid(
                                    z_lattice,
                                    cell_index,
                                    cell_footprint_centroid,
                                ),
                                faces: Self::compute_faces(
                                    static_geometry,
                                    z_lattice,
                                    cell_index,
                                    cell_footprint_pairs,
                                ),
                                lower_z_face: Self::compute_horiz_face(
                                    static_geometry,
                                    z_lattice,
                                    cell_index,
                                    HorizSide::Down,
                                ),
                                upper_z_face: Self::compute_horiz_face(
                                    static_geometry,
                                    z_lattice,
                                    cell_index,
                                    HorizSide::Up,
                                ),
                                classification: cell_indexing.classify_cell(cell_index),
                            },
                        )
                    })
            })
            .flatten()
    }

    fn compute_volume(
        static_geometry: &StaticGeometry,
        z_lattice: &ZLattice,
        cell_index: indexing::CellIndex,
    ) -> Float {
        let z = cell_index.z;
        static_geometry.grid().footprint_area()
            * cell_index
                .footprint
                .vertices_right_handed()
                .into_iter()
                .map(|vertex_footprint_index| {
                    z_lattice.vertex_value(indexing::VertexIndex {
                        footprint: vertex_footprint_index,
                        z: z + 1,
                    }) - z_lattice.vertex_value(indexing::VertexIndex {
                        footprint: vertex_footprint_index,
                        z,
                    })
                })
                .sum::<Float>()
            / 3.
    }

    fn compute_face_centroid(
        static_geometry: &StaticGeometry,
        z_lattice: &ZLattice,
        vertices: impl Iterator<Item = indexing::VertexIndex>,
    ) -> Point3 {
        let (value, count) = vertices.fold(
            (Vector3::zeros(), 0),
            |(sum, count), vertex_index: indexing::VertexIndex| {
                (
                    sum + Vector3::new(
                        static_geometry.grid().x_axis().vertices()[vertex_index.footprint.x],
                        static_geometry.grid().y_axis().vertices()[vertex_index.footprint.y],
                        z_lattice.vertex_value(vertex_index),
                    ),
                    count + 1,
                )
            },
        );
        Point3::new(
            value.x / count as Float,
            value.y / count as Float,
            value.z / count as Float,
        )
    }

    fn compute_cell_centroid(
        z_lattice: &ZLattice,
        cell_index: indexing::CellIndex,
        cell_footprint_centroid: Point2,
    ) -> Point3 {
        let z_centroid = (0..2)
            .map(|delta_z| {
                let z = cell_index.z + delta_z;
                cell_index
                    .footprint
                    .vertices_right_handed()
                    .into_iter()
                    .map(move |vertex_footprint| {
                        z_lattice.vertex_value(indexing::VertexIndex {
                            footprint: vertex_footprint,
                            z,
                        })
                    })
            })
            .flatten()
            .sum::<Float>()
            / indexing::NUM_CELL_VERTICES as Float;
        Point3::new(
            cell_footprint_centroid.x,
            cell_footprint_centroid.y,
            z_centroid,
        )
    }

    fn compute_faces(
        static_geometry: &StaticGeometry,
        z_lattice: &ZLattice,
        cell_index: indexing::CellIndex,
        cell_footprint_pairs: [indexing::CellFootprintPair; 3],
    ) -> [CellFace; 5] {
        [
            Self::compute_vert_face(
                static_geometry,
                z_lattice,
                cell_index,
                cell_footprint_pairs[0],
            ),
            Self::compute_vert_face(
                static_geometry,
                z_lattice,
                cell_index,
                cell_footprint_pairs[1],
            ),
            Self::compute_vert_face(
                static_geometry,
                z_lattice,
                cell_index,
                cell_footprint_pairs[2],
            ),
            Self::compute_horiz_face(static_geometry, z_lattice, cell_index, HorizSide::Up),
            Self::compute_horiz_face(static_geometry, z_lattice, cell_index, HorizSide::Down),
        ]
    }

    fn compute_horiz_face(
        static_geometry: &StaticGeometry,
        z_lattice: &ZLattice,
        cell_index: indexing::CellIndex,
        horiz_side: HorizSide,
    ) -> CellFace {
        let vertex_footprints = cell_index.footprint.vertices_right_handed();
        let vertices_z = match horiz_side {
            HorizSide::Up => cell_index.z + 1,
            HorizSide::Down => cell_index.z,
        };
        let vertices = [
            indexing::VertexIndex {
                footprint: vertex_footprints[0],
                z: vertices_z,
            },
            indexing::VertexIndex {
                footprint: vertex_footprints[1],
                z: vertices_z,
            },
            indexing::VertexIndex {
                footprint: vertex_footprints[2],
                z: vertices_z,
            },
        ];
        let make_vertex_point = |vertex: indexing::VertexIndex| {
            Point3::new(
                static_geometry.grid().x_axis.vertices[vertex.footprint.x],
                static_geometry.grid().y_axis.vertices[vertex.footprint.y],
                z_lattice.vertex_value(vertex),
            )
        };
        let p1 = make_vertex_point(vertices[0]);
        let p2 = make_vertex_point(vertices[1]);
        let p3 = make_vertex_point(vertices[2]);
        let d12 = p2 - p1;
        let d23 = p3 - p2;
        let unnorm_normal = d12.cross(&d23);
        let (normal, norm) = UnitVector3::new_and_get(unnorm_normal);
        let area = norm / 2.;
        let outward_normal = match horiz_side {
            HorizSide::Up => normal,
            HorizSide::Down => -normal,
        };

        let neighbor = match horiz_side {
            HorizSide::Up => {
                let z_plus_1 = cell_index.z + 1;
                if z_plus_1 >= static_geometry.grid().cell_indexing().num_z_cells() {
                    indexing::CellNeighbor::ZBoundary(indexing::Boundary::Upper)
                } else {
                    indexing::CellNeighbor::Cell(indexing::CellIndex {
                        z: z_plus_1,
                        ..cell_index
                    })
                }
            }
            HorizSide::Down => {
                if let Some(z_minus_1) = cell_index.z.checked_sub(1) {
                    indexing::CellNeighbor::Cell(indexing::CellIndex {
                        z: z_minus_1,
                        ..cell_index
                    })
                } else {
                    indexing::CellNeighbor::ZBoundary(indexing::Boundary::Lower)
                }
            }
        };

        let vertices = {
            let z = cell_index.z
                + match horiz_side {
                    HorizSide::Up => 1,
                    HorizSide::Down => 0,
                };
            let vertex_footprints = cell_index.footprint.vertices_right_handed();
            [
                indexing::VertexIndex {
                    footprint: vertex_footprints[0],
                    z,
                },
                indexing::VertexIndex {
                    footprint: vertex_footprints[1],
                    z,
                },
                indexing::VertexIndex {
                    footprint: vertex_footprints[2],
                    z,
                },
            ]
        };
        let centroid =
            Self::compute_face_centroid(static_geometry, z_lattice, vertices.iter().copied());
        CellFace {
            data: CellFaceData::Horiz(CellHorizFace {
                area,
                outward_normal,
                neighbor,
                vertices,
                centroid,
            }),
        }
    }

    fn compute_vert_face(
        static_geometry: &StaticGeometry,
        z_lattice: &ZLattice,
        cell_index: indexing::CellIndex,
        cell_footprint_pair: indexing::CellFootprintPair,
    ) -> CellFace {
        // --------
        // |     /|
        // |    / |
        // |   /  |
        // |  /   |
        // | /    |
        // |/     |
        // --------
        let indexing::CellIndex { z, .. } = cell_index;
        let cell_footprint_edge = static_geometry
            .grid()
            .compute_cell_footprint_edge(cell_footprint_pair);
        let area = {
            let vert_length_1 = z_lattice.vertex_value(indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprints[0],
                z: z + 1,
            }) - z_lattice.vertex_value(indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprints[0],
                z,
            });
            let vert_length_2 = z_lattice.vertex_value(indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprints[1],
                z: z + 1,
            }) - z_lattice.vertex_value(indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprints[1],
                z,
            });
            0.5 * cell_footprint_edge.length * (vert_length_1 + vert_length_2)
        };

        let vertices = [
            indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprints[0],
                z: cell_index.z,
            },
            indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprints[0],
                z: cell_index.z + 1,
            },
            indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprints[1],
                z: cell_index.z,
            },
            indexing::VertexIndex {
                footprint: cell_footprint_edge.vertex_footprints[1],
                z: cell_index.z + 1,
            },
        ];
        let centroid =
            Self::compute_face_centroid(static_geometry, z_lattice, vertices.iter().copied());
        CellFace {
            data: CellFaceData::Vert(CellVertFace {
                area,
                cell_index,
                cell_footprint_edge,
                vertices,
                centroid,
            }),
        }
    }
}
#[pymethods]
impl DynamicGeometry {
    #[pyo3(name = "z_lattice")]
    pub fn z_lattice_py<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.z_lattice.z_lattice(py)
    }
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub index: indexing::CellIndex,
    pub volume: Float,
    pub centroid: Point3,
    pub faces: [CellFace; 5],
    pub lower_z_face: CellFace,
    pub upper_z_face: CellFace,
    pub classification: indexing::CellClassification,
}

#[derive(Clone, Debug)]
pub struct CellFootprintEdge {
    pub cell_footprint_pair: indexing::CellFootprintPair,
    pub outward_normal: UnitVector2,
    pub length: Float,
    pub vertex_footprints: [indexing::VertexFootprintIndex; 2],
}

#[derive(Clone, Debug)]
struct CellVertFace {
    pub cell_index: indexing::CellIndex,
    pub area: Float,
    cell_footprint_edge: CellFootprintEdge,
    vertices: [indexing::VertexIndex; 4],
    centroid: Point3,
}

#[derive(Clone, Debug)]
struct CellHorizFace {
    pub area: Float,
    outward_normal: UnitVector3,
    pub neighbor: indexing::CellNeighbor,
    vertices: [indexing::VertexIndex; 3],
    centroid: Point3,
}

#[derive(Clone, Copy, Debug)]
enum HorizSide {
    Up,
    Down,
}

#[derive(Clone, Debug)]
enum CellFaceData {
    Vert(CellVertFace),
    Horiz(CellHorizFace),
}
#[derive(Clone, Debug)]
pub struct CellFace {
    data: CellFaceData,
}
impl CellFace {
    pub fn area(&self) -> Float {
        match &self.data {
            CellFaceData::Vert(vert_face) => vert_face.area,
            CellFaceData::Horiz(horiz_face) => horiz_face.area,
        }
    }

    pub fn neighbor(&self) -> indexing::CellNeighbor {
        match &self.data {
            CellFaceData::Vert(vert_face) => {
                match vert_face.cell_footprint_edge.cell_footprint_pair.neighbor {
                    indexing::CellFootprintNeighbor::CellFootprint(footprint) => {
                        indexing::CellNeighbor::Cell(indexing::CellIndex {
                            footprint,
                            z: vert_face.cell_index.z,
                        })
                    }
                    indexing::CellFootprintNeighbor::XBoundary(boundary) => {
                        indexing::CellNeighbor::XBoundary(boundary)
                    }
                    indexing::CellFootprintNeighbor::YBoundary(boundary) => {
                        indexing::CellNeighbor::YBoundary(boundary)
                    }
                }
            }
            CellFaceData::Horiz(horiz_face) => horiz_face.neighbor,
        }
    }

    pub fn outward_normal(&self) -> UnitVector3 {
        match &self.data {
            CellFaceData::Vert(vert_face) => {
                let normal_2d = vert_face.cell_footprint_edge.outward_normal;
                UnitVector3::new_unchecked(Vector3::new(normal_2d.x, normal_2d.y, 0.))
            }
            CellFaceData::Horiz(horiz_face) => horiz_face.outward_normal,
        }
    }

    pub fn centroid(&self) -> Point3 {
        match &self.data {
            CellFaceData::Vert(vert_face) => vert_face.centroid,
            CellFaceData::Horiz(horiz_face) => horiz_face.centroid,
        }
    }

    pub fn vertices(&self) -> CellFaceVertices {
        match &self.data {
            CellFaceData::Vert(vert_face) => CellFaceVertices::Vert(&vert_face.vertices),
            CellFaceData::Horiz(horiz_face) => CellFaceVertices::Horiz(&horiz_face.vertices),
        }
    }
}

pub enum CellFaceVertices<'a> {
    Vert(&'a [indexing::VertexIndex; 4]),
    Horiz(&'a [indexing::VertexIndex; 3]),
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
        let height = fields::AreaScalarField::new(&grid, |_, _| 0.);
        let static_geometry = StaticGeometry::new(grid, &|x, y| 10. * x * y);
        let dynamic_geometry = DynamicGeometry::new_from_height(static_geometry, &height);

        let heights = (&dynamic_geometry.z_lattice.lattice.slice(s![.., .., 1..])
            - &dynamic_geometry.z_lattice.lattice.slice(s![.., .., ..-1]))
            .into_owned();
        let expected_heights = 0. * &heights;
        assert_relative_eq!(heights, expected_heights, epsilon = 1e-6);

        let cell_index = indexing::CellIndex {
            footprint: indexing::CellFootprintIndex {
                x: 15,
                y: 16,
                triangle: indexing::Triangle::LowerRight,
            },
            z: 5,
        };
        assert_relative_eq!(dynamic_geometry.cell(cell_index).volume, 0., epsilon = 1e-6);
        assert_relative_eq!(
            dynamic_geometry.cell(cell_index).centroid.z,
            10. * 0.5 * 0.5,
            epsilon = 0.3
        );
    }

    #[test]
    fn test_construct_z_lattice_flat() {
        let grid = Grid::new(Axis::new(0., 1., 29), Axis::new(0., 1., 32), 10);
        let height = fields::AreaScalarField::new(&grid, |_, _| 7.3);

        let static_geometry = StaticGeometry::new(grid, &|_, _| 0.);
        let dynamic_geometry = DynamicGeometry::new_from_height(static_geometry, &height);

        assert_relative_eq!(
            dynamic_geometry.z_lattice.lattice,
            nd::Array1::linspace(
                0.,
                7.3,
                dynamic_geometry.grid().vertex_indexing().num_z_points()
            )
            .slice(nd::s![nd::NewAxis, nd::NewAxis, ..])
            .broadcast(dynamic_geometry.z_lattice.lattice.dim())
            .unwrap(),
            epsilon = 1e-6,
        );

        let cell_index = indexing::CellIndex {
            footprint: indexing::CellFootprintIndex {
                x: 0,
                y: 0,
                triangle: indexing::Triangle::LowerRight,
            },
            z: 0,
        };
        let cell = dynamic_geometry.cell(cell_index);
        assert_relative_eq!(
            cell.volume,
            (1. / 29.) * (1. / 32.) * (7.3 / 10.) / 2.,
            epsilon = 1e-8
        );
        assert_relative_eq!(cell.centroid.z, 7.3 / 10. * 0.5);
        let expected_horiz_centroid = cell_index
            .footprint
            .vertices_right_handed()
            .into_iter()
            .map(|vertex_footprint_index| {
                Vector2::new(
                    dynamic_geometry.grid().x_axis().vertices()[vertex_footprint_index.x],
                    dynamic_geometry.grid().y_axis().vertices()[vertex_footprint_index.y],
                )
            })
            .sum::<Vector2>()
            / 3.;
        assert_relative_eq!(expected_horiz_centroid.x, cell.centroid.x);
        assert_relative_eq!(expected_horiz_centroid.y, cell.centroid.y);
    }

    #[test]
    fn test_construct_z_lattice_grade() {
        let grid = Grid::new(Axis::new(0., 1., 60), Axis::new(0., 1., 31), 10);
        let terrain = fields::Terrain::new(&grid, &|_, _| 0.);
        let mut height = fields::AreaScalarField::new(&grid, |_, _| 0.);
        for cell_footprint_index in grid.cell_footprint_indexing().iter() {
            let centroid = grid.compute_cell_footprint_centroid(cell_footprint_index);
            *height.cell_footprint_value_mut(cell_footprint_index) =
                1. + centroid[0] + 2. * centroid[1];
        }

        let z_lattice = ZLattice::new_from_height(&grid, &terrain, &height);

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
