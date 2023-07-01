use ndarray as nd;
use strum::{EnumCount, IntoEnumIterator};

use crate::{Array1, Float};

pub const NUM_CELL_VERTICES: usize = 6;

#[derive(Copy, Clone, Debug, PartialEq, Eq, strum_macros::EnumIter, strum_macros::EnumCount)]
pub enum Triangle {
    UpperLeft = 0,
    LowerRight = 1,
}
impl Triangle {
    pub fn flip(self) -> Self {
        match self {
            Self::UpperLeft => Self::LowerRight,
            Self::LowerRight => Self::UpperLeft,
        }
    }
}

pub trait Indexing {
    type Index: Index;

    fn shape(&self) -> <<Self as Indexing>::Index as Index>::ArrayIndex;

    fn len(&self) -> usize;

    fn flatten(&self, index: Self::Index) -> usize;
    fn unflatten(&self, flat_index: usize) -> Self::Index;
}
pub trait Index: Copy + std::fmt::Debug + PartialEq + Eq {
    type ArrayIndex: nd::Dimension;

    fn to_array_index(self) -> Self::ArrayIndex;
}

pub fn iter_indices<I: Indexing>(indexing: &I) -> impl Iterator<Item = I::Index> + '_ {
    (0..indexing.len()).map(|flat_index| indexing.unflatten(flat_index))
}

pub fn flatten_array<I: Indexing>(
    indexing: &I,
    array: &nd::Array<Float, <I::Index as Index>::ArrayIndex>,
) -> Array1 {
    let mut flattened = Array1::zeros(indexing.len());
    for (flat_index, value) in flattened.iter_mut().enumerate() {
        let index = indexing.unflatten(flat_index);
        *value = array[index.to_array_index()];
    }
    flattened
}

pub fn unflatten_array<I: Indexing>(
    indexing: &I,
    flattened: &Array1,
) -> nd::Array<Float, <I::Index as Index>::ArrayIndex> {
    let mut unflattened = nd::Array::zeros(indexing.shape());
    for (flat_index, value) in flattened.iter().copied().enumerate() {
        let index = indexing.unflatten(flat_index);
        unflattened[index.to_array_index()] = value;
    }
    unflattened
}

#[derive(Clone)]
pub struct VertexFootprintIndexing {
    num_x_points: usize,
    num_y_points: usize,
}
impl VertexFootprintIndexing {
    pub fn new(num_x_points: usize, num_y_points: usize) -> Self {
        Self {
            num_x_points,
            num_y_points,
        }
    }

    pub fn num_x_points(&self) -> usize {
        self.num_x_points
    }

    pub fn num_y_points(&self) -> usize {
        self.num_y_points
    }

    pub fn neighbors(
        &self,
        center: VertexFootprintIndex,
    ) -> impl Iterator<Item = VertexFootprintIndex> {
        let min_x = center.x.checked_sub(1).unwrap_or(1);
        let max_x = (center.x + 2).min(self.num_x_points);
        let min_y = center.y.checked_sub(1).unwrap_or(1);
        let max_y = (center.y + 2).min(self.num_y_points);

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
impl Indexing for VertexFootprintIndexing {
    type Index = VertexFootprintIndex;

    fn shape(&self) -> <<Self as Indexing>::Index as Index>::ArrayIndex {
        nd::Dim([self.num_x_points, self.num_y_points])
    }

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VertexFootprintIndex {
    pub x: usize,
    pub y: usize,
}
impl VertexFootprintIndex {
    pub fn adjacent_cells(
        self,
        indexing: &VertexFootprintIndexing,
    ) -> impl Iterator<Item = CellFootprintIndex> + '_ {
        (-1..1).flat_map(move |dx| {
            let x = self.x.checked_add_signed(dx).and_then(move |x| {
                if x < indexing.num_x_points - 1 {
                    Some(x)
                } else {
                    None
                }
            });
            (-1..1).flat_map(move |dy| {
                let y = self.y.checked_add_signed(dy).and_then(move |y| {
                    if y < indexing.num_y_points - 1 {
                        Some(y)
                    } else {
                        None
                    }
                });
                Triangle::iter().filter_map(move |triangle| {
                    Some(CellFootprintIndex {
                        x: x?,
                        y: y?,
                        triangle,
                    })
                })
            })
        })
    }

    pub fn increment_x(self, dx: isize) -> Self {
        Self {
            x: self.x.checked_add_signed(dx).unwrap(),
            y: self.y,
        }
    }

    pub fn increment_y(self, dy: isize) -> Self {
        Self {
            x: self.x,
            y: self.y.checked_add_signed(dy).unwrap(),
        }
    }
}
impl Index for VertexFootprintIndex {
    type ArrayIndex = nd::Dim<[usize; 2]>;

    fn to_array_index(self) -> Self::ArrayIndex {
        nd::Dim([self.x, self.y])
    }
}

#[derive(Clone)]
pub struct VertexIndexing {
    cell_indexing: CellIndexing,
}
impl VertexIndexing {
    pub fn new(cell_indexing: CellIndexing) -> Self {
        Self { cell_indexing }
    }

    pub fn cell_indexing(&self) -> &CellIndexing {
        &self.cell_indexing
    }

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
        self.cell_indexing.num_z_cells + 1
    }

    pub fn column(
        &self,
        footprint: VertexFootprintIndex,
    ) -> impl Iterator<Item = VertexIndex> + '_ {
        (0..self.num_z_points()).map(move |z| VertexIndex { footprint, z })
    }

    pub fn classify_vertex(&self, vertex: VertexIndex) -> VertexClassification {
        let VertexIndex { footprint, z } = vertex;
        let VertexFootprintIndex { x, y } = footprint;

        if z == 0 {
            VertexClassification::Floor
        } else if z == self.num_z_points() - 1 {
            VertexClassification::Surface
        } else {
            if x == 0 {
                if y == 0 {
                    VertexClassification::LowerLeft
                } else if y == self.num_y_points() - 1 {
                    VertexClassification::UpperLeft
                } else {
                    VertexClassification::Left
                }
            } else if x == self.num_x_points() - 1 {
                if y == 0 {
                    VertexClassification::LowerRight
                } else if y == self.num_y_points() - 1 {
                    VertexClassification::LowerRight
                } else {
                    VertexClassification::Right
                }
            } else {
                if y == 0 {
                    VertexClassification::Lower
                } else if y == self.num_y_points() - 1 {
                    VertexClassification::Upper
                } else {
                    VertexClassification::Interior
                }
            }
        }
    }
}
impl Indexing for VertexIndexing {
    type Index = VertexIndex;

    fn shape(&self) -> <<Self as Indexing>::Index as Index>::ArrayIndex {
        nd::Dim([
            self.num_x_points(),
            self.num_y_points(),
            self.num_z_points(),
        ])
    }

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VertexIndex {
    pub footprint: VertexFootprintIndex,
    pub z: usize,
}
impl VertexIndex {
    pub fn increment_x(self, dx: isize) -> Self {
        Self {
            footprint: self.footprint.increment_x(dx),
            ..self
        }
    }

    pub fn increment_y(self, dy: isize) -> Self {
        Self {
            footprint: self.footprint.increment_y(dy),
            ..self
        }
    }

    pub fn increment_z(self, dz: isize) -> Self {
        Self {
            z: self.z.checked_add_signed(dz).unwrap(),
            ..self
        }
    }
}
impl Index for VertexIndex {
    type ArrayIndex = nd::Dim<[usize; 3]>;

    fn to_array_index(self) -> Self::ArrayIndex {
        nd::Dim([self.footprint.x, self.footprint.y, self.z])
    }
}

#[derive(Debug)]
pub enum VertexClassification {
    /// For `z = 0`
    Floor,
    /// For `z = num_z_points - 1`
    Surface,
    /// For `x = 0`
    Left,
    /// For `x = num_x_points - 1`
    Right,
    /// For `y = 0`
    Lower,
    /// For `y = num_y_points - 1`
    Upper,
    /// For `x = 0`, `y = 0`
    LowerLeft,
    /// For `x = num_x_points - 1`, `y = 0`
    LowerRight,
    /// For `x = 0`, `y = num_y_points - 1`
    UpperLeft,
    /// For `x = num_x_points - 1`, `y = num_y_points - 1`
    UpperRight,
    /// Otherwise
    Interior,
}

#[derive(Clone)]
pub struct CellFootprintIndexing {
    vertex_footprint_indexing: VertexFootprintIndexing,
}
impl CellFootprintIndexing {
    pub fn new(vertex_footprint_indexing: VertexFootprintIndexing) -> Self {
        Self {
            vertex_footprint_indexing,
        }
    }

    pub fn vertex_footprint_indexing(&self) -> &VertexFootprintIndexing {
        &self.vertex_footprint_indexing
    }

    pub fn num_x_cells(&self) -> usize {
        self.vertex_footprint_indexing.num_x_points - 1
    }

    pub fn num_y_cells(&self) -> usize {
        self.vertex_footprint_indexing.num_y_points - 1
    }

    pub fn compute_footprint_pairs(&self, footprint: CellFootprintIndex) -> [CellFootprintPair; 3] {
        let triangle = footprint.triangle;
        let neighbor_triangle = triangle.flip();
        // ---------------
        // |     /|     /|
        // |    / |    / |
        // |   /  |   /  |
        // |  /   |  /   |
        // | /    | /    |
        // |/     |/     |
        // ---------------
        // |     /|     /|
        // |    / |    / |
        // |   /  |   /  |
        // |  /   |  /   |
        // | /    | /    |
        // |/     |/     |
        // ---------------
        match triangle {
            Triangle::UpperLeft => [
                CellFootprintPair {
                    footprint,
                    triangle_side: TriangleSide::UpperLeft(UpperLeftTriangleSide::Up),
                    neighbor: if footprint.y + 1 < self.num_y_cells() {
                        CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                            y: footprint.y + 1,
                            triangle: neighbor_triangle,
                            ..footprint
                        })
                    } else {
                        CellFootprintNeighbor::YBoundary(Boundary::Upper)
                    },
                },
                CellFootprintPair {
                    footprint,
                    triangle_side: TriangleSide::UpperLeft(UpperLeftTriangleSide::Left),
                    neighbor: if let Some(neighbor_x) = footprint.x.checked_sub(1) {
                        CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                            x: neighbor_x,
                            triangle: neighbor_triangle,
                            ..footprint
                        })
                    } else {
                        CellFootprintNeighbor::XBoundary(Boundary::Lower)
                    },
                },
                CellFootprintPair {
                    footprint,
                    triangle_side: TriangleSide::UpperLeft(UpperLeftTriangleSide::DownRight),
                    neighbor: CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                        triangle: neighbor_triangle,
                        ..footprint
                    }),
                },
            ],
            Triangle::LowerRight => [
                CellFootprintPair {
                    footprint,
                    triangle_side: TriangleSide::LowerRight(LowerRightTriangleSide::Down),
                    neighbor: if let Some(neighbor_y) = footprint.y.checked_sub(1) {
                        CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                            y: neighbor_y,
                            triangle: neighbor_triangle,
                            ..footprint
                        })
                    } else {
                        CellFootprintNeighbor::YBoundary(Boundary::Lower)
                    },
                },
                CellFootprintPair {
                    triangle_side: TriangleSide::LowerRight(LowerRightTriangleSide::Right),
                    footprint,
                    neighbor: if footprint.x + 1 < self.num_x_cells() {
                        CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                            x: footprint.x + 1,
                            triangle: neighbor_triangle,
                            ..footprint
                        })
                    } else {
                        CellFootprintNeighbor::XBoundary(Boundary::Upper)
                    },
                },
                CellFootprintPair {
                    triangle_side: TriangleSide::LowerRight(LowerRightTriangleSide::UpLeft),
                    footprint,
                    neighbor: CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                        triangle: neighbor_triangle,
                        ..footprint
                    }),
                },
            ],
        }
    }
}
impl Indexing for CellFootprintIndexing {
    type Index = CellFootprintIndex;

    fn shape(&self) -> <<Self as Indexing>::Index as Index>::ArrayIndex {
        nd::Dim([self.num_x_cells(), self.num_y_cells(), Triangle::COUNT])
    }

    fn len(&self) -> usize {
        self.num_x_cells() * self.num_y_cells() * Triangle::COUNT
    }

    fn flatten(&self, index: Self::Index) -> usize {
        self.num_y_cells() * Triangle::COUNT * index.x
            + Triangle::COUNT * index.y
            + index.triangle as usize
    }

    fn unflatten(&self, flat_index: usize) -> Self::Index {
        let triangle = match flat_index % Triangle::COUNT {
            x if x == Triangle::LowerRight as usize => Triangle::LowerRight,
            x if x == Triangle::UpperLeft as usize => Triangle::UpperLeft,
            _ => unreachable!(),
        };
        let y = (flat_index / Triangle::COUNT) % self.num_y_cells();
        let x = flat_index / Triangle::COUNT / self.num_y_cells();
        CellFootprintIndex { x, y, triangle }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CellFootprintIndex {
    pub x: usize,
    pub y: usize,
    pub triangle: Triangle,
}
impl CellFootprintIndex {
    /// Vertices in right-handed order looking top down
    pub fn vertices_right_handed(self) -> [VertexFootprintIndex; 3] {
        match self.triangle {
            Triangle::UpperLeft => [
                VertexFootprintIndex {
                    x: self.x,
                    y: self.y,
                },
                VertexFootprintIndex {
                    x: self.x + 1,
                    y: self.y + 1,
                },
                VertexFootprintIndex {
                    x: self.x,
                    y: self.y + 1,
                },
            ],
            Triangle::LowerRight => [
                VertexFootprintIndex {
                    x: self.x,
                    y: self.y,
                },
                VertexFootprintIndex {
                    x: self.x + 1,
                    y: self.y,
                },
                VertexFootprintIndex {
                    x: self.x + 1,
                    y: self.y + 1,
                },
            ],
        }
    }
}
impl Index for CellFootprintIndex {
    type ArrayIndex = nd::Dim<[usize; 3]>;

    fn to_array_index(self) -> Self::ArrayIndex {
        nd::Dim([self.x, self.y, self.triangle as usize])
    }
}

#[derive(Clone)]
pub struct CellIndexing {
    cell_footprint_indexing: CellFootprintIndexing,
    num_z_cells: usize,
}
impl CellIndexing {
    pub fn new(cell_footprint_indexing: CellFootprintIndexing, num_z_cells: usize) -> Self {
        Self {
            cell_footprint_indexing,
            num_z_cells,
        }
    }

    pub fn cell_footprint_indexing(&self) -> &CellFootprintIndexing {
        &self.cell_footprint_indexing
    }

    pub fn num_z_cells(&self) -> usize {
        self.num_z_cells
    }

    pub fn column(&self, footprint: CellFootprintIndex) -> impl Iterator<Item = CellIndex> {
        (0..self.num_z_cells).map(move |z| CellIndex { footprint, z })
    }

    pub fn classify_cell(&self, cell_index: CellIndex) -> CellClassification {
        let CellIndex {
            footprint: CellFootprintIndex { x, y, .. },
            z,
        } = cell_index;
        if z == 0 {
            CellClassification::ZBoundary(Boundary::Lower)
        } else if z == self.num_z_cells() - 1 {
            CellClassification::ZBoundary(Boundary::Upper)
        } else if x == 0 {
            CellClassification::XBoundary(Boundary::Lower)
        } else if x == self.cell_footprint_indexing.num_x_cells() - 1 {
            CellClassification::XBoundary(Boundary::Upper)
        } else if y == 0 {
            CellClassification::YBoundary(Boundary::Lower)
        } else if y == self.cell_footprint_indexing.num_y_cells() - 1 {
            CellClassification::YBoundary(Boundary::Upper)
        } else {
            CellClassification::Interior
        }
    }
}
impl Indexing for CellIndexing {
    type Index = CellIndex;

    fn shape(&self) -> <<Self as Indexing>::Index as Index>::ArrayIndex {
        nd::Dim([
            self.cell_footprint_indexing.num_x_cells(),
            self.cell_footprint_indexing.num_y_cells(),
            self.num_z_cells,
            Triangle::COUNT,
        ])
    }

    fn len(&self) -> usize {
        self.cell_footprint_indexing.len() * self.num_z_cells
    }

    fn flatten(&self, index: Self::Index) -> usize {
        self.cell_footprint_indexing.flatten(index.footprint) * self.num_z_cells + index.z
    }

    fn unflatten(&self, flat_index: usize) -> Self::Index {
        CellIndex {
            footprint: self
                .cell_footprint_indexing
                .unflatten(flat_index / self.num_z_cells),
            z: flat_index % self.num_z_cells,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CellIndex {
    pub footprint: CellFootprintIndex,
    pub z: usize,
}
impl CellIndex {
    pub fn flip(self) -> Self {
        Self {
            footprint: CellFootprintIndex {
                triangle: self.footprint.triangle.flip(),
                ..self.footprint
            },
            ..self
        }
    }
}
impl Index for CellIndex {
    type ArrayIndex = nd::Dim<[usize; 4]>;

    fn to_array_index(self) -> Self::ArrayIndex {
        nd::Dim([
            self.footprint.x,
            self.footprint.y,
            self.z,
            self.footprint.triangle as usize,
        ])
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CellClassification {
    XBoundary(Boundary),
    YBoundary(Boundary),
    ZBoundary(Boundary),
    Interior,
}

#[derive(Clone, Copy, Debug)]
pub enum CellFootprintNeighbor {
    CellFootprint(CellFootprintIndex),
    XBoundary(Boundary),
    YBoundary(Boundary),
}

#[derive(Clone, Copy, Debug)]
pub enum UpperLeftTriangleSide {
    Up,
    Left,
    DownRight,
}
#[derive(Clone, Copy, Debug)]
pub enum LowerRightTriangleSide {
    Down,
    Right,
    UpLeft,
}

#[derive(Clone, Copy, Debug)]
pub enum TriangleSide {
    UpperLeft(UpperLeftTriangleSide),
    LowerRight(LowerRightTriangleSide),
}

#[derive(Clone, Copy, Debug)]
pub struct CellFootprintPair {
    pub footprint: CellFootprintIndex,
    pub triangle_side: TriangleSide,
    pub neighbor: CellFootprintNeighbor,
}

#[derive(Clone, Copy, Debug)]
pub enum Boundary {
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
pub enum CellNeighbor {
    Cell(CellIndex),
    XBoundary(Boundary),
    YBoundary(Boundary),
    ZBoundary(Boundary),
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_vertex_footprint_index() {
        let vertex_footprint_indexing = VertexFootprintIndexing {
            num_x_points: 5,
            num_y_points: 27,
        };

        {
            let vertex_footprint_index = VertexFootprintIndex { x: 0, y: 0 };
            assert_eq!(
                vertex_footprint_index
                    .adjacent_cells(&vertex_footprint_indexing)
                    .collect::<Vec<_>>(),
                vec![
                    CellFootprintIndex {
                        x: 0,
                        y: 0,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 0,
                        y: 0,
                        triangle: Triangle::LowerRight
                    }
                ]
            );
            assert_eq!(
                vertex_footprint_indexing
                    .neighbors(vertex_footprint_index)
                    .collect::<Vec<_>>(),
                vec![
                    VertexFootprintIndex { x: 1, y: 0 },
                    VertexFootprintIndex { x: 0, y: 1 }
                ]
            );
        }
        {
            let vertex_footprint_index = VertexFootprintIndex { x: 2, y: 0 };
            assert_eq!(
                vertex_footprint_index
                    .adjacent_cells(&vertex_footprint_indexing)
                    .collect::<Vec<_>>(),
                vec![
                    CellFootprintIndex {
                        x: 1,
                        y: 0,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 1,
                        y: 0,
                        triangle: Triangle::LowerRight
                    },
                    CellFootprintIndex {
                        x: 2,
                        y: 0,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 2,
                        y: 0,
                        triangle: Triangle::LowerRight
                    },
                ]
            );
            assert_eq!(
                vertex_footprint_indexing
                    .neighbors(vertex_footprint_index)
                    .collect::<Vec<_>>(),
                vec![
                    VertexFootprintIndex { x: 1, y: 0 },
                    VertexFootprintIndex { x: 3, y: 0 },
                    VertexFootprintIndex { x: 2, y: 1 }
                ]
            );
        }
        {
            let vertex_footprint_index = VertexFootprintIndex { x: 0, y: 11 };
            assert_eq!(
                vertex_footprint_index
                    .adjacent_cells(&vertex_footprint_indexing)
                    .collect::<Vec<_>>(),
                vec![
                    CellFootprintIndex {
                        x: 0,
                        y: 10,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 0,
                        y: 10,
                        triangle: Triangle::LowerRight
                    },
                    CellFootprintIndex {
                        x: 0,
                        y: 11,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 0,
                        y: 11,
                        triangle: Triangle::LowerRight
                    },
                ]
            );
            assert_eq!(
                vertex_footprint_indexing
                    .neighbors(vertex_footprint_index)
                    .collect::<Vec<_>>(),
                vec![
                    VertexFootprintIndex { x: 1, y: 11 },
                    VertexFootprintIndex { x: 0, y: 10 },
                    VertexFootprintIndex { x: 0, y: 12 }
                ]
            );
        }
        {
            let vertex_footprint_index = VertexFootprintIndex { x: 4, y: 11 };
            assert_eq!(
                vertex_footprint_index
                    .adjacent_cells(&vertex_footprint_indexing)
                    .collect::<Vec<_>>(),
                vec![
                    CellFootprintIndex {
                        x: 3,
                        y: 10,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 10,
                        triangle: Triangle::LowerRight
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 11,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 11,
                        triangle: Triangle::LowerRight
                    },
                ]
            );
            assert_eq!(
                vertex_footprint_indexing
                    .neighbors(vertex_footprint_index)
                    .collect::<Vec<_>>(),
                vec![
                    VertexFootprintIndex { x: 3, y: 11 },
                    VertexFootprintIndex { x: 4, y: 10 },
                    VertexFootprintIndex { x: 4, y: 12 }
                ]
            );
        }
        {
            let vertex_footprint_index = VertexFootprintIndex { x: 3, y: 26 };
            assert_eq!(
                vertex_footprint_index
                    .adjacent_cells(&vertex_footprint_indexing)
                    .collect::<Vec<_>>(),
                vec![
                    CellFootprintIndex {
                        x: 2,
                        y: 25,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 2,
                        y: 25,
                        triangle: Triangle::LowerRight
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 25,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 25,
                        triangle: Triangle::LowerRight
                    },
                ]
            );
            assert_eq!(
                vertex_footprint_indexing
                    .neighbors(vertex_footprint_index)
                    .collect::<Vec<_>>(),
                vec![
                    VertexFootprintIndex { x: 2, y: 26 },
                    VertexFootprintIndex { x: 4, y: 26 },
                    VertexFootprintIndex { x: 3, y: 25 },
                ]
            );
        }
        {
            let vertex_footprint_index = VertexFootprintIndex { x: 3, y: 20 };
            assert_eq!(
                vertex_footprint_index
                    .adjacent_cells(&vertex_footprint_indexing)
                    .collect::<Vec<_>>(),
                vec![
                    CellFootprintIndex {
                        x: 2,
                        y: 19,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 2,
                        y: 19,
                        triangle: Triangle::LowerRight
                    },
                    CellFootprintIndex {
                        x: 2,
                        y: 20,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 2,
                        y: 20,
                        triangle: Triangle::LowerRight
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 19,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 19,
                        triangle: Triangle::LowerRight
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 20,
                        triangle: Triangle::UpperLeft
                    },
                    CellFootprintIndex {
                        x: 3,
                        y: 20,
                        triangle: Triangle::LowerRight
                    },
                ]
            );
            assert_eq!(
                vertex_footprint_indexing
                    .neighbors(vertex_footprint_index)
                    .collect::<Vec<_>>(),
                vec![
                    VertexFootprintIndex { x: 2, y: 20 },
                    VertexFootprintIndex { x: 4, y: 20 },
                    VertexFootprintIndex { x: 3, y: 19 },
                    VertexFootprintIndex { x: 3, y: 21 }
                ]
            );
        }
    }

    #[test]
    fn test_indexing() {
        let vertex_footprint_indexing = VertexFootprintIndexing {
            num_x_points: 5,
            num_y_points: 27,
        };
        let vertex_footprint = VertexFootprintIndex { x: 2, y: 13 };
        assert_eq!(
            vertex_footprint_indexing.flatten(vertex_footprint.increment_y(2))
                - vertex_footprint_indexing.flatten(vertex_footprint),
            2
        );
        assert_eq!(
            vertex_footprint_indexing.flatten(vertex_footprint.increment_x(1))
                - vertex_footprint_indexing.flatten(vertex_footprint),
            vertex_footprint_indexing.num_y_points()
        );
        test_indexing_impl(&vertex_footprint_indexing);

        let cell_footprint_indexing = CellFootprintIndexing {
            vertex_footprint_indexing,
        };
        test_indexing_impl(&cell_footprint_indexing);

        let cell_indexing = CellIndexing {
            cell_footprint_indexing,
            num_z_cells: 19,
        };
        test_indexing_impl(&cell_indexing);

        let vertex_indexing = VertexIndexing { cell_indexing };
        let vertex = VertexIndex {
            footprint: vertex_footprint,
            z: 4,
        };
        assert_eq!(
            vertex_indexing.flatten(vertex.increment_z(2)) - vertex_indexing.flatten(vertex),
            2
        );
        assert_eq!(
            vertex_indexing.flatten(vertex.increment_y(1)) - vertex_indexing.flatten(vertex),
            vertex_indexing.num_z_points()
        );
        assert_eq!(
            vertex_indexing.flatten(vertex.increment_x(1)) - vertex_indexing.flatten(vertex),
            vertex_indexing.num_y_points() * vertex_indexing.num_z_points()
        );
        test_indexing_impl(&vertex_indexing);
    }

    fn test_indexing_impl<I: Indexing>(indexing: &I) {
        for (expected_flat_index, index) in iter_indices(indexing).enumerate() {
            let flat_index = indexing.flatten(index);
            assert_eq!(expected_flat_index, flat_index);
        }
    }
}
