use ndarray as nd;
use strum::{EnumCount, IntoEnumIterator};

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
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn flatten(&self, index: Self::Index) -> usize;
    fn unflatten(&self, flat_index: usize) -> Self::Index;
}
pub trait Index: Copy + std::fmt::Debug + PartialEq + Eq {
    type ArrayIndex: nd::ShapeBuilder;

    fn to_array_index(self) -> Self::ArrayIndex;
}

pub fn iter_indices<I: Indexing>(indexing: &I) -> impl Iterator<Item = I::Index> + '_ {
    (0..indexing.len()).map(|flat_index| indexing.unflatten(flat_index))
}

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
impl Indexing for VertexFootprintIndexing {
    type Index = VertexFootprintIndex;

    fn shape(&self) -> <<Self as Indexing>::Index as Index>::ArrayIndex {
        [self.num_x_points, self.num_y_points]
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
}
impl Index for VertexFootprintIndex {
    type ArrayIndex = [usize; 2];

    fn to_array_index(self) -> Self::ArrayIndex {
        [self.x, self.y]
    }
}

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
        self.cell_indexing.num_z_cells_per_column + 1
    }

    pub fn column(
        &self,
        footprint: VertexFootprintIndex,
    ) -> impl Iterator<Item = VertexIndex> + '_ {
        (0..self.num_z_points()).map(move |z| VertexIndex { footprint, z })
    }
}
impl Indexing for VertexIndexing {
    type Index = VertexIndex;

    fn shape(&self) -> <<Self as Indexing>::Index as Index>::ArrayIndex {
        [
            self.num_x_points(),
            self.num_y_points(),
            self.num_z_points(),
        ]
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
impl Index for VertexIndex {
    type ArrayIndex = [usize; 3];

    fn to_array_index(self) -> Self::ArrayIndex {
        [self.footprint.x, self.footprint.y, self.z]
    }
}

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

    pub fn compute_footprint_edges(&self, footprint: CellFootprintIndex) -> [CellFootprintEdge; 3] {
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
                CellFootprintEdge {
                    x: footprint.x,
                    y: footprint.y,
                    triangle_edge: TriangleEdge::UpperLeft(UpperLeftTriangleEdge::Up),
                    neighbor: if footprint.y + 1 < self.num_y_cells() {
                        CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                            y: footprint.y + 1,
                            triangle: neighbor_triangle,
                            ..footprint
                        })
                    } else {
                        CellFootprintNeighbor::Boundary(Boundary::Upper)
                    },
                },
                CellFootprintEdge {
                    x: footprint.x,
                    y: footprint.y,
                    triangle_edge: TriangleEdge::UpperLeft(UpperLeftTriangleEdge::Left),
                    neighbor: if let Some(neighbor_x) = footprint.x.checked_sub(1) {
                        CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                            x: neighbor_x,
                            triangle: neighbor_triangle,
                            ..footprint
                        })
                    } else {
                        CellFootprintNeighbor::Boundary(Boundary::Left)
                    },
                },
                CellFootprintEdge {
                    x: footprint.x,
                    y: footprint.y,
                    triangle_edge: TriangleEdge::UpperLeft(UpperLeftTriangleEdge::DownRight),
                    neighbor: CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                        triangle: neighbor_triangle,
                        ..footprint
                    }),
                },
            ],
            Triangle::LowerRight => [
                CellFootprintEdge {
                    x: footprint.x,
                    y: footprint.y,
                    triangle_edge: TriangleEdge::LowerRight(LowerRightTriangleEdge::Down),
                    neighbor: if let Some(neighbor_y) = footprint.y.checked_sub(1) {
                        CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                            y: neighbor_y,
                            triangle: neighbor_triangle,
                            ..footprint
                        })
                    } else {
                        CellFootprintNeighbor::Boundary(Boundary::Lower)
                    },
                },
                CellFootprintEdge {
                    triangle_edge: TriangleEdge::LowerRight(LowerRightTriangleEdge::Right),
                    x: footprint.x,
                    y: footprint.y,
                    neighbor: if footprint.x + 1 < self.num_x_cells() {
                        CellFootprintNeighbor::CellFootprint(CellFootprintIndex {
                            x: footprint.x + 1,
                            triangle: neighbor_triangle,
                            ..footprint
                        })
                    } else {
                        CellFootprintNeighbor::Boundary(Boundary::Right)
                    },
                },
                CellFootprintEdge {
                    triangle_edge: TriangleEdge::LowerRight(LowerRightTriangleEdge::UpLeft),
                    x: footprint.x,
                    y: footprint.y,
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
        [self.num_x_cells(), self.num_y_cells(), Triangle::COUNT]
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
    pub fn vertices(self) -> [VertexFootprintIndex; 3] {
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
    type ArrayIndex = [usize; 3];

    fn to_array_index(self) -> Self::ArrayIndex {
        [self.x, self.y, self.triangle as usize]
    }
}

pub struct CellIndexing {
    cell_footprint_indexing: CellFootprintIndexing,
    num_z_cells_per_column: usize,
}
impl CellIndexing {
    pub fn new(
        cell_footprint_indexing: CellFootprintIndexing,
        num_z_cells_per_column: usize,
    ) -> Self {
        Self {
            cell_footprint_indexing,
            num_z_cells_per_column,
        }
    }

    pub fn cell_footprint_indexing(&self) -> &CellFootprintIndexing {
        &self.cell_footprint_indexing
    }
    pub fn num_z_cells_per_column(&self) -> usize {
        self.num_z_cells_per_column
    }

    pub fn column(&self, footprint: CellFootprintIndex) -> impl Iterator<Item = CellIndex> {
        (0..self.num_z_cells_per_column).map(move |z| CellIndex { footprint, z })
    }
}
impl Indexing for CellIndexing {
    type Index = CellIndex;

    fn shape(&self) -> <<Self as Indexing>::Index as Index>::ArrayIndex {
        [
            self.cell_footprint_indexing.num_x_cells(),
            self.cell_footprint_indexing.num_y_cells(),
            self.num_z_cells_per_column,
            Triangle::COUNT,
        ]
    }

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CellIndex {
    pub footprint: CellFootprintIndex,
    pub z: usize,
}
impl Index for CellIndex {
    type ArrayIndex = [usize; 4];

    fn to_array_index(self) -> Self::ArrayIndex {
        [
            self.footprint.x,
            self.footprint.y,
            self.z,
            self.footprint.triangle as usize,
        ]
    }
}

#[derive(Clone, Copy)]
pub enum CellFootprintNeighbor {
    CellFootprint(CellFootprintIndex),
    Boundary(Boundary),
}

#[derive(Clone, Copy)]
pub enum UpperLeftTriangleEdge {
    Up,
    Left,
    DownRight,
}
#[derive(Clone, Copy)]
pub enum LowerRightTriangleEdge {
    Down,
    Right,
    UpLeft,
}

#[derive(Clone, Copy)]
pub enum TriangleEdge {
    UpperLeft(UpperLeftTriangleEdge),
    LowerRight(LowerRightTriangleEdge),
}

#[derive(Clone, Copy)]
pub struct CellFootprintEdge {
    pub x: usize,
    pub y: usize,
    pub triangle_edge: TriangleEdge,
    pub neighbor: CellFootprintNeighbor,
}

#[derive(Clone, Copy)]
pub enum Boundary {
    Upper,
    Lower,
    Left,
    Right,
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

        assert_eq!(
            VertexFootprintIndex { x: 0, y: 0 }
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
            VertexFootprintIndex { x: 2, y: 0 }
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
            VertexFootprintIndex { x: 0, y: 11 }
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
            VertexFootprintIndex { x: 4, y: 11 }
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
            VertexFootprintIndex { x: 3, y: 26 }
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
            VertexFootprintIndex { x: 3, y: 20 }
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
}
