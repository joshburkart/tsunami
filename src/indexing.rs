use strum::{EnumCount, IntoEnumIterator};

#[derive(Copy, Clone, Debug, PartialEq, Eq, strum_macros::EnumIter, strum_macros::EnumCount)]
pub enum Triangle {
    UpperLeft = 0,
    LowerRight = 1,
}

pub trait Indexing {
    type Index: Copy + std::fmt::Debug + PartialEq + Eq;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn flatten(&self, index: Self::Index) -> usize;
    fn unflatten(&self, flat_index: usize) -> Self::Index;
}

pub fn iter_indices<I: Indexing>(indexing: &I) -> impl Iterator<Item = I::Index> + '_ {
    (0..indexing.len()).map(|flat_index| indexing.unflatten(flat_index))
}

#[derive(Clone, Copy)]
pub struct VertexFootprintIndexing {
    pub num_x_points: usize,
    pub num_y_points: usize,
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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct VertexFootprintIndex {
    pub x: usize,
    pub y: usize,
}
impl VertexFootprintIndex {
    pub fn cells(
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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
                .cells(&vertex_footprint_indexing)
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
                .cells(&vertex_footprint_indexing)
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
                .cells(&vertex_footprint_indexing)
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
                .cells(&vertex_footprint_indexing)
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
                .cells(&vertex_footprint_indexing)
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
                .cells(&vertex_footprint_indexing)
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
