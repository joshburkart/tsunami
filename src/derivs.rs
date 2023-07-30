use crate::{
    fields::{BoundaryCondition, BoundaryConditions, Value, VolField},
    geom,
    indexing::{self, IntoIndexIterator},
    Float, Point3, UnitVector3,
};

pub trait DifferentialOpComputer<V: Value, DV: Value> {
    fn compute_interior_face_value(
        &self,
        cell: &geom::Cell,
        neighbor_cell: &geom::Cell,
        outward_normal: UnitVector3,
        face_centroid: Point3,
    ) -> DV;

    fn compute_boundary_face_value(
        &self,
        cell: &geom::Cell,
        block_paired_cell: &geom::Cell,
        outward_normal: UnitVector3,
        face_centroid: Point3,
        boundary_condition: BoundaryCondition<V>,
    ) -> DV;
}

pub fn compute_field_differential<V: Value, DV: Value, D: DifferentialOpComputer<V, DV>>(
    dynamic_geometry: &geom::DynamicGeometry,
    boundary_conditions: &BoundaryConditions<V>,
    computer: D,
) -> VolField<DV> {
    let cell_indexing = dynamic_geometry.grid().cell_indexing();
    let mut differential_field = VolField::<DV>::zeros(cell_indexing);
    for cell_index in cell_indexing.iter() {
        *differential_field.cell_value_mut(cell_index) =
            compute_cell_differential(dynamic_geometry, &computer, cell_index, boundary_conditions);
    }
    differential_field
}

fn compute_cell_differential<V: Value, DV: Value, D: DifferentialOpComputer<V, DV>>(
    dynamic_geometry: &geom::DynamicGeometry,
    computer: &D,
    cell_index: indexing::CellIndex,
    boundary_conditions: &BoundaryConditions<V>,
) -> DV
where
    DV: std::ops::Mul<Float, Output = DV> + std::ops::Div<Float, Output = DV>,
{
    let cell = dynamic_geometry.cell(cell_index);
    let mut face_accumulator = DV::zero();

    for face in &cell.faces {
        let compute_boundary_face_value = |boundary_condition| {
            let block_paired_cell = dynamic_geometry.cell(cell_index.flip());
            computer.compute_boundary_face_value(
                cell,
                block_paired_cell,
                face.outward_normal(),
                face.centroid(),
                boundary_condition,
            )
        };
        let face_value = match face.neighbor() {
            indexing::CellNeighbor::Cell(neighbor_cell_index) => computer
                .compute_interior_face_value(
                    cell,
                    dynamic_geometry.cell(neighbor_cell_index),
                    face.outward_normal(),
                    face.centroid(),
                ),
            indexing::CellNeighbor::XBoundary(boundary) => match boundary {
                indexing::Boundary::Lower => {
                    compute_boundary_face_value(boundary_conditions.horiz.x.lower)
                }
                indexing::Boundary::Upper => {
                    compute_boundary_face_value(boundary_conditions.horiz.x.upper)
                }
            },
            indexing::CellNeighbor::YBoundary(boundary) => match boundary {
                indexing::Boundary::Lower => {
                    compute_boundary_face_value(boundary_conditions.horiz.y.lower)
                }
                indexing::Boundary::Upper => {
                    compute_boundary_face_value(boundary_conditions.horiz.y.upper)
                }
            },
            indexing::CellNeighbor::ZBoundary(boundary) => match boundary {
                indexing::Boundary::Lower => computer.compute_boundary_face_value(
                    cell,
                    cell,
                    face.outward_normal(),
                    face.centroid(),
                    boundary_conditions
                        .z
                        .lower
                        .boundary_condition(cell_index.footprint),
                ),
                indexing::Boundary::Upper => computer.compute_boundary_face_value(
                    cell,
                    cell,
                    face.outward_normal(),
                    face.centroid(),
                    boundary_conditions
                        .z
                        .upper
                        .boundary_condition(cell_index.footprint),
                ),
            },
        };
        face_accumulator += face_value * face.area();
    }
    face_accumulator / cell.volume
}
