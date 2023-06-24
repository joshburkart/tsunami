use nalgebra as na;
use ndarray::{self as nd, s};
use numpy::IntoPyArray;
use pyo3::prelude::*;

use crate::{
    fields,
    indexing::{self, Index, Indexing},
    Array1, Array2, Array3, Float, Point2, Point3, UnitVector2, UnitVector3, Vector2, Vector3,
};

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
        let mut centers = Array3::zeros(self.cell_footprint_indexing().shape());
        for cell_footprint_index in indexing::iter_indices(self.cell_footprint_indexing()) {
            let centroid = self.compute_cell_footprint_centroid(cell_footprint_index);
            centers[cell_footprint_index.to_array_index()] = f(centroid[0], centroid[1]);
        }
        centers
    }

    pub fn make_vertex_footprint_array<F: Fn(Float, Float) -> Float>(&self, f: F) -> Array2 {
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
    pub fn new<F: Fn(Float, Float) -> Float>(grid: Grid, terrain_func: &F) -> Self {
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
    // Indexed by [`indexing::VertexFootprintIndexing`]
    spacings: Array2,
}
impl ZLattice {
    pub fn new_lp(grid: &Grid, terrain: &fields::Terrain, height: &fields::HeightField) -> Self {
        Self::new_lp_first_pass(grid, terrain, height)
            .unwrap_or_else(|_| Self::new_lp_second_pass(grid, terrain, height))
    }

    pub fn new_averaging(
        grid: &Grid,
        terrain: &fields::Terrain,
        height: &fields::HeightField,
    ) -> Self {
        let vertex_indexing = grid.vertex_indexing();
        let mut lattice = Array3::zeros(vertex_indexing.shape());
        let mut spacings = Array2::zeros(grid.vertex_footprint_indexing().shape());
        let vertex_footprint_indexing = grid.vertex_footprint_indexing();
        for vertex_footprint_index in indexing::iter_indices(vertex_footprint_indexing) {
            let terrain_height = terrain.vertex_value(vertex_footprint_index);
            let height = mean(
                vertex_footprint_index
                    .adjacent_cells(vertex_footprint_indexing)
                    .map(|cell_footprint_index| height.center_value(cell_footprint_index)),
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
        self.spacings[vertex_footprint_index.to_array_index()]
    }

    pub fn z_spacing_array(&self) -> &Array2 {
        &self.spacings
    }

    pub fn vertex_value(&self, vertex_index: indexing::VertexIndex) -> Float {
        self.lattice[vertex_index.to_array_index()]
    }

    pub fn z_lattice<'py>(&self, py: Python<'py>) -> &'py numpy::PyArray3<Float> {
        self.lattice.clone().into_pyarray(py)
    }

    /// Attempt to find a lattice of points such that the average height of
    /// every triangle is equal to the corresponding height cell center
    /// value.
    fn new_lp_first_pass(
        grid: &Grid,
        terrain: &fields::Terrain,
        height: &fields::HeightField,
    ) -> Result<Self, highs::HighsModelStatus> {
        let mut problem = highs::RowProblem::new();

        // Define variables to be solved for.
        let vertex_variables = (0..grid.vertex_footprint_indexing().len())
            .map(|_| problem.add_column(0., (0.)..Float::INFINITY))
            .collect::<Vec<_>>();

        // Define the principal constraints, which specify that the average of the
        // heights at each cell footprint's three vertices be equal to the
        // height at the cell footprint center.
        for cell_footprint_index in indexing::iter_indices(grid.cell_footprint_indexing()) {
            let center_height = height.center_value(cell_footprint_index);
            problem.add_row(
                center_height..center_height,
                &cell_footprint_index
                    .vertices_right_handed()
                    .into_iter()
                    .map(|vertex_footprint_index| {
                        (
                            vertex_variables[grid
                                .vertex_footprint_indexing()
                                .flatten(vertex_footprint_index)],
                            1. / 3.,
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        }

        // Determine how to compute the mesh Laplacian at a single vertex. TODO: Account
        // for mesh spacing.
        let compute_laplacian =
            |vertex_footprint_index: indexing::VertexFootprintIndex| -> Vec<(highs::Col, Float)> {
                let neighbors = grid
                    .vertex_footprint_indexing()
                    .neighbors(vertex_footprint_index);
                let mut num_neighbors = 0;
                let mut linear_expr = neighbors
                    .map(|neighbor_vertex_footprint_index| {
                        num_neighbors += 1;
                        (
                            vertex_variables[grid
                                .vertex_footprint_indexing()
                                .flatten(neighbor_vertex_footprint_index)],
                            -1.,
                        )
                    })
                    .collect::<Vec<_>>();
                linear_expr.push((
                    vertex_variables[grid
                        .vertex_footprint_indexing()
                        .flatten(vertex_footprint_index)],
                    num_neighbors as Float,
                ));
                linear_expr
            };

        // Add slack variables and associated constraints to effect an absolute value
        // objective function on the mesh Laplacian.
        for vertex_footprint_index in indexing::iter_indices(grid.vertex_footprint_indexing()) {
            let slack_variable = problem.add_column(1., (0.)..Float::INFINITY);
            let laplacian = compute_laplacian(vertex_footprint_index);

            // L + s >= 0  ->  s >= -L
            let mut slack_constraint = laplacian.clone();
            slack_constraint.push((slack_variable, 1.));
            problem.add_row((0.)..Float::INFINITY, slack_constraint);

            // L - s <= 0  ->  s >= L
            let mut slack_constraint = laplacian;
            slack_constraint.push((slack_variable, -1.));
            problem.add_row(Float::NEG_INFINITY..0., slack_constraint);
        }

        // Solve.
        let solved_model = Self::solve_lp(problem);
        if solved_model.status() != highs::HighsModelStatus::Optimal {
            return Err(solved_model.status());
        }
        Ok(Self::extract_lp_solution(
            solved_model.get_solution(),
            vertex_variables,
            grid,
            terrain,
        ))
    }

    /// Find a lattice of points such that the average height of every triangle
    /// is as close as possible in an L1 sense to the corresponding height
    /// cell center value.
    fn new_lp_second_pass(
        grid: &Grid,
        terrain: &fields::Terrain,
        height: &fields::HeightField,
    ) -> Self {
        let mut problem = highs::RowProblem::new();

        // Define variables to be solved for.
        let vertex_variables = (0..grid.vertex_footprint_indexing().len())
            .map(|_| problem.add_column(0., (0.)..Float::INFINITY))
            .collect::<Vec<_>>();

        // Define the optimization objective, which specifies that the average of the
        // heights at each cell footprint's three vertices be as close as
        // possible in an L1 sense to the height at the cell footprint center.
        for cell_footprint_index in indexing::iter_indices(grid.cell_footprint_indexing()) {
            // Delta = 1/3 * (z1 + z2 + z3) - h
            let center_height = height.center_value(cell_footprint_index);
            let center_height_val = problem.add_column(0., center_height..center_height);
            let mut diff_height = cell_footprint_index
                .vertices_right_handed()
                .into_iter()
                .map(|vertex_footprint_index| {
                    (
                        vertex_variables[grid
                            .vertex_footprint_indexing()
                            .flatten(vertex_footprint_index)],
                        1. / 3.,
                    )
                })
                .collect::<Vec<_>>();
            diff_height.push((center_height_val, -1.));

            // Delta + s >= 0  ->  s >= -Delta
            let slack_variable = problem.add_column(1., (0.)..Float::INFINITY);
            let mut slack_constraint = diff_height.clone();
            slack_constraint.push((slack_variable, 1.));
            problem.add_row((0.)..Float::INFINITY, slack_constraint);

            // Delta - s <= 0  ->  s >= Delta
            let mut slack_constraint = diff_height;
            slack_constraint.push((slack_variable, -1.));
            problem.add_row(Float::NEG_INFINITY..(0.), slack_constraint);
        }

        let solved_model = Self::solve_lp(problem);
        if solved_model.status() != highs::HighsModelStatus::Optimal {
            panic!(
                "Second pass was infeasible, should be impossible: {:?}",
                solved_model.status()
            );
        }
        Self::extract_lp_solution(solved_model.get_solution(), vertex_variables, grid, terrain)
    }

    fn solve_lp(problem: highs::RowProblem) -> highs::SolvedModel {
        let mut model = problem.optimise(highs::Sense::Minimise);
        model.set_option("parallel", "on");
        // model.set_option("time_limit", 10.);
        model.solve()
    }

    fn extract_lp_solution(
        solution: highs::Solution,
        vertex_variables: Vec<highs::Col>,
        grid: &Grid,
        terrain: &fields::Terrain,
    ) -> Self {
        // Unpack the solution into a lattice. Evenly distribute z points across each
        // column of cells.
        let num_z_points = grid.vertex_indexing().num_z_points();
        let mut lattice = Array3::zeros((
            grid.vertex_indexing.num_x_points(),
            grid.vertex_indexing.num_y_points(),
            num_z_points,
        ));
        let mut spacings = Array2::zeros((
            grid.vertex_indexing.num_x_points(),
            grid.vertex_indexing.num_y_points(),
        ));
        for (vertex_footprint_flat_index, _) in vertex_variables.iter().enumerate() {
            let vertex_footprint_index = grid
                .vertex_footprint_indexing()
                .unflatten(vertex_footprint_flat_index);
            let height = solution.columns()[vertex_footprint_flat_index];
            let spacing = height / (num_z_points as Float - 1.);
            spacings[vertex_footprint_index.to_array_index()] = spacing;
            for k in 0..num_z_points {
                let vertex_index = indexing::VertexIndex {
                    footprint: vertex_footprint_index,
                    z: k,
                };
                lattice[vertex_index.to_array_index()] =
                    terrain.vertex_value(vertex_footprint_index) + spacing * k as Float;
            }
        }
        Self { lattice, spacings }
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
    pub fn new(static_geometry: StaticGeometry, height: &fields::HeightField) -> Self {
        let z_lattice =
            ZLattice::new_averaging(static_geometry.grid(), static_geometry.terrain(), &height);
        let mut cells = nd::Array::default(static_geometry.grid().cell_indexing().shape());
        for (cell_index, cell) in Self::iter_cells(&static_geometry, &z_lattice) {
            cells[cell_index.to_array_index()] = cell;
        }
        Self {
            static_geometry,
            z_lattice,
            cells,
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

    pub fn cell(&self, index: indexing::CellIndex) -> &Cell {
        &self.cells[index.to_array_index()]
    }

    pub fn make_cell_array<V: Default, F: Fn(Float, Float, Float) -> V>(
        &self,
        f: F,
    ) -> nd::Array4<V> {
        let mut cells = nd::Array4::<V>::default(self.grid().cell_indexing().shape());
        for cell_index in indexing::iter_indices(self.grid().cell_indexing()) {
            let centroid = self.cell(cell_index).centroid;
            cells[cell_index.to_array_index()] = f(centroid.x, centroid.y, centroid.z);
        }
        cells
    }

    pub fn make_vertex_array<V: Default, F: Fn(Float, Float, Float) -> V>(
        &self,
        f: F,
    ) -> nd::Array3<V> {
        let mut vertices = nd::Array3::<V>::default(self.grid().vertex_indexing().shape());
        for indexing::VertexIndex {
            footprint: indexing::VertexFootprintIndex { x, y },
            z,
        } in indexing::iter_indices(self.grid().vertex_indexing())
        {
            vertices[[x, y, z]] = f(
                self.grid().x_axis.vertices[x],
                self.grid().y_axis.vertices[y],
                self.z_lattice.lattice[[x, y, z]],
            );
        }
        vertices
    }

    fn iter_cells<'a>(
        static_geometry: &'a StaticGeometry,
        z_lattice: &'a ZLattice,
    ) -> impl Iterator<Item = (indexing::CellIndex, Cell)> + 'a {
        let grid = static_geometry.grid();
        let cell_footprint_indexing = grid.cell_footprint_indexing();
        let cell_indexing = grid.cell_indexing();
        indexing::iter_indices(static_geometry.grid().cell_footprint_indexing())
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
                                volume: Self::compute_volume(
                                    static_geometry,
                                    z_lattice,
                                    cell_index,
                                ),
                                centroid: Self::compute_centroid(
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
                                classification: if 0 < cell_index.footprint.x
                                    && cell_index.footprint.x
                                        < cell_footprint_indexing.num_x_cells() - 1
                                    && 0 < cell_index.footprint.y
                                    && cell_index.footprint.y
                                        < cell_footprint_indexing.num_y_cells() - 1
                                    && 0 < cell_index.z
                                    && cell_index.z < cell_indexing.num_z_cells() - 1
                                {
                                    CellClassification::Interior
                                } else {
                                    CellClassification::Boundary
                                },
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
        static_geometry.grid().footprint_area()
            * cell_index
                .footprint
                .vertices_right_handed()
                .into_iter()
                .map(|vertex_footprint_index| z_lattice.z_spacing(vertex_footprint_index))
                .sum::<Float>()
            / 3.
    }

    fn compute_centroid(
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
            Self::compute_vertical_face(
                static_geometry,
                z_lattice,
                cell_index,
                cell_footprint_pairs[0],
            ),
            Self::compute_vertical_face(
                static_geometry,
                z_lattice,
                cell_index,
                cell_footprint_pairs[1],
            ),
            Self::compute_vertical_face(
                static_geometry,
                z_lattice,
                cell_index,
                cell_footprint_pairs[2],
            ),
            Self::compute_horizontal_face(
                static_geometry,
                z_lattice,
                cell_index,
                HorizontalSide::Up,
            ),
            Self::compute_horizontal_face(
                static_geometry,
                z_lattice,
                cell_index,
                HorizontalSide::Down,
            ),
        ]
    }

    fn compute_horizontal_face(
        static_geometry: &StaticGeometry,
        z_lattice: &ZLattice,
        cell_index: indexing::CellIndex,
        horizontal_side: HorizontalSide,
    ) -> CellFace {
        let vertex_footprints = cell_index.footprint.vertices_right_handed();
        let vertices_z = match horizontal_side {
            HorizontalSide::Up => cell_index.z + 1,
            HorizontalSide::Down => cell_index.z,
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
        let outward_normal = match horizontal_side {
            HorizontalSide::Up => normal,
            HorizontalSide::Down => -normal,
        };

        let neighbor = match horizontal_side {
            HorizontalSide::Up => {
                let z_plus_1 = cell_index.z + 1;
                if z_plus_1 >= static_geometry.grid().cell_indexing().num_z_cells() {
                    indexing::CellNeighbor::VerticalBoundary(indexing::VerticalBoundary::Surface)
                } else {
                    indexing::CellNeighbor::Interior(indexing::CellIndex {
                        z: z_plus_1,
                        ..cell_index
                    })
                }
            }
            HorizontalSide::Down => {
                if let Some(z_minus_1) = cell_index.z.checked_sub(1) {
                    indexing::CellNeighbor::Interior(indexing::CellIndex {
                        z: z_minus_1,
                        ..cell_index
                    })
                } else {
                    indexing::CellNeighbor::VerticalBoundary(indexing::VerticalBoundary::Floor)
                }
            }
        };

        CellFace {
            data: CellFaceData::Horizontal(CellHorizontalFace {
                cell_index,
                area,
                outward_normal,
                direction: horizontal_side,
                neighbor,
            }),
        }
    }

    fn compute_vertical_face(
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

        CellFace {
            data: CellFaceData::Vertical(CellVerticalFace {
                area,
                cell_index,
                cell_footprint_edge,
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
    pub volume: Float,
    pub centroid: Point3,
    pub faces: [CellFace; 5],
    pub classification: CellClassification,
}
// Needed so we can have `nd::Array`s of `Cell`s.
impl Default for Cell {
    fn default() -> Self {
        Self {
            volume: Default::default(),
            centroid: Default::default(),
            faces: [
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
            ],
            classification: CellClassification::Interior,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CellClassification {
    Boundary,
    Interior,
}

#[derive(Clone, Debug)]
pub struct CellFootprintEdge {
    pub cell_footprint_pair: indexing::CellFootprintPair,
    pub outward_normal: UnitVector2,
    pub length: Float,
    pub vertex_footprints: [indexing::VertexFootprintIndex; 2],
}

#[derive(Clone, Debug)]
struct CellVerticalFace {
    pub cell_index: indexing::CellIndex,
    pub area: Float,
    cell_footprint_edge: CellFootprintEdge,
}

#[derive(Clone, Debug)]
struct CellHorizontalFace {
    pub cell_index: indexing::CellIndex,
    pub area: Float,
    outward_normal: UnitVector3,
    pub direction: HorizontalSide,
    pub neighbor: indexing::CellNeighbor,
}

#[derive(Clone, Copy, Debug)]
enum HorizontalSide {
    Up,
    Down,
}

#[derive(Clone, Debug)]
enum CellFaceData {
    Vertical(CellVerticalFace),
    Horizontal(CellHorizontalFace),
}
#[derive(Clone, Debug)]
pub struct CellFace {
    data: CellFaceData,
}
// Needed so we can have `nd::Array`s of `Cell`s.
impl Default for CellFace {
    fn default() -> Self {
        CellFace {
            data: CellFaceData::Horizontal(CellHorizontalFace {
                cell_index: indexing::CellIndex {
                    footprint: indexing::CellFootprintIndex {
                        x: 0,
                        y: 0,
                        triangle: indexing::Triangle::LowerRight,
                    },
                    z: 0,
                },
                area: 0.,
                outward_normal: Vector3::x_axis(),
                direction: HorizontalSide::Up,
                neighbor: indexing::CellNeighbor::HorizontalBoundary(
                    indexing::HorizontalBoundary::Left,
                ),
            }),
        }
    }
}
impl CellFace {
    pub fn area(&self) -> Float {
        match &self.data {
            CellFaceData::Vertical(vertical_face) => vertical_face.area,
            CellFaceData::Horizontal(horizontal_face) => horizontal_face.area,
        }
    }

    pub fn neighbor(&self) -> indexing::CellNeighbor {
        match &self.data {
            CellFaceData::Vertical(vertical_face) => match vertical_face
                .cell_footprint_edge
                .cell_footprint_pair
                .neighbor
            {
                indexing::CellFootprintNeighbor::CellFootprint(footprint) => {
                    indexing::CellNeighbor::Interior(indexing::CellIndex {
                        footprint,
                        z: vertical_face.cell_index.z,
                    })
                }
                indexing::CellFootprintNeighbor::Boundary(boundary) => {
                    indexing::CellNeighbor::HorizontalBoundary(boundary)
                }
            },
            CellFaceData::Horizontal(horizontal_face) => horizontal_face.neighbor,
        }
    }

    pub fn outward_normal(&self) -> UnitVector3 {
        match &self.data {
            CellFaceData::Vertical(vertical_face) => {
                let normal_2d = vertical_face.cell_footprint_edge.outward_normal;
                UnitVector3::new_unchecked(Vector3::new(normal_2d.x, normal_2d.y, 0.))
            }
            CellFaceData::Horizontal(horizontal_face) => horizontal_face.outward_normal,
        }
    }

    pub fn vertices(&self) -> CellFaceVertices {
        match &self.data {
            CellFaceData::Vertical(vertical_face) => CellFaceVertices::Vertical([
                indexing::VertexIndex {
                    footprint: vertical_face.cell_footprint_edge.vertex_footprints[0],
                    z: vertical_face.cell_index.z,
                },
                indexing::VertexIndex {
                    footprint: vertical_face.cell_footprint_edge.vertex_footprints[0],
                    z: vertical_face.cell_index.z + 1,
                },
                indexing::VertexIndex {
                    footprint: vertical_face.cell_footprint_edge.vertex_footprints[1],
                    z: vertical_face.cell_index.z,
                },
                indexing::VertexIndex {
                    footprint: vertical_face.cell_footprint_edge.vertex_footprints[1],
                    z: vertical_face.cell_index.z + 1,
                },
            ]),
            CellFaceData::Horizontal(horizontal_face) => {
                let z = horizontal_face.cell_index.z
                    + match horizontal_face.direction {
                        HorizontalSide::Up => 1,
                        HorizontalSide::Down => 0,
                    };
                let vertex_footprints =
                    horizontal_face.cell_index.footprint.vertices_right_handed();
                CellFaceVertices::Horizontal([
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
                ])
            }
        }
    }
}

pub enum CellFaceVertices {
    Vertical([indexing::VertexIndex; 4]),
    Horizontal([indexing::VertexIndex; 3]),
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
    use crate::Matrix3;

    #[test]
    fn test_construct_z_lattice_zero_height() {
        let grid = Grid::new(Axis::new(0., 1., 29), Axis::new(0., 1., 32), 10);
        let height = fields::HeightField::new(&grid, |_, _| 0.);
        let static_geometry = StaticGeometry::new(grid, &|x, y| 10. * x * y);
        let dynamic_geometry = DynamicGeometry::new(static_geometry, &height);

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
        let height = fields::HeightField::new(&grid, |_, _| 7.3);

        let static_geometry = StaticGeometry::new(grid, &|_, _| 0.);
        let dynamic_geometry = DynamicGeometry::new(static_geometry, &height);

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
        let expected_horizontal_centroid = cell_index
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
        assert_relative_eq!(expected_horizontal_centroid.x, cell.centroid.x);
        assert_relative_eq!(expected_horizontal_centroid.y, cell.centroid.y);
    }

    #[test]
    fn test_construct_z_lattice_grade() {
        let grid = Grid::new(Axis::new(0., 1., 60), Axis::new(0., 1., 31), 10);
        let terrain = fields::Terrain::new(&grid, &|_, _| 0.);
        let mut height = fields::HeightField::new(&grid, |_, _| 0.);
        for cell_footprint_index in indexing::iter_indices(grid.cell_footprint_indexing()) {
            let centroid = grid.compute_cell_footprint_centroid(cell_footprint_index);
            *height.center_value_mut(cell_footprint_index) = 1. + centroid[0] + 2. * centroid[1];
        }

        let z_lattice = ZLattice::new_averaging(&grid, &terrain, &height);

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

    #[test]
    fn test_compute_shear_flat_geometry() {
        let dynamic_geometry = make_flat_geometry();

        // Zero velocity.
        {
            let velocity = fields::VelocityField::zeros(dynamic_geometry.grid().cell_indexing());
            let shear = velocity.compute_shear(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                shear,
                fields::ShearField::zeros(dynamic_geometry.grid().cell_indexing()),
                epsilon = 1e-5
            );
        }

        // Constant velocity.
        {
            let velocity =
                fields::VelocityField::new(&dynamic_geometry, |_, _, _| Vector3::new(2., -1., 7.));
            let shear = velocity.compute_shear(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                shear,
                fields::ShearField::zeros(dynamic_geometry.grid().cell_indexing()),
                epsilon = 1e-5
            );
        }

        // Linearly increasing velocity.
        {
            let velocity = fields::VelocityField::new(&dynamic_geometry, |x, _, _| {
                Vector3::new(2. * x, 0., 0.)
            });
            let shear = velocity.compute_shear(&dynamic_geometry);
            let expected_shear = fields::ShearField::new(&dynamic_geometry, |_, _, _| {
                Matrix3::new(2., 0., 0., 0., 0., 0., 0., 0., 0.)
            });

            for cell_index in indexing::iter_indices(dynamic_geometry.grid().cell_indexing()) {
                if let CellClassification::Interior =
                    dynamic_geometry.cell(cell_index).classification
                {
                    approx::assert_abs_diff_eq!(
                        shear.cell_value(cell_index),
                        expected_shear.cell_value(cell_index),
                        epsilon = 1e-5
                    );
                }
            }
        }
    }
    #[test]
    fn test_compute_laplacian_flat_geometry() {
        let dynamic_geometry = make_flat_geometry();

        // Zero velocity.
        {
            let velocity = fields::VelocityField::zeros(dynamic_geometry.grid().cell_indexing());
            let laplacian = velocity.compute_laplacian(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                laplacian,
                fields::VelocityField::zeros(dynamic_geometry.grid().cell_indexing()),
                epsilon = 1e-5
            );
        }

        // Constant velocity.
        {
            let velocity =
                fields::VelocityField::new(&dynamic_geometry, |_, _, _| Vector3::new(2., -1., 7.));
            let laplacian = velocity.compute_laplacian(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                laplacian,
                fields::VelocityField::zeros(dynamic_geometry.grid().cell_indexing()),
                epsilon = 1e-5
            );
        }

        // Linearly increasing velocity.
        {
            let velocity = fields::VelocityField::new(&dynamic_geometry, |x, y, z| {
                Vector3::new(2. * x - z, -z, y)
            });
            let laplacian = velocity.compute_laplacian(&dynamic_geometry);
            let expected_laplacian =
                fields::VelocityField::zeros(dynamic_geometry.grid().cell_indexing());

            for cell_index in [
                indexing::CellIndex {
                    footprint: indexing::CellFootprintIndex {
                        x: 2,
                        y: 2,
                        triangle: indexing::Triangle::LowerRight,
                    },
                    z: 2,
                },
                indexing::CellIndex {
                    footprint: indexing::CellFootprintIndex {
                        x: 2,
                        y: 2,
                        triangle: indexing::Triangle::UpperLeft,
                    },
                    z: 3,
                },
            ] {
                if let CellClassification::Interior =
                    dynamic_geometry.cell(cell_index).classification
                {
                    approx::assert_abs_diff_eq!(
                        laplacian.cell_value(cell_index),
                        expected_laplacian.cell_value(cell_index),
                        epsilon = 1e-5
                    );
                }
            }
        }
    }

    #[test]
    fn test_compute_gradient_flat_geometry() {
        test_compute_gradient_impl(&make_flat_geometry());
    }

    #[test]
    fn test_compute_gradient_ramp_geometry() {
        test_compute_gradient_impl(&make_ramp_geometry(false));
        test_compute_gradient_impl(&make_ramp_geometry(true));
    }

    fn test_compute_gradient_impl(dynamic_geometry: &DynamicGeometry) {
        // Zero pressure.
        {
            let pressure = fields::PressureField::zeros(dynamic_geometry.grid().vertex_indexing());
            let gradient = pressure.compute_gradient(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                gradient,
                fields::VelocityField::zeros(dynamic_geometry.grid().cell_indexing()),
                epsilon = 1e-5
            );
        }

        // Constant pressure.
        {
            let pressure = fields::PressureField::new(&dynamic_geometry, |_, _, _| 1.9);
            let gradient = pressure.compute_gradient(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                gradient,
                fields::VelocityField::zeros(dynamic_geometry.grid().cell_indexing()),
                epsilon = 1e-5
            );
        }

        // Linearly rising pressure.
        {
            let pressure = fields::PressureField::new(&dynamic_geometry, |_, _, z| 2. * z);
            let gradient = pressure.compute_gradient(&dynamic_geometry);
            approx::assert_abs_diff_eq!(
                gradient,
                fields::VelocityField::new(&dynamic_geometry, |_, _, _| Vector3::new(0., 0., 2.)),
                epsilon = 1e-1
            );
        }
    }

    fn make_flat_geometry() -> DynamicGeometry {
        let grid = Grid::new(Axis::new(0., 1., 8), Axis::new(0., 1., 8), 8);
        let height = fields::HeightField::new(&grid, |_, _| 7.3);

        let static_geometry = StaticGeometry::new(grid, &|_, _| 0.);
        DynamicGeometry::new(static_geometry, &height)
    }

    fn make_ramp_geometry(swap_xy: bool) -> DynamicGeometry {
        let mut x_axis = Axis::new(0., 1., 300);
        let mut y_axis = Axis::new(0., 0.001, 1);
        if swap_xy {
            (x_axis, y_axis) = (y_axis, x_axis);
        }
        let grid = Grid::new(x_axis, y_axis, 100);
        let height = fields::HeightField::new(&grid, |x, y| 7.3 * x + y + 1.);

        let static_geometry = StaticGeometry::new(grid, &|x, y| 0.1 * (x + y));
        DynamicGeometry::new(static_geometry, &height)
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
