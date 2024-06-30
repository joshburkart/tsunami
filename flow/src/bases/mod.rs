#![allow(non_snake_case)]

use ndarray as nd;

use crate::{ComplexFloat, Float};
pub mod fourier;
pub mod ylm;

pub trait Basis: Sync + Send {
    type SpectralScalarField: Sync + Send;
    type SpectralVectorField: Sync + Send;

    fn scalar_spectral_size(&self) -> usize;
    fn vector_spectral_size(&self) -> usize {
        self.scalar_spectral_size() * 2
    }

    fn lengths(&self) -> [Float; 2];
    fn axes(&self) -> [&nd::Array1<Float>; 2];

    fn make_random_points(&self) -> nd::Array2<Float>;

    fn make_scalar<F: Fn(Float, Float) -> Float>(&self, f: F) -> nd::Array2<Float> {
        let [xs, ys] = self.axes();
        let scalar = nd::Array2::build_uninit((xs.len(), ys.len()), |mut scalar| {
            for (i, &x) in xs.iter().enumerate() {
                for (j, &y) in ys.iter().enumerate() {
                    scalar[[i, j]].write(f(x, y));
                }
            }
        });
        unsafe { scalar.assume_init() }
    }
    fn make_vector<F: Fn(Float, Float) -> [Float; 2]>(&self, f: F) -> nd::Array3<Float> {
        let [xs, ys] = self.axes();
        let vector = nd::Array3::build_uninit((2, xs.len(), ys.len()), |mut vector| {
            for (i, &x) in xs.iter().enumerate() {
                for (j, &y) in ys.iter().enumerate() {
                    let value = f(x, y);
                    vector[[0, i, j]].write(value[0]);
                    vector[[1, i, j]].write(value[1]);
                }
            }
        });
        unsafe { vector.assume_init() }
    }

    fn scalar_from_slice(&self, slice: &[ComplexFloat]) -> Self::SpectralScalarField;
    fn vector_from_slice(&self, slice: &[ComplexFloat]) -> Self::SpectralVectorField;

    fn scalar_to_slice(&self, spectral: &Self::SpectralScalarField, slice: &mut [ComplexFloat]);
    fn vector_to_slice(&self, spectral: &Self::SpectralVectorField, slice: &mut [ComplexFloat]);

    fn scalar_to_grid(&self, spectral: &Self::SpectralScalarField) -> nd::Array2<Float>;
    fn vector_to_grid(&self, spectral: &Self::SpectralVectorField) -> nd::Array3<Float>;

    fn scalar_to_spectral(&self, grid: &nd::Array2<Float>) -> Self::SpectralScalarField;
    fn vector_to_spectral(&self, grid: &nd::Array3<Float>) -> Self::SpectralVectorField;

    fn scalar_to_points(
        &self,
        grid: &nd::Array2<Float>,
        points: nd::ArrayView2<'_, Float>,
    ) -> nd::Array1<Float>;
    fn velocity_to_points(
        &self,
        grid: &nd::Array3<Float>,
        points: nd::ArrayView2<'_, Float>,
    ) -> nd::Array2<Float>;

    fn gradient(&self, spectral: &Self::SpectralScalarField) -> Self::SpectralVectorField;
    fn divergence(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralScalarField;
    fn vector_laplacian(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralVectorField;
    fn vector_advection(
        &self,
        grid: &nd::Array3<Float>,
        spectral: &Self::SpectralVectorField,
    ) -> Self::SpectralVectorField;
    fn z_cross(&self, grid: &nd::Array3<Float>) -> nd::Array3<Float>;
    fn tidal_force(&self, lunar_distance: Float, lunar_phase: Float) -> Self::SpectralVectorField;
}

trait FftDimension: nd::Dimension {
    fn change_last_axis(shape: Self, size: usize) -> Self;
}

impl FftDimension for nd::Dim<[usize; 2]> {
    fn change_last_axis(shape: Self, size: usize) -> Self {
        nd::Dim([shape[0], size])
    }
}
impl FftDimension for nd::Dim<[usize; 3]> {
    fn change_last_axis(shape: Self, size: usize) -> Self {
        nd::Dim([shape[0], shape[1], size])
    }
}
impl FftDimension for nd::Dim<[usize; 4]> {
    fn change_last_axis(shape: Self, size: usize) -> Self {
        nd::Dim([shape[0], shape[1], shape[2], size])
    }
}

pub struct GridLine {
    pub index: usize,
    pub value: Float,
}

type GridSearchResult = [[GridLine; 2]; 2];

fn periodic_grid_search<B: Basis>(basis: &B, point: nd::ArrayView1<'_, Float>) -> GridSearchResult {
    let axes = basis.axes();
    let lengths = basis.lengths();
    let axis_starts = [axes[0][[0]], axes[1][[0]]];

    let search_axis = |axis_index: usize| {
        let val = (point[[axis_index]] - axis_starts[axis_index]).rem_euclid(lengths[axis_index])
            + axis_starts[axis_index];
        let mapping = |i: usize| {
            [
                i.checked_sub(1).unwrap_or(axes[axis_index].len() - 1),
                i % axes[axis_index].len(),
            ]
        };
        axes[axis_index]
            .as_slice()
            .unwrap()
            .binary_search_by(|other| other.total_cmp(&val))
            .map(mapping)
            .unwrap_or_else(mapping)
    };
    let make_bracket = |axis_index: usize| {
        let [index_1, index_2] = search_axis(axis_index);
        let value_1 = axes[axis_index][[index_1]];
        let value_2 = axes[axis_index][[index_2]];
        [
            GridLine {
                index: index_1,
                value: value_1,
            },
            GridLine {
                index: index_2,
                value: value_2,
            },
        ]
    };
    [make_bracket(0), make_bracket(1)]
}

#[inline(always)]
fn periodic_bilinear_interpolate(
    point: nd::ArrayView1<Float>,
    grid_search_result: &GridSearchResult,
    grid: nd::ArrayView2<'_, Float>,
    lengths: &[Float; 2],
) -> Float {
    let [x_bracket, y_bracket] = grid_search_result;

    let left_vec = nd::array![
        (x_bracket[1].value - point[[0]]).rem_euclid(lengths[0]),
        (point[[0]] - x_bracket[0].value).rem_euclid(lengths[0])
    ];
    let right_vec = nd::array![
        (y_bracket[1].value - point[[1]]).rem_euclid(lengths[1]),
        (point[[1]] - y_bracket[0].value).rem_euclid(lengths[1])
    ];

    let v11 = grid[[x_bracket[0].index, y_bracket[0].index]];
    let v12 = grid[[x_bracket[0].index, y_bracket[1].index]];
    let v21 = grid[[x_bracket[1].index, y_bracket[0].index]];
    let v22 = grid[[x_bracket[1].index, y_bracket[1].index]];
    let matrix = nd::array![[v11, v12], [v21, v22]];

    left_vec.dot(&matrix.dot(&right_vec))
        / (x_bracket[1].value - x_bracket[0].value).rem_euclid(lengths[0])
        / (y_bracket[1].value - y_bracket[0].value).rem_euclid(lengths[1])
}
