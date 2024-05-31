#![allow(non_snake_case)]

use crate::{ComplexFloat, Float};
use ndarray as nd;
pub mod fourier;
pub mod ylm;

pub trait Basis {
    type SpectralScalarField;
    type SpectralVectorField;

    fn scalar_grid_size(&self) -> usize;
    fn vector_grid_size(&self) -> usize {
        self.scalar_grid_size() * 2
    }

    fn axes(&self) -> [nd::Array1<Float>; 2];

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
        let vector = nd::Array3::build_uninit((xs.len(), ys.len(), 2), |mut vector| {
            for (i, &x) in xs.iter().enumerate() {
                for (j, &y) in ys.iter().enumerate() {
                    let value = f(x, y);
                    vector[[i, j, 0]].write(value[0]);
                    vector[[i, j, 1]].write(value[1]);
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

    fn gradient(&self, spectral: &Self::SpectralScalarField) -> Self::SpectralVectorField;
    fn divergence(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralScalarField;
    fn vector_laplacian(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralVectorField;
    fn vector_advection(
        &self,
        grid: &nd::Array3<Float>,
        spectral: &Self::SpectralVectorField,
    ) -> Self::SpectralVectorField;
}

trait FftDimension: nd::Dimension {
    fn change_axis_0(shape: Self, size: usize) -> Self;
}

impl FftDimension for nd::Dim<[usize; 2]> {
    fn change_axis_0(shape: Self, size: usize) -> Self {
        nd::Dim([size, shape[1]])
    }
}
impl FftDimension for nd::Dim<[usize; 3]> {
    fn change_axis_0(shape: Self, size: usize) -> Self {
        nd::Dim([size, shape[1], shape[2]])
    }
}
impl FftDimension for nd::Dim<[usize; 4]> {
    fn change_axis_0(shape: Self, size: usize) -> Self {
        nd::Dim([size, shape[1], shape[2], shape[3]])
    }
}
