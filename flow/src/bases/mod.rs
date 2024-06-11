#![allow(non_snake_case)]

use ndarray as nd;

use crate::{ComplexFloat, Float};
pub mod fourier;
pub mod ylm;

pub trait Basis {
    type SpectralScalarField;
    type SpectralVectorField;

    fn scalar_spectral_size(&self) -> usize;
    fn vector_spectral_size(&self) -> usize {
        self.scalar_spectral_size() * 2
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

    fn gradient(&self, spectral: &Self::SpectralScalarField) -> Self::SpectralVectorField;
    fn divergence(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralScalarField;
    fn vector_laplacian(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralVectorField;
    fn vector_advection(
        &self,
        grid: &nd::Array3<Float>,
        spectral: &Self::SpectralVectorField,
    ) -> Self::SpectralVectorField;
    fn z_cross(&self, spectral: &nd::Array3<Float>) -> nd::Array3<Float>;
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
