use ndarray as nd;

use crate::{
    bases::{Basis, FftDimension},
    float_consts, ComplexFloat, Float, RawFloatData,
};

pub struct RectangularPeriodicBasis {
    pub num_points: [usize; 2],

    pub lengths: [Float; 2],

    wavenumbers: nd::Array3<ComplexFloat>,
    wavenumbers_sq: nd::Array3<Float>,

    rfft_handler: std::sync::Mutex<ndrustfft::R2cFftHandler<Float>>,
    fft_handler: std::sync::Mutex<ndrustfft::FftHandler<Float>>,
    rfft_output_size: usize,
}

impl RectangularPeriodicBasis {
    pub fn new(num_points: [usize; 2], lengths: [Float; 2]) -> Self {
        let rfft_handler = std::sync::Mutex::new(ndrustfft::R2cFftHandler::new(num_points[1]));
        let fft_handler = std::sync::Mutex::new(ndrustfft::FftHandler::new(num_points[0]));
        let rfft_output_size = num_points[1] / 2 + 1;
        // See https://math.mit.edu/~stevenj/fft-deriv.pdf
        let reflect = |index: i64, num_points: i64| {
            if 2 * index < num_points {
                index
            } else if 2 * index > num_points {
                index - num_points
            } else {
                0
            }
        };
        let wavenumbers =
            nd::Array3::build_uninit((2, num_points[0], rfft_output_size), |mut uninit| {
                for j in 0..num_points[0] {
                    let j_reflected = reflect(j as i64, num_points[0] as i64) as Float;
                    for k in 0..rfft_output_size {
                        uninit[[0, j, k]].write(
                            ComplexFloat::i() * float_consts::TAU * j_reflected / lengths[0],
                        );
                        uninit[[1, j, k]]
                            .write(ComplexFloat::i() * float_consts::TAU * k as Float / lengths[1]);
                    }
                }
            });
        let wavenumbers = unsafe { wavenumbers.assume_init() };
        let wavenumbers_sq = wavenumbers
            .mapv(|comp| comp.norm_sqr())
            .sum_axis(nd::Axis(0))
            .slice(nd::s![nd::NewAxis, .., ..])
            .to_owned();

        Self {
            num_points,
            lengths,
            wavenumbers,
            wavenumbers_sq,
            rfft_handler,
            fft_handler,
            rfft_output_size,
        }
    }

    fn to_grid<D: FftDimension>(
        &self,
        spectral: &nd::Array<ComplexFloat, D>,
    ) -> nd::Array<Float, D> {
        let first_spatial_axis = spectral.shape().len() - 2;
        let mut temp_field_1 = spectral.clone();
        let mut temp_field_2 = nd::Array::zeros(D::change_last_axis(
            temp_field_1.raw_dim(),
            self.num_points[1],
        ));
        ndrustfft::ndifft_par(
            &spectral,
            &mut temp_field_1,
            &mut self.fft_handler.lock().unwrap(),
            first_spatial_axis,
        );
        ndrustfft::ndifft_r2c_par(
            &temp_field_1,
            &mut temp_field_2,
            &mut self.rfft_handler.lock().unwrap(),
            first_spatial_axis + 1,
        );
        temp_field_2
    }

    fn to_spectral<S: RawFloatData, D: FftDimension>(
        &self,
        grid: &nd::ArrayBase<S, D>,
    ) -> nd::Array<ComplexFloat, D> {
        let first_spatial_axis = grid.shape().len() - 2;
        let shape = D::change_last_axis(grid.raw_dim(), self.rfft_output_size);
        let mut temp_field_1 = nd::Array::zeros(shape);
        let mut temp_field_2 = temp_field_1.clone();
        ndrustfft::ndfft_r2c_par(
            &grid,
            &mut temp_field_1,
            &mut self.rfft_handler.lock().unwrap(),
            first_spatial_axis + 1,
        );
        ndrustfft::ndfft_par(
            &temp_field_1,
            &mut temp_field_2,
            &mut self.fft_handler.lock().unwrap(),
            first_spatial_axis,
        );
        temp_field_2
    }
}

impl Basis for RectangularPeriodicBasis {
    type SpectralScalarField = nd::Array2<ComplexFloat>;
    type SpectralVectorField = nd::Array3<ComplexFloat>;

    fn scalar_spectral_size(&self) -> usize {
        self.num_points[0] * (self.num_points[1] / 2 + 1)
    }

    fn axes(&self) -> [nd::Array1<Float>; 2] {
        [
            nd::Array1::linspace(
                0.,
                self.lengths[0] * (1. - 1. / self.num_points[0] as Float),
                self.num_points[0],
            ),
            nd::Array1::linspace(
                0.,
                self.lengths[1] * (1. - 1. / self.num_points[1] as Float),
                self.num_points[1],
            ),
        ]
    }

    fn scalar_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralScalarField {
        nd::ArrayView2::from_shape((self.num_points[0], self.rfft_output_size), slice)
            .unwrap()
            .to_owned()
    }

    fn vector_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralVectorField {
        nd::ArrayView3::from_shape((2, self.num_points[0], self.rfft_output_size), slice)
            .unwrap()
            .to_owned()
    }

    fn scalar_to_slice(&self, spectral: &Self::SpectralScalarField, slice: &mut [ComplexFloat]) {
        let mut array_mut =
            nd::ArrayViewMut2::from_shape((self.num_points[0], self.rfft_output_size), slice)
                .unwrap();
        array_mut.assign(spectral);
    }

    fn vector_to_slice(&self, spectral: &Self::SpectralVectorField, slice: &mut [ComplexFloat]) {
        let mut array_mut =
            nd::ArrayViewMut3::from_shape((2, self.num_points[0], self.rfft_output_size), slice)
                .unwrap();
        array_mut.assign(spectral);
    }

    fn scalar_to_grid(&self, spectral: &Self::SpectralScalarField) -> nd::Array2<Float> {
        Self::to_grid(&self, spectral)
    }

    fn vector_to_grid(&self, spectral: &Self::SpectralVectorField) -> nd::Array3<Float> {
        Self::to_grid(&self, spectral)
    }

    fn scalar_to_spectral(&self, grid: &nd::Array2<Float>) -> Self::SpectralScalarField {
        Self::to_spectral(&self, grid)
    }

    fn vector_to_spectral(&self, grid: &nd::Array3<Float>) -> Self::SpectralVectorField {
        Self::to_spectral(&self, grid)
    }

    fn gradient(&self, spectral: &Self::SpectralScalarField) -> Self::SpectralVectorField {
        &spectral.slice(nd::s![nd::NewAxis, .., ..]) * &self.wavenumbers
    }

    fn divergence(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralScalarField {
        (spectral * &self.wavenumbers).sum_axis(nd::Axis(0))
    }

    fn vector_laplacian(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralVectorField {
        -spectral * &self.wavenumbers_sq
    }

    fn vector_advection(
        &self,
        grid: &nd::Array3<Float>,
        spectral: &Self::SpectralVectorField,
    ) -> Self::SpectralVectorField {
        let grad_spectral = &spectral.slice(nd::s![nd::NewAxis, .., .., ..]).to_owned()
            * &self
                .wavenumbers
                .slice(nd::s![.., nd::NewAxis, .., ..,])
                .to_owned();
        let grad_grid = self.to_grid(&grad_spectral);
        self.vector_to_spectral(
            &(&grid.slice(nd::s![.., nd::NewAxis, .., ..]) * &grad_grid).sum_axis(nd::Axis(1)),
        )
    }

    fn z_cross(&self, spectral: &nd::Array3<Float>) -> nd::Array3<Float> {
        // Not implemented.
        nd::Array3::zeros(spectral.raw_dim())
    }
}

#[cfg(test)]
mod tests {
    use float_consts::PI;

    use super::*;
    use crate::test_util::assert_all_close;

    #[test]
    fn test_gradient() {
        for num_points in [[200, 201], [201, 200], [300, 201], [301, 200]] {
            let basis = RectangularPeriodicBasis::new(num_points, [10., 11.]);
            let field_grid = basis.make_scalar(|x, y| {
                1. + 0.5 * ((PI * x / 10.).sin() * (PI * y / 11.).sin()).powi(50)
            });
            let field_spectral = basis.scalar_to_spectral(&field_grid);

            assert_all_close(&field_grid, &basis.scalar_to_grid(&field_spectral))
                .with_abs_tol(1e-5);

            let field_gradient_spectral = basis.gradient(&field_spectral);
            let field_gradient_grid = basis.vector_to_grid(&field_gradient_spectral);

            let expected_field_gradient_grid = basis.make_vector(|x, y| {
                [
                    0.5 * 50.
                        * (PI * x / 10.).sin().powi(49)
                        * (PI * x / 10.).cos()
                        * (PI / 10.)
                        * (PI * y / 11.).sin().powi(50),
                    0.5 * 50.
                        * (PI * x / 10.).sin().powi(50)
                        * (PI * y / 11.).sin().powi(49)
                        * (PI * y / 11.).cos()
                        * (PI / 11.),
                ]
            });

            assert_all_close(&field_gradient_grid, &expected_field_gradient_grid)
                .with_abs_tol(1e-2);
        }
    }
}
