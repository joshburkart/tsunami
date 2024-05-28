#![allow(non_snake_case)]

use crate::{float_consts, ComplexFloat, Float, RawFloatData};
use ndarray as nd;

pub trait Basis {
    type SpectralScalarField;
    type SpectralVectorField;

    fn scalar_size(&self) -> usize;
    fn vector_size(&self) -> usize;

    fn axes(&self) -> [nd::Array1<Float>; 2];

    fn make_scalar<F: Fn(Float, Float) -> Float>(&self, f: F) -> nd::Array2<Float>;
    fn make_vector<F: Fn(Float, Float) -> [Float; 2]>(&self, f: F) -> nd::Array3<Float>;

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
        let rfft_handler = std::sync::Mutex::new(ndrustfft::R2cFftHandler::new(num_points[0]));
        let fft_handler = std::sync::Mutex::new(ndrustfft::FftHandler::new(num_points[1]));
        let rfft_output_size = num_points[0] / 2 + 1;
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
            nd::Array3::build_uninit((rfft_output_size, num_points[1], 2), |mut uninit| {
                for j in 0..rfft_output_size {
                    for k in 0..num_points[1] {
                        let k_reflected = reflect(k as i64, num_points[1] as i64) as Float;
                        uninit[[j, k, 0]].write(
                            ComplexFloat::i() * float_consts::TAU * (j as Float) / lengths[0],
                        );
                        uninit[[j, k, 1]].write(
                            ComplexFloat::i() * float_consts::TAU * k_reflected / lengths[1],
                        );
                    }
                }
            });
        let wavenumbers = unsafe { wavenumbers.assume_init() };
        let wavenumbers_sq = wavenumbers
            .mapv(|comp| comp.norm_sqr())
            .sum_axis(nd::Axis(2))
            .slice(nd::s![.., .., nd::NewAxis])
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
        let mut temp_field_1 = spectral.clone();
        let mut temp_field_2 =
            nd::Array::zeros(D::change_axis_0(temp_field_1.raw_dim(), self.num_points[0]));
        ndrustfft::ndifft(
            &spectral,
            &mut temp_field_1,
            &mut self.fft_handler.lock().unwrap(),
            1,
        );
        ndrustfft::ndifft_r2c(
            &temp_field_1,
            &mut temp_field_2,
            &mut self.rfft_handler.lock().unwrap(),
            0,
        );
        temp_field_2
    }

    fn to_spectral<S: RawFloatData, D: FftDimension>(
        &self,
        grid: &nd::ArrayBase<S, D>,
    ) -> nd::Array<ComplexFloat, D> {
        let shape = D::change_axis_0(grid.raw_dim(), self.rfft_output_size);
        let mut temp_field_1 = nd::Array::zeros(shape);
        let mut temp_field_2 = temp_field_1.clone();
        ndrustfft::ndfft_r2c(
            &grid,
            &mut temp_field_1,
            &mut self.rfft_handler.lock().unwrap(),
            0,
        );
        ndrustfft::ndfft(
            &temp_field_1,
            &mut temp_field_2,
            &mut self.fft_handler.lock().unwrap(),
            1,
        );
        temp_field_2
    }
}

impl Basis for RectangularPeriodicBasis {
    type SpectralScalarField = nd::Array2<ComplexFloat>;
    type SpectralVectorField = nd::Array3<ComplexFloat>;

    fn scalar_size(&self) -> usize {
        (self.num_points[0] / 2 + 1) * self.num_points[1]
    }
    fn vector_size(&self) -> usize {
        (self.num_points[0] / 2 + 1) * self.num_points[1] * 2
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

    fn make_scalar<F: Fn(Float, Float) -> Float>(&self, f: F) -> nd::Array2<Float> {
        let [xs, ys] = self.axes();
        let scalar =
            nd::Array2::build_uninit((self.num_points[0], self.num_points[1]), |mut scalar| {
                for i in 0..self.num_points[0] {
                    for j in 0..self.num_points[1] {
                        scalar[[i, j]].write(f(xs[i], ys[j]));
                    }
                }
            });
        unsafe { scalar.assume_init() }
    }
    fn make_vector<F: Fn(Float, Float) -> [Float; 2]>(&self, f: F) -> nd::Array3<Float> {
        let [xs, ys] = self.axes();
        let vector =
            nd::Array3::build_uninit((self.num_points[0], self.num_points[1], 2), |mut vector| {
                for i in 0..self.num_points[0] {
                    for j in 0..self.num_points[1] {
                        let value = f(xs[i], ys[j]);
                        vector[[i, j, 0]].write(value[0]);
                        vector[[i, j, 1]].write(value[1]);
                    }
                }
            });
        unsafe { vector.assume_init() }
    }

    fn scalar_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralScalarField {
        nd::ArrayView2::from_shape((self.rfft_output_size, self.num_points[1]), slice)
            .unwrap()
            .to_owned()
    }
    fn vector_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralVectorField {
        nd::ArrayView3::from_shape((self.rfft_output_size, self.num_points[1], 2), slice)
            .unwrap()
            .to_owned()
    }

    fn scalar_to_slice(&self, spectral: &Self::SpectralScalarField, slice: &mut [ComplexFloat]) {
        let mut array_mut =
            nd::ArrayViewMut2::from_shape((self.rfft_output_size, self.num_points[1]), slice)
                .unwrap();
        for i in 0..spectral.shape()[0] {
            for j in 0..spectral.shape()[1] {
                array_mut[[i, j]] = spectral[[i, j]];
            }
        }
    }
    fn vector_to_slice(&self, spectral: &Self::SpectralVectorField, slice: &mut [ComplexFloat]) {
        let mut array_mut =
            nd::ArrayViewMut3::from_shape((self.rfft_output_size, self.num_points[1], 2), slice)
                .unwrap();
        for i in 0..spectral.shape()[0] {
            for j in 0..spectral.shape()[1] {
                for k in 0..2 {
                    array_mut[[i, j, k]] = spectral[[i, j, k]];
                }
            }
        }
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
        &spectral.slice(nd::s![.., .., nd::NewAxis]) * &self.wavenumbers
    }
    fn divergence(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralScalarField {
        (spectral * &self.wavenumbers).sum_axis(nd::Axis(2))
    }
    fn vector_laplacian(&self, spectral: &Self::SpectralVectorField) -> Self::SpectralVectorField {
        -spectral * &self.wavenumbers_sq
    }
    fn vector_advection(
        &self,
        grid: &nd::Array3<Float>,
        spectral: &Self::SpectralVectorField,
    ) -> Self::SpectralVectorField {
        let grad_spectral = &spectral.slice(nd::s![.., .., nd::NewAxis, ..]).to_owned()
            * &self
                .wavenumbers
                .slice(nd::s![.., .., .., nd::NewAxis])
                .to_owned();
        let grad_grid = self.to_grid(&grad_spectral);
        self.vector_to_spectral(
            &(&grid.slice(nd::s![.., .., .., nd::NewAxis]) * &grad_grid).sum_axis(nd::Axis(3)),
        )
    }
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

struct SphericalHarmonicBasis {
    max_l: usize,
    Lambda_sq: nd::Array1<Float>,

    radius: Float,

    mu_grid: nd::Array1<Float>,
    gauss_legendre_weights: nd::Array1<Float>,

    fft_handler: std::sync::Mutex<ndrustfft::R2cFftHandler<Float>>,
    P_nm_mu: nd::Array3<Float>,
}

impl SphericalHarmonicBasis {
    pub fn new(radius: Float, max_l: usize) -> Self {
        let gauss_quad::GaussLegendre {
            nodes: mu_grid,
            weights: gauss_legendre_weights,
        } = gauss_quad::GaussLegendre::init(max_l);
        let mu_grid =
            nd::Array1::<Float>::from_vec(mu_grid.into_iter().map(|f| f as f32).collect());
        let gauss_legendre_weights = nd::Array1::<Float>::from_vec(
            gauss_legendre_weights
                .into_iter()
                .map(|f| f as f32)
                .collect(),
        );

        let Lambda_sq = (0..max_l + 1).map(|l| (l * (l + 1)) as Float).collect();

        let fft_handler =
            ndrustfft::R2cFftHandler::new(max_l).normalization(ndrustfft::Normalization::None);
        let P_nm_mu = compute_legendre_polys(max_l, &mu_grid);

        Self {
            max_l,
            Lambda_sq,
            radius,
            mu_grid,
            gauss_legendre_weights,
            fft_handler: fft_handler.into(),
            P_nm_mu,
        }
    }
}

impl Basis for SphericalHarmonicBasis {
    type SpectralScalarField = SphericalHarmonicField;
    type SpectralVectorField = VectorSphericalHarmonicField;

    fn scalar_size(&self) -> usize {
        self.mu_grid.len() * self.max_l
    }
    fn vector_size(&self) -> usize {
        self.mu_grid.len() * self.max_l * 2
    }

    fn axes(&self) -> [nd::Array1<Float>; 2] {
        [
            nd::Array1::linspace(0., float_consts::TAU, self.max_l),
            self.mu_grid.map(|mu| mu.acos()),
        ]
    }

    fn make_scalar<F: Fn(Float, Float) -> Float>(&self, f: F) -> nd::Array2<Float> {
        todo!()
    }
    fn make_vector<F: Fn(Float, Float) -> [Float; 2]>(&self, f: F) -> nd::Array3<Float> {
        todo!()
    }

    fn scalar_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralScalarField {
        todo!()
    }
    fn vector_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralVectorField {
        let split_point = slice.len() / 2;
        Self::SpectralVectorField {
            Psi: self.scalar_from_slice(&slice[0..split_point]),
            Phi: self.scalar_from_slice(&slice[split_point..]),
        }
    }

    fn scalar_to_slice(&self, spectral: &Self::SpectralScalarField, slice: &mut [ComplexFloat]) {
        todo!()
    }
    fn vector_to_slice(&self, spectral: &Self::SpectralVectorField, slice: &mut [ComplexFloat]) {
        todo!()
    }

    fn scalar_to_grid(&self, spectral: &SphericalHarmonicField) -> nd::Array2<Float> {
        let mut values_theta_phi = nd::Array2::zeros((self.mu_grid.len(), 2 * self.max_l + 1));
        ndrustfft::ndifft_r2c(
            &spectral.coefs_lm,
            &mut values_theta_phi,
            &mut *self.fft_handler.lock().unwrap(),
            1,
        );
        todo!();
        values_theta_phi
    }
    fn vector_to_grid(&self, spectral: &VectorSphericalHarmonicField) -> nd::Array3<Float> {
        todo!()
    }

    fn scalar_to_spectral(&self, grid: &nd::Array2<Float>) -> SphericalHarmonicField {
        todo!()
    }
    fn vector_to_spectral(&self, grid: &nd::Array3<Float>) -> VectorSphericalHarmonicField {
        todo!()
    }

    fn gradient(&self, spectral: &SphericalHarmonicField) -> VectorSphericalHarmonicField {
        todo!()
    }
    fn divergence(&self, spectral: &VectorSphericalHarmonicField) -> SphericalHarmonicField {
        todo!()
    }
    fn vector_laplacian(
        &self,
        field: &VectorSphericalHarmonicField,
    ) -> VectorSphericalHarmonicField {
        todo!()
    }
    fn vector_advection(
        &self,
        grid: &nd::Array3<Float>,
        spectral: &Self::SpectralVectorField,
    ) -> Self::SpectralVectorField {
        todo!()
    }
}

#[derive(Clone)]
struct SphericalHarmonicField {
    coefs_lm: nd::Array2<ComplexFloat>,
}

impl std::ops::Neg for SphericalHarmonicField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            coefs_lm: -self.coefs_lm,
        }
    }
}

impl std::ops::Add<Self> for SphericalHarmonicField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            coefs_lm: self.coefs_lm + rhs.coefs_lm,
        }
    }
}

impl std::ops::Mul<SphericalHarmonicField> for ComplexFloat {
    type Output = SphericalHarmonicField;

    fn mul(self, rhs: SphericalHarmonicField) -> Self::Output {
        Self::Output {
            coefs_lm: self * rhs.coefs_lm,
        }
    }
}

#[derive(Clone)]
struct VectorSphericalHarmonicField {
    Psi: SphericalHarmonicField,
    Phi: SphericalHarmonicField,
}

impl std::ops::Neg for VectorSphericalHarmonicField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            Psi: -self.Psi,
            Phi: -self.Phi,
        }
    }
}

impl std::ops::Add<Self> for VectorSphericalHarmonicField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            Psi: self.Psi + rhs.Psi,
            Phi: self.Phi + rhs.Phi,
        }
    }
}

impl std::ops::Mul<VectorSphericalHarmonicField> for ComplexFloat {
    type Output = VectorSphericalHarmonicField;

    fn mul(self, rhs: VectorSphericalHarmonicField) -> Self::Output {
        Self::Output {
            Psi: self * rhs.Psi,
            Phi: self * rhs.Phi,
        }
    }
}

fn compute_legendre_polys(max_l: usize, mu_grid: &nd::Array1<Float>) -> nd::Array3<Float> {
    let a_nm = {
        let mut a_nm = nd::Array2::zeros((max_l + 1, max_l + 1));

        // Compute $a_m^m$.
        for m in 1..max_l + 1 {
            let k = m as Float;
            a_nm[[m, m]] = a_nm[[m - 1, m - 1]] * ((2. * k + 1.) / (2. * k)).sqrt();
        }
        a_nm *= 1. / (4. * float_consts::PI).sqrt();

        // Compute $a_n^m$.
        for m in 0..max_l + 1 {
            for n in 0..max_l + 1 {
                if n == m {
                    continue;
                }
                a_nm[[n, m]] =
                    ((4 * n.pow(2) - 1) as Float / (n.pow(2) - m.pow(2)) as Float).sqrt();
            }
        }
    };

    let b_mn = {
        let mut b_mn = nd::Array2::<Float>::zeros((max_l + 1, max_l + 1));

        todo!();

        b_mn
    };

    let mut P_nm_mu = nd::Array3::zeros((max_l + 1, max_l + 1, mu_grid.len()));

    todo!();

    P_nm_mu
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_util::assert_all_close;

    use float_consts::PI;

    #[test]
    fn test_gradient() {
        for num_points in [[200, 201], [201, 200], [300, 201], [301, 200]] {
            let basis = RectangularPeriodicBasis::new(num_points, [10., 11.]);
            let field_grid = basis.make_scalar(|x, y| {
                1. + 0.5 * ((PI * x / 10.).sin() * (PI * y / 11.).sin()).powi(50)
            });
            let field_spectral = basis.scalar_to_spectral(&field_grid);

            assert_all_close(&field_grid, &basis.scalar_to_grid(&field_spectral))
                .abs_tol(Some(1e-5));

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
                .abs_tol(Some(1e-2));
        }
    }
}
