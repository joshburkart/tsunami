use ndarray as nd;
use ndarray::parallel::prelude::*;

use crate::{
    bases::{periodic_grid_search, periodic_linear_interpolate, Basis},
    float_consts, ComplexFloat, Float,
};

pub struct SphericalHarmonicBasis {
    max_l: usize,
    Lambda_sq: nd::Array1<Float>,

    mu_gauss_legendre_quad: GaussLegendreQuadrature,
    phi_grid: nd::Array1<Float>,
    rfft_handler: ndrustfft::R2cFftHandler<Float>,

    vector_spherical_harmonics: VectorSphericalHarmonics,
}

impl SphericalHarmonicBasis {
    pub fn new(max_l: usize) -> Self {
        let phi_grid = nd::Array1::linspace(
            0.,
            float_consts::TAU * (1. - 1. / (2 * max_l + 1) as Float),
            2 * max_l + 1,
        );

        let mu_gauss_legendre_quad = GaussLegendreQuadrature::new(max_l);

        let Lambda_sq = (0..=max_l)
            .map(|l| (l as Float * (l as Float + 1.)))
            .collect();

        let rfft_handler = ndrustfft::R2cFftHandler::new(2 * max_l + 1)
            .normalization(ndrustfft::Normalization::None);
        let vector_spherical_harmonics =
            VectorSphericalHarmonics::new(max_l, &mu_gauss_legendre_quad.nodes);

        Self {
            max_l,
            Lambda_sq,
            mu_gauss_legendre_quad,
            phi_grid,
            rfft_handler,
            vector_spherical_harmonics,
        }
    }
}

pub enum Component {
    Theta = 0,
    Phi = 1,
}

impl Basis for SphericalHarmonicBasis {
    type SpectralScalarField = SphericalHarmonicField;
    type SpectralVectorField = VectorSphericalHarmonicField;

    fn scalar_spectral_size(&self) -> usize {
        (self.max_l + 2) * (self.max_l + 1) / 2
    }

    fn lengths(&self) -> [Float; 2] {
        [2., float_consts::TAU]
    }

    fn axes(&self) -> [&nd::Array1<Float>; 2] {
        [&self.mu_gauss_legendre_quad.nodes, &self.phi_grid]
    }

    fn make_random_points(&self) -> nd::Array2<Float> {
        let mut rng = frand::Rand::with_seed(0);
        let mut points = nd::Array2::zeros((2, crate::physics::NUM_TRACER_POINTS));
        for i in 0..crate::physics::NUM_TRACER_POINTS {
            // Crazy: https://math.stackexchange.com/a/1586015/146975
            let mu = rng.gen_range((-1.)..(1.));
            let phi = rng.gen_range((0.)..(float_consts::TAU));
            points[[Component::Theta as usize, i]] = mu;
            points[[Component::Phi as usize, i]] = phi;
        }
        points
    }

    fn scalar_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralScalarField {
        assert_eq!(self.scalar_spectral_size(), slice.len());
        let mut f_l_m = nd::Array2::zeros((self.max_l + 1, self.max_l + 1));
        let mut i = 0;
        for l in 0..=self.max_l {
            for m in 0..=l {
                f_l_m[[l, m]] = slice[i];
                i += 1;
            }
        }
        assert_eq!(i, self.scalar_spectral_size());
        Self::SpectralScalarField { f_l_m }
    }

    fn vector_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralVectorField {
        assert_eq!(self.vector_spectral_size(), slice.len());
        let split_point = slice.len() / 2;
        Self::SpectralVectorField {
            Psi: self.scalar_from_slice(&slice[0..split_point]),
            Phi: Some(self.scalar_from_slice(&slice[split_point..])),
        }
    }

    fn scalar_to_slice(&self, spectral: &Self::SpectralScalarField, slice: &mut [ComplexFloat]) {
        assert_eq!(self.scalar_spectral_size(), slice.len());
        let mut i = 0;
        for l in 0..=self.max_l {
            for m in 0..=l {
                slice[i] = spectral.f_l_m[[l, m]];
                i += 1;
            }
        }
        assert_eq!(i, self.scalar_spectral_size());
    }

    fn vector_to_slice(&self, spectral: &Self::SpectralVectorField, slice: &mut [ComplexFloat]) {
        assert_eq!(self.vector_spectral_size(), slice.len());
        let scalar_spectral_size = self.scalar_spectral_size();
        self.scalar_to_slice(&spectral.Psi, &mut slice[0..scalar_spectral_size]);
        if let Some(Phi) = &spectral.Phi {
            self.scalar_to_slice(&Phi, &mut slice[scalar_spectral_size..]);
        } else {
            for val in &mut slice[scalar_spectral_size..] {
                *val = (0.).into();
            }
        }
    }

    fn scalar_to_grid(&self, spectral: &SphericalHarmonicField) -> nd::Array2<Float> {
        let mut f_mu_m =
            nd::Array2::zeros((self.mu_gauss_legendre_quad.nodes.len(), self.max_l + 1));
        f_mu_m
            .axis_iter_mut(nd::Axis(0))
            .into_par_iter()
            .zip_eq(
                self.vector_spherical_harmonics
                    .P_l_m_mu
                    .axis_iter(nd::Axis(2)),
            )
            .for_each(|(mut f_m, P_l_m)| {
                for m in 0..=self.max_l {
                    for l in m..=self.max_l {
                        f_m[[m]] += spectral.f_l_m[[l, m]] * P_l_m[[l, m]];
                    }
                }
            });

        let mut f_mu_phi =
            nd::Array2::zeros((self.mu_gauss_legendre_quad.nodes.len(), self.phi_grid.len()));
        ndrustfft::ndifft_r2c_par(&f_mu_m, &mut f_mu_phi, &self.rfft_handler, 1);

        f_mu_phi
    }

    fn vector_to_grid(&self, spectral: &VectorSphericalHarmonicField) -> nd::Array3<Float> {
        let Q_l_m_mu = &self.vector_spherical_harmonics.Q_l_m_mu;
        let iR_l_m_mu = &self.vector_spherical_harmonics.iR_l_m_mu;

        let mut f_comp_mu_m =
            nd::Array3::zeros((2, self.mu_gauss_legendre_quad.nodes.len(), self.max_l + 1));

        f_comp_mu_m
            .axis_iter_mut(nd::Axis(1))
            .into_par_iter()
            .zip_eq(Q_l_m_mu.axis_iter(nd::Axis(2)))
            .zip_eq(iR_l_m_mu.axis_iter(nd::Axis(2)))
            .for_each(|((mut f_comp_m, Q_l_m), iR_l_m)| {
                for l in 1..=self.max_l {
                    for m in 0..=l {
                        let V_l_m = spectral.Psi.f_l_m[[l, m]];
                        let W_l_m = if let Some(Phi) = &spectral.Phi {
                            Phi.f_l_m[[l, m]]
                        } else {
                            ComplexFloat::from(0.)
                        };
                        f_comp_m[[Component::Theta as usize, m]] +=
                            V_l_m * Q_l_m[[l, m]] - W_l_m * iR_l_m[[l, m]];
                        f_comp_m[[Component::Phi as usize, m]] +=
                            V_l_m * iR_l_m[[l, m]] + W_l_m * Q_l_m[[l, m]];
                    }
                }
            });

        let mut f_comp_mu_phi = nd::Array3::zeros((
            2,
            self.mu_gauss_legendre_quad.nodes.len(),
            self.phi_grid.len(),
        ));
        ndrustfft::ndifft_r2c_par(&f_comp_mu_m, &mut f_comp_mu_phi, &self.rfft_handler, 2);

        f_comp_mu_phi
    }

    fn scalar_to_spectral(&self, grid: &nd::Array2<Float>) -> SphericalHarmonicField {
        let mut f_mu_m =
            nd::Array2::zeros((self.mu_gauss_legendre_quad.nodes.len(), self.max_l + 1));
        ndrustfft::ndfft_r2c_par(&grid, &mut f_mu_m, &self.rfft_handler, 1);
        f_mu_m *= ComplexFloat::from(float_consts::TAU / (2 * self.max_l + 1) as Float);

        let P_l_m_mu = &self.vector_spherical_harmonics.P_l_m_mu;
        let w = &self.mu_gauss_legendre_quad.weights;
        let mut f_l_m = nd::Array2::zeros((self.max_l + 1, self.max_l + 1));
        f_l_m
            .axis_iter_mut(nd::Axis(0))
            .into_par_iter()
            .zip_eq(P_l_m_mu.axis_iter(nd::Axis(0)))
            .zip_eq(0..=(self.max_l as u16))
            .for_each(|((mut f_m, P_m_mu), l)| {
                for m in 0..=l as usize {
                    for i in 0..self.mu_gauss_legendre_quad.nodes.len() {
                        f_m[[m]] += P_m_mu[[m, i]] * f_mu_m[[i, m]] * w[i];
                    }
                }
            });
        SphericalHarmonicField { f_l_m }
    }

    fn vector_to_spectral(&self, grid: &nd::Array3<Float>) -> VectorSphericalHarmonicField {
        let mut f_comp_mu_m =
            nd::Array3::zeros((2, self.mu_gauss_legendre_quad.nodes.len(), self.max_l + 1));
        ndrustfft::ndfft_r2c_par(&grid, &mut f_comp_mu_m, &self.rfft_handler, 2);
        f_comp_mu_m *= ComplexFloat::from(float_consts::TAU / (2 * self.max_l + 1) as Float);
        let fw_comp_mu_m = {
            f_comp_mu_m.zip_mut_with(
                &self
                    .mu_gauss_legendre_quad
                    .weights
                    .slice(nd::s![nd::NewAxis, .., nd::NewAxis]),
                |f, w| *f *= w,
            );
            f_comp_mu_m
        };

        let Q_l_m_mu = &self.vector_spherical_harmonics.Q_l_m_mu;
        let iR_l_m_mu = &self.vector_spherical_harmonics.iR_l_m_mu;
        let mut Psi_f_l_m = nd::Array2::<ComplexFloat>::zeros((self.max_l + 1, self.max_l + 1));
        let mut Phi_f_l_m = nd::Array2::<ComplexFloat>::zeros((self.max_l + 1, self.max_l + 1));
        Psi_f_l_m
            .axis_iter_mut(nd::Axis(0))
            .into_par_iter()
            .zip_eq(Phi_f_l_m.axis_iter_mut(nd::Axis(0)))
            .zip_eq(Q_l_m_mu.axis_iter(nd::Axis(0)))
            .zip_eq(iR_l_m_mu.axis_iter(nd::Axis(0)))
            .zip_eq(0..=self.max_l as u16)
            .for_each(|((((mut Psi_f_m, mut Phi_f_m), Q_m_mu), iR_m_mu), l)| {
                for i in 0..self.mu_gauss_legendre_quad.nodes.len() {
                    for m in 0..=l as usize {
                        Psi_f_m[[m]] += Q_m_mu[[m, i]]
                            * fw_comp_mu_m[[Component::Theta as usize, i, m]]
                            - iR_m_mu[[m, i]] * fw_comp_mu_m[[Component::Phi as usize, i, m]];
                        Phi_f_m[[m]] += iR_m_mu[[m, i]]
                            * fw_comp_mu_m[[Component::Theta as usize, i, m]]
                            + Q_m_mu[[m, i]] * fw_comp_mu_m[[Component::Phi as usize, i, m]];
                    }
                }
            });
        VectorSphericalHarmonicField {
            Psi: SphericalHarmonicField { f_l_m: Psi_f_l_m },
            Phi: Some(SphericalHarmonicField { f_l_m: Phi_f_l_m }),
        }
    }

    fn scalar_to_points(
        &self,
        grid: &nd::Array2<Float>,
        points: nd::ArrayView2<'_, Float>,
    ) -> nd::Array1<Float> {
        let mut output = nd::Array1::zeros(points.shape()[1]);
        let top_value = grid.slice(nd::s![0, ..]).mean().unwrap();
        let bottom_value = grid.slice(nd::s![-1, ..]).mean().unwrap();
        output
            .axis_iter_mut(nd::Axis(0))
            .into_par_iter()
            .zip_eq(points.axis_iter(nd::Axis(1)))
            .for_each(|(mut output_value, point)| {
                let grid_search_result = periodic_grid_search(self, point);
                output_value[[]] = if grid_search_result[0][1].index == 0 {
                    if grid_search_result[0][1].value > 0. {
                        top_value
                    } else {
                        bottom_value
                    }
                } else {
                    periodic_linear_interpolate(
                        point,
                        &grid_search_result,
                        grid.view(),
                        &self.lengths(),
                    )
                };
            });
        output
    }

    fn velocity_to_points(
        &self,
        velocity_grid: &nd::Array3<Float>,
        points: nd::ArrayView2<'_, Float>,
    ) -> nd::Array2<Float> {
        // Project the velocity grid to Cartesian so we can interpolate in Cartesian.
        let [mu_grid, phi_grid] = self.axes();
        let cartesian_velocity_grid = {
            let mut values = nd::Array3::zeros((3, mu_grid.len(), phi_grid.len()));
            for (j, &phi) in phi_grid.iter().enumerate() {
                let (sin_phi, cos_phi) = phi.sin_cos();
                for (i, &mu) in mu_grid.iter().enumerate() {
                    let mu = mu.clamp(-1., 1.);
                    let sin_theta = (1. - mu.powi(2)).sqrt();
                    let v_mu = velocity_grid[[0, i, j]];
                    let v_phi = velocity_grid[[1, i, j]];
                    values[[0, i, j]] = v_mu * mu * cos_phi - v_phi * sin_phi;
                    values[[1, i, j]] = v_mu * mu * sin_phi + v_phi * cos_phi;
                    values[[2, i, j]] = -v_mu * sin_theta;
                }
            }
            values
        };

        let top_cartesian_velocity = cartesian_velocity_grid
            .slice(nd::s![.., 0, ..])
            .mean_axis(nd::Axis(1))
            .unwrap();
        let bottom_cartesian_velocity = cartesian_velocity_grid
            .slice(nd::s![.., -1, ..])
            .mean_axis(nd::Axis(1))
            .unwrap();

        let mut output = nd::Array2::zeros((2, points.shape()[1]));
        output
            .axis_iter_mut(nd::Axis(1))
            .into_par_iter()
            .zip_eq(points.axis_iter(nd::Axis(1)))
            .for_each(|(mut output_value, point)| {
                let grid_search_result = periodic_grid_search(self, point);
                let mut interpolated_cartesian_velocity = nd::Array1::zeros(3);
                if grid_search_result[0][1].index == 0 {
                    // We're at a polar cap. Use the mean polar Cartesian velocity.
                    if grid_search_result[0][1].value > 0. {
                        interpolated_cartesian_velocity.assign(&top_cartesian_velocity);
                    } else {
                        interpolated_cartesian_velocity.assign(&bottom_cartesian_velocity);
                    };
                } else {
                    for k in 0..3 {
                        interpolated_cartesian_velocity[[k]] = periodic_linear_interpolate(
                            point,
                            &grid_search_result,
                            cartesian_velocity_grid.slice(nd::s![k, .., ..]),
                            &self.lengths(),
                        );
                    }
                }

                // Project Cartesian velocity to spherical unit vectors and incorporate
                // $\csc(\theta)$ factor for $\phi$ time derivative (see notes).
                let (sin_phi, cos_phi) = point[[1]].sin_cos();
                let cos_theta = point[[0]].clamp(-1., 1.);
                let sin_theta = (1. - cos_theta.powi(2)).sqrt();
                let dtheta_dt = cos_theta * cos_phi * interpolated_cartesian_velocity[[0]]
                    + cos_theta * sin_phi * interpolated_cartesian_velocity[[1]]
                    - sin_theta * interpolated_cartesian_velocity[[2]];
                output_value[[0]] = -sin_theta * dtheta_dt;
                output_value[[1]] = (-sin_phi * interpolated_cartesian_velocity[[0]]
                    + cos_phi * interpolated_cartesian_velocity[[1]])
                    * (1. / sin_theta).min(1000.);
            });

        output
    }

    fn gradient(&self, spectral: &SphericalHarmonicField) -> VectorSphericalHarmonicField {
        VectorSphericalHarmonicField {
            Psi: SphericalHarmonicField {
                f_l_m: spectral.f_l_m.clone(),
            },
            Phi: None,
        }
    }

    fn divergence(&self, spectral: &VectorSphericalHarmonicField) -> SphericalHarmonicField {
        SphericalHarmonicField {
            f_l_m: -&spectral.Psi.f_l_m * &self.Lambda_sq.slice(nd::s![.., nd::NewAxis]),
        }
    }

    fn vector_laplacian(
        &self,
        field: &VectorSphericalHarmonicField,
    ) -> VectorSphericalHarmonicField {
        let neg_Lambda_sq = -&self.Lambda_sq.slice(nd::s![.., nd::NewAxis]);
        VectorSphericalHarmonicField {
            Psi: SphericalHarmonicField {
                f_l_m: &field.Psi.f_l_m * &neg_Lambda_sq,
            },
            Phi: field.Phi.as_ref().map(|Phi| SphericalHarmonicField {
                f_l_m: &Phi.f_l_m * &neg_Lambda_sq,
            }),
        }
    }

    fn vector_advection(
        &self,
        _grid: &nd::Array3<Float>,
        spectral: &VectorSphericalHarmonicField,
    ) -> VectorSphericalHarmonicField {
        // TODO fill this in correctly!
        VectorSphericalHarmonicField {
            Psi: SphericalHarmonicField {
                f_l_m: nd::Array::zeros(spectral.Psi.f_l_m.raw_dim()),
            },
            Phi: None,
        }
    }

    fn z_cross(&self, grid: &nd::Array3<Float>) -> nd::Array3<Float> {
        let mut output = nd::Array3::zeros(grid.raw_dim());
        output
            .slice_mut(nd::s![Component::Theta as usize, .., ..])
            .assign(&grid.slice(nd::s![Component::Phi as usize, .., ..]));
        output
            .slice_mut(nd::s![Component::Phi as usize, .., ..])
            .assign(&grid.slice(nd::s![Component::Theta as usize, .., ..]));
        for (i, &mu) in self.mu_gauss_legendre_quad.nodes.iter().enumerate() {
            for j in 0..self.phi_grid.len() {
                output[[Component::Theta as usize, i, j]] *= -mu;
                output[[Component::Phi as usize, i, j]] *= mu;
            }
        }
        output
    }
}

#[derive(Clone, Debug)]
pub struct SphericalHarmonicField {
    pub f_l_m: nd::Array2<ComplexFloat>,
}

impl std::ops::Neg for SphericalHarmonicField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { f_l_m: -self.f_l_m }
    }
}

impl<'a> std::ops::Add<Self> for &'a SphericalHarmonicField {
    type Output = SphericalHarmonicField;

    fn add(self, rhs: Self) -> Self::Output {
        SphericalHarmonicField {
            f_l_m: &self.f_l_m + &rhs.f_l_m,
        }
    }
}

impl<'a> std::ops::Sub<Self> for &'a SphericalHarmonicField {
    type Output = SphericalHarmonicField;

    fn sub(self, rhs: Self) -> Self::Output {
        SphericalHarmonicField {
            f_l_m: &self.f_l_m - &rhs.f_l_m,
        }
    }
}

impl std::ops::Mul<SphericalHarmonicField> for ComplexFloat {
    type Output = SphericalHarmonicField;

    fn mul(self, rhs: SphericalHarmonicField) -> Self::Output {
        Self::Output {
            f_l_m: self * rhs.f_l_m,
        }
    }
}

#[derive(Clone)]
pub struct VectorSphericalHarmonicField {
    /// The $\mathbf{\Psi}$ spherical harmonic coefficients.
    Psi: SphericalHarmonicField,
    /// The $\mathbf{\Phi}$ spherical harmonic coefficients, or [`None`] to
    /// indicate zero (as an optimization, since e.g. the gradient of
    /// spherical harmonic field has zero $\mathbf{\Phi}$ component).
    Phi: Option<SphericalHarmonicField>,
}

impl std::ops::Neg for VectorSphericalHarmonicField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            Psi: -self.Psi,
            Phi: self.Phi.map(|Phi| -Phi),
        }
    }
}

impl std::ops::Add<Self> for VectorSphericalHarmonicField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            Psi: &self.Psi + &rhs.Psi,
            Phi: match (&self.Phi, &rhs.Phi) {
                (None, None) => None,
                (None, Some(Phi_rhs)) => Some(Phi_rhs.clone()),
                (Some(Phi_lhs), None) => Some(Phi_lhs.clone()),
                (Some(Phi_lhs), Some(Phi_rhs)) => Some(Phi_lhs + Phi_rhs),
            },
        }
    }
}

impl std::ops::Sub<Self> for VectorSphericalHarmonicField {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            Psi: &self.Psi - &rhs.Psi,
            Phi: match (&self.Phi, &rhs.Phi) {
                (None, None) => None,
                (None, Some(Phi_rhs)) => Some(-Phi_rhs.clone()),
                (Some(Phi_lhs), None) => Some(Phi_lhs.clone()),
                (Some(Phi_lhs), Some(Phi_rhs)) => Some(Phi_lhs - Phi_rhs),
            },
        }
    }
}

impl std::ops::Mul<VectorSphericalHarmonicField> for ComplexFloat {
    type Output = VectorSphericalHarmonicField;

    fn mul(self, rhs: VectorSphericalHarmonicField) -> Self::Output {
        Self::Output {
            Psi: self * rhs.Psi,
            Phi: rhs.Phi.map(|Phi| self * Phi),
        }
    }
}

struct VectorSphericalHarmonics {
    /// Associated Legendre functions, evaluated on a grid, normalized as
    /// documented in [`Self::compute_legendre_funcs`].
    pub P_l_m_mu: nd::Array3<Float>,
    /// First vector spherical harmonic function, normalized as per
    /// [`Self::P_l_m_mu`].
    pub Q_l_m_mu: nd::Array3<ComplexFloat>,
    /// Second vector spherical harmonic function, normalized as per
    /// [`Self::P_l_m_mu`].
    pub iR_l_m_mu: nd::Array3<ComplexFloat>,
}

impl VectorSphericalHarmonics {
    pub fn new(max_l: usize, mu_grid: &nd::Array1<Float>) -> Self {
        let P_l_m_mu = Self::compute_legendre_funcs(max_l, mu_grid);
        let (Q_l_m_mu, R_l_m_mu) = Self::compute_vector_spherical_harmonic_funcs(&P_l_m_mu);
        let Q_l_m_mu = Q_l_m_mu.mapv(|Q| ComplexFloat::new(Q, 0.));
        let iR_l_m_mu = R_l_m_mu.mapv(|R| ComplexFloat::new(0., R));

        Self {
            P_l_m_mu,
            Q_l_m_mu,
            iR_l_m_mu,
        }
    }

    /// Compute the associated Legendre polynomials on a grid of input values up
    /// to a given max order.
    ///
    /// Uses the Legendre polynomial normalization geared towards simple
    /// normalized spherical harmonic computation from: Schaeffer,
    /// Nathanaël. "Efficient spherical harmonic transforms
    /// aimed at pseudospectral numerical simulations." Geochemistry,
    /// Geophysics, Geosystems 14.3 (2013): 751-758.
    ///
    /// Note that eq. (2) from that paper erroneously includes a factor of
    /// $(-1)^m$, which is inconsistent with the rest of its results. We do
    /// not include this Condon-Shortley phase factor in this function.
    fn compute_legendre_funcs(max_l: usize, mu_grid: &nd::Array1<Float>) -> nd::Array3<Float> {
        let a_lm = {
            let mut a_lm = nd::Array2::zeros((max_l + 1, max_l + 1));

            // Compute $a_l^m$.
            a_lm[[0, 0]] = 1.;
            for m in 1..=max_l {
                let k = m as Float;
                a_lm[[m, m]] = a_lm[[m - 1, m - 1]] * ((2. * k + 1.) / (2. * k)).sqrt();
            }
            a_lm *= 1. / (4. * float_consts::PI).sqrt();

            // Compute $a_l^m$.
            for l in 0..=max_l {
                for m in 0..l {
                    a_lm[[l, m]] =
                        ((4 * l.pow(2) - 1) as Float / (l.pow(2) - m.pow(2)) as Float).sqrt();
                }
            }

            a_lm
        };

        let b_lm = {
            let mut b_lm = nd::Array2::<Float>::zeros((max_l + 1, max_l + 1));

            for l in 2..=max_l {
                for m in 0..l {
                    let nf = l as Float;
                    let mf = m as Float;
                    b_lm[[l, m]] = -((2. * nf + 1.) / (2. * nf - 3.)
                        * ((nf - 1.).powi(2) - mf.powi(2))
                        / (nf.powi(2) - mf.powi(2)))
                    .sqrt();
                }
            }

            b_lm
        };

        let mut P_l_m_mu = nd::Array3::zeros((max_l + 1, max_l + 1, mu_grid.len()));

        // First, fill diagonals and off diagonals.
        for m in 0..=max_l {
            for (i, &mu) in mu_grid.iter().enumerate() {
                P_l_m_mu[[m, m, i]] = a_lm[[m, m]] * (1. - mu.powi(2)).powf(m as Float / 2.);
                if m < max_l {
                    P_l_m_mu[[m + 1, m, i]] = a_lm[[m + 1, m]] * mu * P_l_m_mu[[m, m, i]];
                }
            }
        }

        // Next, fill the rest.
        for m in 0..=max_l {
            for l in m + 1..=max_l {
                for (i, &mu) in mu_grid.iter().enumerate() {
                    P_l_m_mu[[l, m, i]] = a_lm[[l, m]] * mu * P_l_m_mu[[l - 1, m, i]];
                }
                if l == m + 1 {
                    continue;
                }
                for i in 0..mu_grid.len() {
                    P_l_m_mu[[l, m, i]] += b_lm[[l, m]] * P_l_m_mu[[l - 2, m, i]];
                }
            }
        }

        P_l_m_mu
    }

    fn compute_vector_spherical_harmonic_funcs(
        P_l_m_mu: &nd::Array3<Float>,
    ) -> (nd::Array3<Float>, nd::Array3<Float>) {
        let mut Q_l_m_mu = nd::Array::zeros(P_l_m_mu.raw_dim());
        let mut R_l_m_mu = nd::Array::zeros(P_l_m_mu.raw_dim());

        let sqrt_i32 = |x: i32| {
            if x > 0 { (x as Float).sqrt() } else { 0. }
        };

        let max_l = P_l_m_mu.shape()[0] - 1;
        let num_mus = P_l_m_mu.shape()[2];
        for l in 0..=max_l {
            let Lambda = (if l > 0 { l * (l + 1) } else { 1 } as Float).sqrt();
            let l_sqrt_factor = if l > 0 {
                ((2 * l + 1) as Float / (2 * l - 1) as Float).sqrt()
            } else {
                0.
            };
            let li = l as i32;

            for m in 0..=l {
                let mi = m as i32;

                for i in 0..num_mus {
                    let P_l_mp1 = if m < l { P_l_m_mu[[l, m + 1, i]] } else { 0. };
                    let P_l_mm1 = if m > 0 {
                        P_l_m_mu[[l, m - 1, i]]
                    } else {
                        if l > 0 { -P_l_m_mu[[l, 1, i]] } else { 0. }
                    };
                    let P_lm1_mp1 = if l > 0 {
                        if m < l - 1 {
                            P_l_m_mu[[l - 1, m + 1, i]]
                        } else {
                            0.
                        }
                    } else {
                        0.
                    };
                    let P_lm1_mm1 = if l > 0 {
                        if m > 0 {
                            P_l_m_mu[[l - 1, m - 1, i]]
                        } else {
                            -P_l_m_mu[[l - 1, 1, i]]
                        }
                    } else {
                        0.
                    };

                    Q_l_m_mu[[l, m, i]] = -0.5
                        * (-sqrt_i32((li - mi) * (li + mi + 1)) * P_l_mp1
                            + sqrt_i32((li + mi) * (li - mi + 1)) * P_l_mm1)
                        / Lambda;
                    R_l_m_mu[[l, m, i]] = -0.5
                        * l_sqrt_factor
                        * (sqrt_i32((li - mi) * (li - mi - 1)) * P_lm1_mp1
                            + sqrt_i32((li + mi) * (li + mi - 1)) * P_lm1_mm1)
                        / Lambda;
                }
            }
        }

        (Q_l_m_mu, R_l_m_mu)
    }
}

#[derive(Debug)]
struct GaussLegendreQuadrature {
    // Gauss-Legendre nodes at the degree used to construct the instance.
    pub nodes: nd::Array1<Float>,
    // Quadrature weights for Gauss-Legendre nodes.
    pub weights: nd::Array1<Float>,
}

impl GaussLegendreQuadrature {
    pub fn new(max_l: usize) -> Self {
        use itertools::Itertools;
        let gauss_legendre = gauss_quad::GaussLegendre::init(max_l + 1);
        let (nodes, weights): (Vec<Float>, Vec<Float>) = gauss_legendre
            .nodes
            .into_iter()
            .map(|x| x as Float)
            .zip_eq(gauss_legendre.weights.into_iter().map(|x| x as Float))
            .sorted_by(|(x, _), (y, _)| x.partial_cmp(y).unwrap())
            .unzip();
        Self {
            nodes: nodes.into(),
            weights: weights.into(),
        }
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use float_consts::FRAC_1_PI;

    use super::*;
    use crate::test_util::assert_all_close;

    #[test]
    fn test_scalar_roundtrip_zeroth() {
        let basis = SphericalHarmonicBasis::new(0);

        let grid = basis.make_scalar(|_, _| 1.);
        let spectral = basis.scalar_to_spectral(&grid);
        let grid_roundtrip = basis.scalar_to_grid(&spectral);

        assert_eq!(grid.shape(), &[1, 1]);
        assert_eq!(spectral.f_l_m.shape(), &[1, 1]);
        assert_eq!(grid_roundtrip.shape(), &[1, 1]);
        assert_relative_eq!(grid[[0, 0]], 1.);
        assert_relative_eq!(
            spectral.f_l_m[[0, 0]],
            ComplexFloat::from(2. * float_consts::PI.sqrt())
        );
        assert_relative_eq!(grid_roundtrip[[0, 0]], 1.);
    }

    #[test]
    fn test_scalar_roundtrip_first() {
        let basis = SphericalHarmonicBasis::new(1);

        let grid = basis.make_scalar(|mu, _| mu);
        let spectral = basis.scalar_to_spectral(&grid);
        let grid_roundtrip = basis.scalar_to_grid(&spectral);

        assert_eq!(grid.shape(), &[2, 3]);
        assert_eq!(spectral.f_l_m.shape(), &[2, 2]);
        assert_eq!(grid_roundtrip.shape(), &[2, 3]);
        assert_all_close(
            &spectral.f_l_m,
            &nd::array![[0., 0.], [2. * (float_consts::PI / 3.).sqrt(), 0.,]]
                .mapv(ComplexFloat::from),
        )
        .with_print_ratio(true)
        .with_abs_tol(1e-5);
        assert_all_close(&grid, &grid_roundtrip).with_print_ratio(true);
    }

    #[test]
    fn test_scalar_roundtrip() {
        test_scalar_roundtrip_impl(0, |_mu, _phi| 1.);
        test_scalar_roundtrip_impl(2, |mu, phi| (1. - mu.powi(2)) * (2. * phi).sin());
        test_scalar_roundtrip_impl(7, |mu, phi| (1. - mu.powi(2)) * (2. * phi).sin());
        test_scalar_roundtrip_impl(7, |mu, _| mu.powi(4));
    }

    fn test_scalar_roundtrip_impl<F: Fn(Float, Float) -> Float>(max_l: usize, f: F) {
        let basis = SphericalHarmonicBasis::new(max_l);

        // Make an arbitrary scalar field in grid space.
        let grid = basis.make_scalar(f);

        // Perform a roundtrip conversion.
        let spectral = basis.scalar_to_spectral(&grid);
        let roundtrip_grid = basis.scalar_to_grid(&spectral);
        let roundtrip_spectral = basis.scalar_to_spectral(&roundtrip_grid);

        // Ensure the roundtrip was lossless.
        assert_all_close(&grid, &roundtrip_grid)
            .with_rel_tol(1e-5)
            .with_abs_tol(1e-5)
            .with_print_ratio(true);
        assert_all_close(&roundtrip_spectral.f_l_m, &spectral.f_l_m)
            .with_rel_tol(1e-5)
            .with_abs_tol(1e-5);
    }

    #[test]
    fn test_vector_roundtrip() {
        let basis = SphericalHarmonicBasis::new(8);

        let mut Psi = SphericalHarmonicField {
            f_l_m: nd::Array::zeros((8 + 1, 8 + 1)),
        };
        let mut Phi = SphericalHarmonicField {
            f_l_m: nd::Array::zeros((8 + 1, 8 + 1)),
        };
        Psi.f_l_m[[4, 2]] = ComplexFloat::new(7., 1.5);
        Psi.f_l_m[[5, 0]] = ComplexFloat::new(-0.9, 0.);
        Phi.f_l_m[[1, 1]] = ComplexFloat::new(4.4, 1.);
        Phi.f_l_m[[2, 1]] = ComplexFloat::new(0.4, -1.2);
        Phi.f_l_m[[5, 0]] = ComplexFloat::new(-6.6, 0.);
        Phi.f_l_m[[1, 0]] = ComplexFloat::new(6.1, 0.);
        let spectral = VectorSphericalHarmonicField {
            Psi,
            Phi: Some(Phi),
        };

        let grid = basis.vector_to_grid(&spectral);
        let roundtrip_spectral = basis.vector_to_spectral(&grid);

        assert_all_close(&spectral.Psi.f_l_m, &roundtrip_spectral.Psi.f_l_m).with_abs_tol(1e-5);
        assert_all_close(
            &spectral.Phi.unwrap().f_l_m,
            &roundtrip_spectral.Phi.unwrap().f_l_m,
        )
        .with_abs_tol(1e-5);
    }

    #[test]
    fn test_vsh_normalization_and_orthogonality() {
        let max_l = 13;
        let gauss_legendre_quad = GaussLegendreQuadrature::new(max_l);
        let vsh = VectorSphericalHarmonics::new(max_l, &gauss_legendre_quad.nodes);

        // Test normalization.
        for l in 1..10 {
            for m in 0..=l {
                assert_relative_eq!(
                    1.,
                    float_consts::TAU
                        * (&(vsh.Q_l_m_mu.slice(nd::s![l, m, ..]).mapv(|Q| Q.norm_sqr())
                            + vsh.iR_l_m_mu.slice(nd::s![l, m, ..]).mapv(|R| R.norm_sqr()))
                            * &gauss_legendre_quad.weights)
                            .sum(),
                    max_relative = 1e-5
                );
            }
        }

        // Test that $\mathbf{\Psi}_{lm}$ is orthogonal to $\mathbf{\Phi}_{lm}^*$.
        for l in 1..10 {
            for m in 0..=l {
                assert_relative_eq!(
                    ComplexFloat::new(0., 0.),
                    (&vsh.Q_l_m_mu.slice(nd::s![l, m, ..])
                        * &vsh.iR_l_m_mu.slice(nd::s![l, m, ..])
                        * &gauss_legendre_quad.weights)
                        .sum(),
                    max_relative = 1e-5
                );
            }
        }

        // Test that $\mathbf{\Psi}_{lm}$ is orthogonal to $\mathbf{\Psi}_{l'm}^*$ for
        // $l \ne l'$ and similarly with $\mathbf{\Phi}$.
        for l in [0, 3, 8] {
            for lp in [1, 2, 7] {
                for m in 0..=l.min(lp) {
                    assert_relative_eq!(
                        ComplexFloat::new(0., 0.),
                        ((&vsh.Q_l_m_mu.slice(nd::s![l, m, ..])
                            * &vsh.Q_l_m_mu.slice(nd::s![lp, m, ..])
                            - &vsh.iR_l_m_mu.slice(nd::s![l, m, ..])
                                * &vsh.iR_l_m_mu.slice(nd::s![lp, m, ..]))
                            * &gauss_legendre_quad.weights)
                            .sum(),
                        epsilon = 1e-5
                    );
                }
            }
        }
    }

    #[test]
    fn test_legendre_normalization_and_orthogonality() {
        let max_l = 13;
        let gauss_legendre_quad = GaussLegendreQuadrature::new(max_l);
        let P_l_m_mu =
            VectorSphericalHarmonics::compute_legendre_funcs(max_l, &gauss_legendre_quad.nodes);

        // Test normalization.
        for l in [0, 3, 8] {
            for m in 0..=l {
                assert_relative_eq!(
                    1.,
                    float_consts::TAU
                        * (&P_l_m_mu.slice(nd::s![l, m, ..]).mapv(|P| P.powi(2))
                            * &gauss_legendre_quad.weights)
                            .sum(),
                    epsilon = 1e-5
                );
            }
        }

        // Test orthogonality.
        for l in [0, 3, 8] {
            for lp in [1, 2, 7] {
                for m in 0..=l.min(lp) {
                    assert_relative_eq!(
                        0.,
                        (&P_l_m_mu.slice(nd::s![l, m, ..])
                            * &P_l_m_mu.slice(nd::s![lp, m, ..])
                            * &gauss_legendre_quad.weights)
                            .sum(),
                        epsilon = 1e-5
                    );
                }
            }
        }
    }

    #[test]
    fn test_spherical_harmonics() {
        // From Wikipedia: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
        let sin = |mu: Float| (1. - mu.powi(2)).sqrt();
        let manual_Y_lm = |l: u32, m: u32, mu: Float| match (l, m) {
            (0, 0) => 1. / 2. * FRAC_1_PI.sqrt(),
            (1, 0) => 1. / 2. * (3. * FRAC_1_PI).sqrt() * mu,
            (1, 1) => -1. / 2. * (3. * FRAC_1_PI / 2.).sqrt() * sin(mu),
            (2, 0) => 1. / 4. * (5. * FRAC_1_PI).sqrt() * (3. * mu.powi(2) - 1.),
            (2, 1) => -1. / 2. * (15. / 2. * FRAC_1_PI).sqrt() * sin(mu) * mu,
            (2, 2) => 1. / 4. * (15. / 2. * FRAC_1_PI).sqrt() * (1. - mu.powi(2)),
            (3, 0) => 1. / 4. * (7. * FRAC_1_PI).sqrt() * (5. * mu.powi(3) - 3. * mu),
            (3, 1) => -1. / 8. * (21. * FRAC_1_PI).sqrt() * sin(mu) * (5. * mu.powi(2) - 1.),
            (3, 2) => 1. / 4. * (105. / 2. * FRAC_1_PI).sqrt() * sin(mu).powi(2) * mu,
            (3, 3) => -1. / 8. * (35. * FRAC_1_PI).sqrt() * sin(mu).powi(3),
            _ => panic!(),
        };

        // (-1)^m to match Condon-Shortley phase used in Wikipedia.
        let condon_shortley = |m: u32| (-1i32).pow(m as u32) as Float;

        let mu_grid = nd::arr1(&[0., 0.123, 0.55555, 0.87, 1., -0.5, -0.1]);
        let P_l_m_mu = VectorSphericalHarmonics::compute_legendre_funcs(3, &mu_grid);
        for l in 0..4 {
            for m in 0..=l {
                for (i, &mu) in mu_grid.iter().enumerate() {
                    let Y_lm = condon_shortley(m as u32) * P_l_m_mu[[l, m, i]];
                    assert_relative_eq!(
                        Y_lm,
                        manual_Y_lm(l as u32, m as u32, mu),
                        max_relative = 1e-5
                    );
                }
            }
        }
    }

    #[test]
    fn test_vector_spherical_harmonics() {
        let one = ComplexFloat::from(1.);
        let zero = ComplexFloat::from(0.);
        let thetah = || nd::array![one, zero];
        let phih = || nd::array![zero, one];

        // From Wikipedia: https://en.wikipedia.org/wiki/Vector_spherical_harmonics
        let sin = |mu: Float| (1. - mu.powi(2)).sqrt();
        let manual_Psi_lm = |l: u32, m: u32, mu: Float| match (l, m) {
            (0, 0) => zero * thetah(),
            (1, 0) => -one / 2. * (3. * FRAC_1_PI).sqrt() * sin(mu) * thetah(),
            (1, 1) => {
                -one / 2.
                    * (3. * FRAC_1_PI / 2.).sqrt()
                    * (one * mu * thetah() + ComplexFloat::i() * phih())
            }
            (2, 0) => -one * 3. / 2. * (5. * FRAC_1_PI).sqrt() * sin(mu) * mu * thetah(),
            (2, 1) => {
                -one * (15. / 8. * FRAC_1_PI).sqrt()
                    * (one * (2. * mu.acos()).cos() * thetah() + ComplexFloat::i() * mu * phih())
            }
            (2, 2) => {
                one * (15. / 8. * FRAC_1_PI).sqrt()
                    * sin(mu)
                    * (one * mu * thetah() + ComplexFloat::i() * phih())
            }
            _ => panic!(),
        };
        let manual_Phi_lm = |l: u32, m: u32, mu: Float| match (l, m) {
            (0, 0) => zero * thetah(),
            (1, 0) => -one / 2. * (3. * FRAC_1_PI).sqrt() * sin(mu) * phih(),
            (1, 1) => {
                one / 2.
                    * (3. * FRAC_1_PI / 2.).sqrt()
                    * (-one * mu * phih() + ComplexFloat::i() * thetah())
            }
            (2, 0) => -one * 3. / 2. * (5. * FRAC_1_PI).sqrt() * sin(mu) * mu * phih(),
            (2, 1) => {
                one * (15. / 8. * FRAC_1_PI).sqrt()
                    * (-one * (2. * mu.acos()).cos() * phih() + ComplexFloat::i() * mu * thetah())
            }
            (2, 2) => {
                one * (15. / 8. * FRAC_1_PI).sqrt()
                    * sin(mu)
                    * (one * mu * phih() - ComplexFloat::i() * thetah())
            }
            _ => panic!(),
        };

        // (-1)^m to match Condon-Shortley phase used in Wikipedia.
        let condon_shortley = |m: u32| (-1i32).pow(m as u32) as Float;

        let mu_grid = nd::arr1(&[0., 0.123, 0.55555, 0.87, 1., -0.5, -0.1]);
        let P_l_m_mu = VectorSphericalHarmonics::compute_legendre_funcs(7, &mu_grid);
        let (Q_l_m_mu, R_l_m_mu) =
            VectorSphericalHarmonics::compute_vector_spherical_harmonic_funcs(&P_l_m_mu);
        for (i, &mu) in mu_grid.iter().enumerate() {
            for l in 0..=2 {
                let Lambda = ((l * (l + 1)) as Float).sqrt();
                for m in 0..=l {
                    println!("l={l},m={m},mu={mu:.4}");
                    let Psi_lm = ComplexFloat::from(condon_shortley(m as u32))
                        * nd::array![
                            one * Q_l_m_mu[[l, m, i]],
                            ComplexFloat::i() * R_l_m_mu[[l, m, i]],
                        ]
                        * Lambda;
                    let Phi_lm = ComplexFloat::from(condon_shortley(m as u32))
                        * nd::array![
                            -ComplexFloat::i() * R_l_m_mu[[l, m, i]],
                            one * Q_l_m_mu[[l, m, i]],
                        ]
                        * Lambda;
                    assert_relative_eq!(
                        -Psi_lm, // TODO NO MINUS
                        manual_Psi_lm(l as u32, m as u32, mu),
                        max_relative = 1e-5
                    );
                    assert_relative_eq!(
                        -Phi_lm, // TODO NO MINUS
                        manual_Phi_lm(l as u32, m as u32, mu),
                        max_relative = 1e-5
                    );
                }
            }

            assert_relative_eq!(
                Q_l_m_mu[[1, 0, i]] * (2. as Float).sqrt(),
                -0.5 * (-(2 as Float).sqrt() * P_l_m_mu[[1, 1, i]]
                    - (2 as Float).sqrt() * P_l_m_mu[[1, 1, i]])
            );
        }
    }

    #[test]
    fn test_to_from_slice() {
        let max_l = 6;
        let basis = SphericalHarmonicBasis::new(max_l);
        let mut f_l_m = nd::Array2::zeros((max_l + 1, max_l + 1));
        for l in 0..=max_l {
            for m in 0..=l {
                f_l_m[[l, m]] = ComplexFloat::new(l as Float, m as Float);
            }
        }

        let scalar = SphericalHarmonicField { f_l_m };

        {
            let mut storage = vec![(0.).into(); basis.scalar_spectral_size()];
            basis.scalar_to_slice(&scalar, &mut storage[..]);
            assert_all_close(&basis.scalar_from_slice(&storage).f_l_m, &scalar.f_l_m);
        }

        {
            let vector_1 = VectorSphericalHarmonicField {
                Psi: -scalar.clone(),
                Phi: None,
            };
            let mut storage = vec![(0.).into(); basis.vector_spectral_size()];
            basis.vector_to_slice(&vector_1, &mut storage[..]);
            let vector_1_roundtrip = basis.vector_from_slice(&storage);
            assert_all_close(&vector_1_roundtrip.Psi.f_l_m, &vector_1.Psi.f_l_m);
            assert_all_close(
                &vector_1_roundtrip.Phi.unwrap().f_l_m,
                &(ComplexFloat::from(0.) * &vector_1.Psi.f_l_m),
            );
        }

        {
            let vector_2 = VectorSphericalHarmonicField {
                Psi: -scalar.clone(),
                Phi: Some(ComplexFloat::i() * scalar),
            };
            let mut storage = vec![(0.).into(); basis.vector_spectral_size()];
            basis.vector_to_slice(&vector_2, &mut storage[..]);
            let vector_1_roundtrip = basis.vector_from_slice(&storage);
            assert_all_close(&vector_1_roundtrip.Psi.f_l_m, &vector_2.Psi.f_l_m);
            assert_all_close(
                &vector_1_roundtrip.Phi.unwrap().f_l_m,
                &vector_2.Phi.unwrap().f_l_m,
            );
        }
    }
}
