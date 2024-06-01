use crate::{bases::Basis, float_consts, ComplexFloat, Float};
use ndarray as nd;

struct SphericalHarmonicBasis {
    max_l: usize,
    Lambda_sq: nd::Array1<Float>,

    radius: Float,

    mu_gauss_legendre_quad: GaussLegendreQuadrature,
    phi_grid: nd::Array1<Float>,
    rfft_handler: std::sync::Mutex<ndrustfft::R2cFftHandler<Float>>,

    vector_spherical_harmonics: VectorSphericalHarmonics,
}

impl SphericalHarmonicBasis {
    pub fn new(radius: Float, max_l: usize) -> Self {
        let phi_grid = nd::Array1::linspace(
            0.,
            float_consts::TAU * (1. - 1. / (2 * max_l + 1) as Float),
            2 * max_l + 1,
        );

        let mu_gauss_legendre_quad = GaussLegendreQuadrature::new(max_l);

        let Lambda_sq = (0..=max_l).map(|l| (l * (l + 1)) as Float).collect();

        let rfft_handler = ndrustfft::R2cFftHandler::new(2 * max_l + 1)
            .normalization(ndrustfft::Normalization::None);
        let vector_spherical_harmonics =
            VectorSphericalHarmonics::new(max_l, &mu_gauss_legendre_quad.nodes);

        Self {
            max_l,
            Lambda_sq,
            radius,
            mu_gauss_legendre_quad,
            phi_grid,
            rfft_handler: rfft_handler.into(),
            vector_spherical_harmonics,
        }
    }

    // Convert a flattened index into an (l, m) pair (for unpacking from a 1-D slice).
    fn unflatten(&self, i: usize) -> (usize, usize) {
        (i / (self.max_l + 1), i % (self.max_l + 1))
    }
    fn scalar_spectral_size(&self) -> usize {
        (self.max_l + 2) * (self.max_l + 1) / 2
    }
}

pub enum Component {
    Phi = 0,
    Theta = 1,
}

impl Basis for SphericalHarmonicBasis {
    type SpectralScalarField = SphericalHarmonicField;
    type SpectralVectorField = VectorSphericalHarmonicField;

    fn scalar_spectral_size(&self) -> usize {
        self.mu_gauss_legendre_quad.nodes.len() * self.phi_grid.len()
    }

    fn axes(&self) -> [nd::Array1<Float>; 2] {
        [
            self.mu_gauss_legendre_quad.nodes.clone(),
            self.phi_grid.clone(),
        ]
    }

    fn scalar_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralScalarField {
        let mut f_l_m = nd::Array2::zeros((self.max_l + 1, self.max_l + 1));
        for i in 0..self.scalar_spectral_size() {
            let (l, m) = self.unflatten(i);
            f_l_m[[l, m]] = slice[i];
        }
        Self::SpectralScalarField { f_l_m: f_l_m }
    }
    fn vector_from_slice<'a>(&self, slice: &'a [ComplexFloat]) -> Self::SpectralVectorField {
        let split_point = slice.len() / 2;
        Self::SpectralVectorField {
            Psi: self.scalar_from_slice(&slice[0..split_point]),
            Phi: self.scalar_from_slice(&slice[split_point..]),
        }
    }

    fn scalar_to_slice(&self, spectral: &Self::SpectralScalarField, slice: &mut [ComplexFloat]) {
        for i in 0..self.scalar_spectral_size() {
            let (l, m) = self.unflatten(i);
            slice[i] = spectral.f_l_m[[l, m]];
        }
    }
    fn vector_to_slice(&self, spectral: &Self::SpectralVectorField, slice: &mut [ComplexFloat]) {
        let scalar_spectral_size = self.scalar_spectral_size();
        for i in 0..scalar_spectral_size {
            let (l, m) = self.unflatten(i);
            slice[i] = spectral.Psi.f_l_m[[l, m]];
            slice[i + scalar_spectral_size] = spectral.Phi.f_l_m[[l, m]];
        }
    }

    fn scalar_to_grid(&self, spectral: &SphericalHarmonicField) -> nd::Array2<Float> {
        let mut f_mu_m =
            nd::Array2::zeros((self.mu_gauss_legendre_quad.nodes.len(), self.max_l + 1));
        for i in 0..self.mu_gauss_legendre_quad.nodes.len() {
            for m in 0..=self.max_l {
                for l in m..=self.max_l {
                    f_mu_m[[i, m]] += spectral.f_l_m[[l, m]]
                        * self.vector_spherical_harmonics.P_l_m_mu[[l, m, i]];
                }
            }
        }

        let mut f_mu_phi =
            nd::Array2::zeros((self.mu_gauss_legendre_quad.nodes.len(), self.phi_grid.len()));
        ndrustfft::ndifft_r2c(
            &f_mu_m,
            &mut f_mu_phi,
            &mut *self.rfft_handler.lock().unwrap(),
            1,
        );

        f_mu_phi
    }
    fn vector_to_grid(&self, spectral: &VectorSphericalHarmonicField) -> nd::Array3<Float> {
        // let mut f_mu_m_comp =
        //     nd::Array3::zeros((self.mu_gauss_legendre_quad.nodes.len(), self.max_l + 1, 2));
        // for i in 0..self.mu_gauss_legendre_quad.nodes.len() {
        //     for m in 0..=self.max_l {
        //         for l in m..=self.max_l {
        //             f_mu_m_comp[[i, m, 0]] += spectral.Psi.f_l_m[[l, m]]
        //                 * self.vector_spherical_harmonics.P_l_m_mu[[l, m, i]];
        //         }
        //     }
        // }
        todo!()
    }

    fn scalar_to_spectral(&self, grid: &nd::Array2<Float>) -> SphericalHarmonicField {
        let mut f_mu_m =
            nd::Array2::zeros((self.mu_gauss_legendre_quad.nodes.len(), self.max_l + 1));
        ndrustfft::ndfft_r2c(
            &grid,
            &mut f_mu_m,
            &mut *self.rfft_handler.lock().unwrap(),
            1,
        );
        f_mu_m *= ComplexFloat::from(float_consts::TAU / (2 * self.max_l + 1) as Float);

        let mut f_l_m = nd::Array2::zeros((self.max_l + 1, self.max_l + 1));
        for l in 0..=self.max_l {
            for m in 0..=l {
                for i in 0..self.mu_gauss_legendre_quad.nodes.len() {
                    f_l_m[[l, m]] += self.vector_spherical_harmonics.P_l_m_mu[[l, m, i]]
                        * f_mu_m[[i, m]]
                        * self.mu_gauss_legendre_quad.weights[i];
                }
            }
        }
        SphericalHarmonicField { f_l_m }
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

#[derive(Clone, Debug)]
struct SphericalHarmonicField {
    f_l_m: nd::Array2<ComplexFloat>,
}

impl std::ops::Neg for SphericalHarmonicField {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { f_l_m: -self.f_l_m }
    }
}

impl std::ops::Add<Self> for SphericalHarmonicField {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            f_l_m: self.f_l_m + rhs.f_l_m,
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

struct VectorSphericalHarmonics {
    /// Associated Legendre functions, evaluated on a grid, normalized as documented in
    /// [`Self::compute_legendre_funcs`].
    pub P_l_m_mu: nd::Array3<Float>,
    /// First vector spherical harmonic function, normalized as per [`Self::P_l_m_mu`].
    pub Q_l_m_mu: nd::Array3<Float>,
    /// Second vector spherical harmonic function, normalized as per [`Self::P_l_m_mu`].
    pub R_l_m_mu: nd::Array3<Float>,
}

impl VectorSphericalHarmonics {
    pub fn new(max_l: usize, mu_grid: &nd::Array1<Float>) -> Self {
        let P_l_m_mu = Self::compute_legendre_funcs(max_l, mu_grid);
        let (Q_l_m_mu, R_l_m_mu) = Self::compute_vector_spherical_harmonic_funcs(&P_l_m_mu);

        Self {
            P_l_m_mu,
            Q_l_m_mu,
            R_l_m_mu,
        }
    }

    /// Compute the associated Legendre polynomials on a grid of input values up to a given max
    /// order.
    ///
    /// Uses the Legendre polynomial normalization geared towards simple normalized spherical
    /// harmonic computation from: Schaeffer, Nathanaël. "Efficient spherical harmonic transforms
    /// aimed at pseudospectral numerical simulations." Geochemistry, Geophysics, Geosystems 14.3
    /// (2013): 751-758.
    ///
    /// Note that eq. (2) from that paper erroneously includes a factor of $(-1)^m$, which is
    /// inconsistent with the rest of its results. We do not include this Condon-Shortley phase
    /// factor in this function.
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
            if x > 0 {
                (x as Float).sqrt()
            } else {
                0.
            }
        };

        let max_l = P_l_m_mu.shape()[0] - 1;
        let num_mus = P_l_m_mu.shape()[2];
        for l in 0..=max_l {
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
                        if l > 0 {
                            -P_l_m_mu[[l, 1, i]]
                        } else {
                            0.
                        }
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
                            + sqrt_i32((li + mi) * (li - mi + 1)) * P_l_mm1);
                    R_l_m_mu[[l, m, i]] = -0.5
                        * l_sqrt_factor
                        * (sqrt_i32((li - mi) * (li - mi - 1)) * P_lm1_mp1
                            + sqrt_i32((li + mi) * (li + mi - 1)) * P_lm1_mm1);
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
        let gauss_legendre = gauss_quad::GaussLegendre::init(max_l + 1);
        Self {
            nodes: gauss_legendre
                .nodes
                .into_iter()
                .map(|node| node as Float)
                .collect(),
            weights: gauss_legendre
                .weights
                .into_iter()
                .map(|weight| weight as Float)
                .collect(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_util::assert_all_close;
    use approx::assert_relative_eq;
    use float_consts::FRAC_1_PI;

    #[test]
    fn test_roundtrip_zeroth() {
        use super::Basis;

        let basis = SphericalHarmonicBasis::new(1., 0);

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
    fn test_roundtrip_first() {
        use super::Basis;

        let basis = SphericalHarmonicBasis::new(1., 1);

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
        .with_print_ratio(true);
        assert_all_close(&grid, &grid_roundtrip).with_print_ratio(true);
    }

    #[test]
    fn test_roundtrip() {
        test_roundtrip_impl(0, |_mu, _phi| 1.);
        test_roundtrip_impl(2, |mu, phi| (1. - mu.powi(2)) * (2. * phi).sin());
        test_roundtrip_impl(7, |mu, phi| (1. - mu.powi(2)) * (2. * phi).sin());
        test_roundtrip_impl(7, |mu, _| mu.powi(4));
    }

    fn test_roundtrip_impl<F: Fn(Float, Float) -> Float>(max_l: usize, f: F) {
        use super::Basis;

        let basis = SphericalHarmonicBasis::new(1., max_l);

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
    fn test_vsh_normalization_and_orthogonality() {
        let max_l = 13;
        let gauss_legendre_quad = GaussLegendreQuadrature::new(max_l);
        let vsh = VectorSphericalHarmonics::new(max_l, &gauss_legendre_quad.nodes);

        // Test normalization.
        for l in 1..10 {
            for m in 0..=l {
                assert_relative_eq!(
                    (l * (l + 1)) as Float,
                    float_consts::TAU
                        * (&(vsh.Q_l_m_mu.slice(nd::s![l, m, ..]).mapv(|Q| Q.powi(2))
                            + vsh.R_l_m_mu.slice(nd::s![l, m, ..]).mapv(|R| R.powi(2)))
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
                    0.,
                    (&vsh.Q_l_m_mu.slice(nd::s![l, m, ..])
                        * &vsh.R_l_m_mu.slice(nd::s![l, m, ..])
                        * &gauss_legendre_quad.weights)
                        .sum(),
                    max_relative = 1e-5
                );
            }
        }

        // Test that $\mathbf{\Psi}_{lm}$ is orthogonal to $\mathbf{\Psi}_{l'm}^*$ and similarly
        // with $\mathbf{\Phi}$.
        for l in [0, 3, 8] {
            for lp in [1, 2, 7] {
                for m in 0..=l.min(lp) {
                    assert_relative_eq!(
                        0.,
                        ((&vsh.Q_l_m_mu.slice(nd::s![l, m, ..])
                            * &vsh.Q_l_m_mu.slice(nd::s![lp, m, ..])
                            + &vsh.R_l_m_mu.slice(nd::s![l, m, ..])
                                * &vsh.R_l_m_mu.slice(nd::s![lp, m, ..]))
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
            for m in 0..l + 1 {
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
}
