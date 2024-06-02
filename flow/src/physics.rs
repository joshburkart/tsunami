use ndarray as nd;

use crate::{bases, odeint, ComplexFloat, Float, RawComplexFloatData};

pub struct Fields<S: RawComplexFloatData, B: bases::Basis> {
    basis: std::sync::Arc<B>,
    storage: nd::ArrayBase<S, nd::Ix1>,
}
impl<'a, B: bases::Basis> Fields<nd::ViewRepr<&'a ComplexFloat>, B> {
    pub fn new(
        basis: std::sync::Arc<B>,
        storage: nd::ArrayBase<nd::ViewRepr<&'a ComplexFloat>, nd::Ix1>,
    ) -> Self {
        Self { basis, storage }
    }
}
impl<'a, B: bases::Basis> Fields<nd::ViewRepr<&'a mut ComplexFloat>, B> {
    pub fn new_mut(
        basis: std::sync::Arc<B>,
        storage: nd::ArrayBase<nd::ViewRepr<&'a mut ComplexFloat>, nd::Ix1>,
    ) -> Self {
        Self { basis, storage }
    }
}
impl<B: bases::Basis> Fields<nd::OwnedRepr<ComplexFloat>, B> {
    pub fn new_owned(
        basis: std::sync::Arc<B>,
        storage: nd::ArrayBase<nd::OwnedRepr<ComplexFloat>, nd::Ix1>,
    ) -> Self {
        Self { basis, storage }
    }
}
impl<B: bases::Basis> Fields<nd::OwnedRepr<ComplexFloat>, B> {
    pub fn zeros(basis: std::sync::Arc<B>) -> Self {
        let len = basis.scalar_spectral_size() + basis.vector_spectral_size();
        let storage = nd::Array1::zeros([len]);
        Self { basis, storage }
    }
}
impl<S: RawComplexFloatData, B: bases::Basis> Fields<S, B> {
    pub fn size(&self) -> usize {
        self.basis.scalar_spectral_size() + self.basis.vector_spectral_size()
    }

    pub fn height_spectral(&self) -> B::SpectralScalarField {
        let scalar_size = self.basis.scalar_spectral_size();
        self.basis
            .scalar_from_slice(&self.storage.as_slice().unwrap()[..scalar_size])
    }
    pub fn height_grid(&self) -> nd::Array2<Float> {
        let spectral = self.height_spectral();
        self.basis.scalar_to_grid(&spectral)
    }
    pub fn velocity_spectral(&self) -> B::SpectralVectorField {
        let scalar_size = self.basis.scalar_spectral_size();
        self.basis
            .vector_from_slice(&self.storage.as_slice().unwrap()[scalar_size..])
    }
    pub fn velocity_grid(&self) -> nd::Array3<Float> {
        let spectral = self.velocity_spectral();
        self.basis.vector_to_grid(&spectral)
    }
}
impl<S: RawComplexFloatData + nd::DataMut, B: bases::Basis> Fields<S, B> {
    pub fn assign_height(&mut self, height: &B::SpectralScalarField) {
        self.basis.scalar_to_slice(
            height,
            &mut self.storage.as_slice_mut().unwrap()[..self.basis.scalar_spectral_size()],
        )
    }
    pub fn assign_velocity(&mut self, velocity: &B::SpectralVectorField) {
        self.basis.vector_to_slice(
            velocity,
            &mut self.storage.as_slice_mut().unwrap()[self.basis.scalar_spectral_size()..],
        )
    }
}

pub struct Problem<B: bases::Basis> {
    pub basis: std::sync::Arc<B>,

    pub terrain_height: B::SpectralScalarField,

    pub grav_accel: Float,
    pub kinematic_viscosity: Float,

    pub rtol: Float,
    pub atol: Float,
}

impl<B: bases::Basis> odeint::System for Problem<B>
where
    ComplexFloat: std::ops::Mul<B::SpectralScalarField, Output = B::SpectralScalarField>,
    ComplexFloat: std::ops::Mul<B::SpectralVectorField, Output = B::SpectralVectorField>,
    B::SpectralScalarField: std::ops::Neg<Output = B::SpectralScalarField> + Sized,
    for<'a> &'a B::SpectralScalarField: std::ops::Add<&'a B::SpectralScalarField, Output = B::SpectralScalarField>
        + std::ops::Sub<&'a B::SpectralScalarField, Output = B::SpectralScalarField>,
    B::SpectralVectorField: std::ops::Neg<Output = B::SpectralVectorField>
        + std::ops::Add<B::SpectralVectorField, Output = B::SpectralVectorField>
        + std::ops::Sub<B::SpectralVectorField, Output = B::SpectralVectorField>
        + Sized,
{
    type Value = ComplexFloat;

    fn system<S: nd::RawDataMut<Elem = Self::Value> + nd::Data + nd::DataMut>(
        &self,
        y: &nd::Array1<Self::Value>,
        dy: &mut nd::ArrayBase<S, nd::Ix1>,
    ) {
        let fields = Fields::new(self.basis.clone(), y.view());
        let mut fields_time_deriv = Fields::new_mut(self.basis.clone(), dy.view_mut());

        let height = fields.height_spectral();
        let velocity = fields.velocity_spectral();

        let height_grid = self.basis.scalar_to_grid(&height);
        let velocity_grid = self.basis.vector_to_grid(&velocity);

        height_grid
            .iter()
            .for_each(|&h| assert!(h >= 0., "Water column height became negative"));

        // Height time derivative.
        fields_time_deriv.assign_height(
            &(-self.basis.divergence(&self.basis.vector_to_spectral(
                &(&height_grid.slice(nd::s![nd::NewAxis, .., ..]) * &velocity_grid),
            ))),
        );

        // Velocity time derivative.
        let viscous =
            ComplexFloat::from(self.kinematic_viscosity) * self.basis.vector_laplacian(&velocity);
        let advection = -self.basis.vector_advection(&velocity_grid, &velocity);
        let gravity = -ComplexFloat::from(self.grav_accel)
            * self.basis.gradient(&(&height + &self.terrain_height));
        fields_time_deriv.assign_velocity(&(viscous + advection + gravity));
    }
}

pub struct Solver<B: bases::Basis>
where
    ComplexFloat: std::ops::Mul<B::SpectralScalarField, Output = B::SpectralScalarField>,
    ComplexFloat: std::ops::Mul<B::SpectralVectorField, Output = B::SpectralVectorField>,
    B::SpectralScalarField: std::ops::Neg<Output = B::SpectralScalarField> + Sized,
    for<'a> &'a B::SpectralScalarField: std::ops::Add<&'a B::SpectralScalarField, Output = B::SpectralScalarField>
        + std::ops::Sub<&'a B::SpectralScalarField, Output = B::SpectralScalarField>,
    B::SpectralVectorField: std::ops::Neg<Output = B::SpectralVectorField>
        + std::ops::Add<B::SpectralVectorField, Output = B::SpectralVectorField>
        + std::ops::Sub<B::SpectralVectorField, Output = B::SpectralVectorField>
        + Sized,
{
    problem: Problem<B>,
    integrator: odeint::Integrator<Problem<B>>,
}

impl<B: bases::Basis> Solver<B>
where
    ComplexFloat: std::ops::Mul<B::SpectralScalarField, Output = B::SpectralScalarField>,
    ComplexFloat: std::ops::Mul<B::SpectralVectorField, Output = B::SpectralVectorField>,
    B::SpectralScalarField: std::ops::Neg<Output = B::SpectralScalarField> + Sized,
    for<'a> &'a B::SpectralScalarField: std::ops::Add<&'a B::SpectralScalarField, Output = B::SpectralScalarField>
        + std::ops::Sub<&'a B::SpectralScalarField, Output = B::SpectralScalarField>,
    B::SpectralVectorField: std::ops::Neg<Output = B::SpectralVectorField>
        + std::ops::Add<B::SpectralVectorField, Output = B::SpectralVectorField>
        + std::ops::Sub<B::SpectralVectorField, Output = B::SpectralVectorField>
        + Sized,
{
    pub fn new(
        problem: Problem<B>,
        initial_fields: Fields<nd::OwnedRepr<ComplexFloat>, B>,
    ) -> Self {
        let mut storage = nd::Array1::zeros(initial_fields.size());
        let mut fields = Fields::new_mut(problem.basis.clone(), storage.view_mut());
        fields.assign_height(&initial_fields.height_spectral());
        fields.assign_velocity(&initial_fields.velocity_spectral());

        let integrator = odeint::Integrator::new(storage, 1e-2, 1e-5)
            .abs_tol(problem.atol)
            .rel_tol(problem.rtol);

        Self {
            problem,
            integrator,
        }
    }

    pub fn problem(&self) -> &Problem<B> {
        &self.problem
    }
    pub fn problem_mut(&mut self) -> &mut Problem<B> {
        &mut self.problem
    }

    pub fn integrate(&mut self, delta_t: Float) {
        self.integrator.integrate(&self.problem, delta_t)
    }

    pub fn fields(&self) -> Fields<nd::OwnedRepr<ComplexFloat>, B> {
        Fields::new_owned(self.problem.basis.clone(), self.integrator.y())
    }
}
