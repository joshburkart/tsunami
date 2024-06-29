use ndarray as nd;

use crate::{
    bases,
    odeint::{self, Integrator},
    ComplexFloat, Float, RawComplexFloatData, RawFloatData,
};

pub const NUM_TRACER_POINTS: usize = 6000;

pub struct FieldsSnapshot<B: bases::Basis> {
    pub t: Float,
    pub fields: Fields<nd::OwnedRepr<ComplexFloat>, B>,
}

pub fn interp_between<'a, B: bases::Basis>(
    fields_prev: &'a FieldsSnapshot<B>,
    fields_next: &'a FieldsSnapshot<B>,
    num: usize,
) -> impl Iterator<Item = FieldsSnapshot<B>> + 'a {
    assert!(fields_prev.t <= fields_next.t);

    let delta_t = fields_next.t - fields_prev.t;
    (0..num).map(move |substep| {
        if substep == 0 {
            fields_prev.clone()
        } else {
            let t = fields_prev.t + delta_t * (substep as Float) / (num as Float);

            let weight_prev = (fields_next.t - t) / delta_t;
            let weight_next = (t - fields_prev.t) / delta_t;

            let storage_between = &fields_prev.fields.storage * weight_prev
                + &fields_next.fields.storage * weight_next;
            FieldsSnapshot {
                t,
                fields: Fields::new_owned(fields_prev.fields.basis.clone(), storage_between),
            }
        }
    })
}

impl<B: bases::Basis> Clone for FieldsSnapshot<B> {
    fn clone(&self) -> Self {
        Self {
            t: self.t.clone(),
            fields: self.fields.clone(),
        }
    }
}

pub struct Fields<S: nd::RawData + nd::Data, B: bases::Basis> {
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

impl<'a, A, B: bases::Basis> Fields<nd::ViewRepr<&'a mut A>, B> {
    pub fn new_mut(
        basis: std::sync::Arc<B>,
        storage: nd::ArrayBase<nd::ViewRepr<&'a mut A>, nd::Ix1>,
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
        let len =
            basis.scalar_spectral_size() + basis.vector_spectral_size() + 1 + NUM_TRACER_POINTS;
        let storage = nd::Array1::zeros([len]);
        Self { basis, storage }
    }
}

impl<S: RawComplexFloatData, B: bases::Basis> Fields<S, B> {
    pub fn size(&self) -> usize {
        self.basis.scalar_spectral_size()
            + self.basis.vector_spectral_size()
            + 1
            + NUM_TRACER_POINTS
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
        let vector_size = self.basis.vector_spectral_size();
        self.basis.vector_from_slice(
            &self.storage.as_slice().unwrap()[scalar_size..scalar_size + vector_size],
        )
    }

    pub fn velocity_grid(&self) -> nd::Array3<Float> {
        let spectral = self.velocity_spectral();
        self.basis.vector_to_grid(&spectral)
    }

    pub fn tracer_points(&self) -> nd::Array2<Float> {
        let scalar_size = self.basis.scalar_spectral_size();
        let vector_size = self.basis.vector_spectral_size();
        let points_complex = &self.storage.as_slice().unwrap()[scalar_size + vector_size + 1..];
        let mut points = nd::Array2::zeros((2, points_complex.len()));
        for (i, point_complex) in points_complex.iter().enumerate() {
            points[[0, i]] = point_complex.re;
            points[[1, i]] = point_complex.im;
        }
        points
    }

    pub fn lunar_phase(&self) -> Float {
        let scalar_size = self.basis.scalar_spectral_size();
        let vector_size = self.basis.vector_spectral_size();
        self.storage[[scalar_size + vector_size]].re
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
        let scalar_size = self.basis.scalar_spectral_size();
        let vector_size = self.basis.vector_spectral_size();
        self.basis.vector_to_slice(
            velocity,
            &mut self.storage.as_slice_mut().unwrap()[scalar_size..scalar_size + vector_size],
        )
    }

    pub fn assign_tracers(&mut self, tracer_points: nd::ArrayView2<Float>) {
        let offset = self.basis.scalar_spectral_size() + self.basis.vector_spectral_size() + 1;
        for (i, point) in tracer_points.axis_iter(nd::Axis(1)).enumerate() {
            self.storage[[offset + i]] = ComplexFloat::new(point[[0]], point[[1]]);
        }
    }

    pub fn assign_lunar_phase(&mut self, lunar_phase: Float) {
        self.storage[[self.basis.scalar_spectral_size() + self.basis.vector_spectral_size()]] =
            ComplexFloat::new(lunar_phase, 0.);
    }
}

impl<S: RawFloatData + nd::DataMut, B: bases::Basis> Fields<S, B> {
    pub fn height_flat_mut(&mut self) -> nd::ArrayViewMut1<'_, Float> {
        self.storage
            .slice_mut(nd::s![..self.basis.scalar_spectral_size()])
    }

    pub fn velocity_flat_mut(&mut self) -> nd::ArrayViewMut1<'_, Float> {
        self.storage.slice_mut(nd::s![
            self.basis.scalar_spectral_size()
                ..self.basis.scalar_spectral_size() + self.basis.vector_spectral_size()
        ])
    }

    pub fn tracers_flat_mut(&mut self) -> nd::ArrayViewMut1<'_, Float> {
        self.storage.slice_mut(nd::s![
            self.basis.scalar_spectral_size() + self.basis.vector_spectral_size() + 1..
        ])
    }

    pub fn lunar_phase_mut(&mut self) -> &mut Float {
        &mut self.storage[[self.basis.scalar_spectral_size() + self.basis.vector_spectral_size()]]
    }
}

impl<S: RawComplexFloatData + nd::RawDataClone, B: bases::Basis> Clone for Fields<S, B> {
    fn clone(&self) -> Self {
        Self {
            basis: self.basis.clone(),
            storage: self.storage.clone(),
        }
    }
}

pub struct Problem<B: bases::Basis> {
    pub basis: std::sync::Arc<B>,

    pub terrain_height: B::SpectralScalarField,

    pub kinematic_viscosity: Float,
    pub rotation_angular_speed: Float,
    pub tidal_prefactor: Float,
    pub lunar_distance: Float,

    pub velocity_exaggeration_factor: Float,

    pub height_tolerances: Tolerances,
    pub velocity_tolerances: Tolerances,
    pub tracers_tolerances: Tolerances,
}

pub struct Tolerances {
    pub rel: Float,
    pub abs: Float,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            rel: 1e-5,
            abs: 1e-5,
        }
    }
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
        y: nd::ArrayView1<Self::Value>,
        mut dy: nd::ArrayBase<S, nd::Ix1>,
    ) {
        let fields = Fields::new(self.basis.clone(), y.view());
        let mut fields_time_deriv = Fields::new_mut(self.basis.clone(), dy.view_mut());

        let height = fields.height_spectral();
        let velocity = fields.velocity_spectral();
        let lunar_phase = fields.lunar_phase();

        let height_grid = self.basis.scalar_to_grid(&height);
        let velocity_grid = self.basis.vector_to_grid(&velocity);

        height_grid.iter().for_each(|&h| {
            if h < 0. {
                log::error!("Water column height became negative: {h:?}");
                panic!()
            }
        });

        // Compute derivatives in parallel.
        let height_time_deriv = std::sync::Arc::new(std::sync::Mutex::new(None));
        let velocity_time_deriv = std::sync::Arc::new(std::sync::Mutex::new(None));
        let tracers_time_deriv = std::sync::Arc::new(std::sync::Mutex::new(None));
        rayon::scope(|s| {
            let velocity_grid = &velocity_grid;

            // Height time derivative.
            {
                let height_time_deriv = height_time_deriv.clone();
                s.spawn(move |_| {
                    let mut height_time_deriv = height_time_deriv.lock().unwrap();
                    *height_time_deriv =
                        Some(-self.basis.divergence(&self.basis.vector_to_spectral(
                            &(&height_grid.slice(nd::s![nd::NewAxis, .., ..]) * velocity_grid),
                        )));
                });
            }

            // Velocity time derivative.
            {
                let velocity_time_deriv = velocity_time_deriv.clone();
                s.spawn(move |_| {
                    let viscous = ComplexFloat::from(self.kinematic_viscosity)
                        * self.basis.vector_laplacian(&velocity);
                    let advection = -self.basis.vector_advection(&velocity_grid, &velocity);
                    let gravity = -self.basis.gradient(&(&height + &self.terrain_height));
                    let coriolis = self.basis.vector_to_spectral(
                        &((-2. * self.rotation_angular_speed)
                            * &self.basis.z_cross(&velocity_grid)),
                    );
                    let tidal = ComplexFloat::from(-self.tidal_prefactor)
                        * self.basis.tidal_force(self.lunar_distance, lunar_phase);
                    let mut velocity_time_deriv = velocity_time_deriv.lock().unwrap();
                    *velocity_time_deriv = Some(viscous + advection + gravity + coriolis + tidal);
                });
            }

            // Tracer points time derivative.
            {
                let tracers_time_deriv = tracers_time_deriv.clone();
                s.spawn(move |_| {
                    let mut tracers_time_deriv = tracers_time_deriv.lock().unwrap();
                    *tracers_time_deriv = Some(
                        self.velocity_exaggeration_factor
                            * self
                                .basis
                                .velocity_to_points(velocity_grid, fields.tracer_points().view()),
                    );
                });
            }
        });
        fields_time_deriv.assign_height(&height_time_deriv.lock().unwrap().as_ref().unwrap());
        fields_time_deriv.assign_velocity(&velocity_time_deriv.lock().unwrap().as_ref().unwrap());
        fields_time_deriv
            .assign_tracers(tracers_time_deriv.lock().unwrap().as_ref().unwrap().view());
        // Ignore lunar orbital motion.
        fields_time_deriv.assign_lunar_phase(-self.rotation_angular_speed);
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
    integrator:
        odeint::IntegratorNordsieckAdamsBashforth<Problem<B>, odeint::AdaptiveStepSizeManager>,
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
        let size = initial_fields.size();

        let mut storage = nd::Array1::zeros(size);
        let mut fields = Fields::new_mut(problem.basis.clone(), storage.view_mut());
        fields.assign_height(&initial_fields.height_spectral());
        fields.assign_velocity(&initial_fields.velocity_spectral());
        fields.assign_tracers(problem.basis.make_random_points().view());

        let mut abs_tol_storage = nd::Array1::<Float>::zeros(size);
        let mut abs_tol_fields = Fields::new_mut(problem.basis.clone(), abs_tol_storage.view_mut());
        *&mut abs_tol_fields.height_flat_mut() += problem.height_tolerances.abs;
        *&mut abs_tol_fields.velocity_flat_mut() += problem.velocity_tolerances.abs;
        *&mut abs_tol_fields.tracers_flat_mut() += problem.tracers_tolerances.abs;
        *abs_tol_fields.lunar_phase_mut() = problem.tracers_tolerances.abs;

        let mut rel_tol_storage = nd::Array1::<Float>::zeros(size);
        let mut rel_tol_fields = Fields::new_mut(problem.basis.clone(), rel_tol_storage.view_mut());
        *&mut rel_tol_fields.height_flat_mut() += problem.height_tolerances.rel;
        *&mut rel_tol_fields.velocity_flat_mut() += problem.velocity_tolerances.rel;
        *&mut rel_tol_fields.tracers_flat_mut() += problem.tracers_tolerances.rel;
        *rel_tol_fields.lunar_phase_mut() = problem.tracers_tolerances.rel;

        let order = 3;
        let integrator = odeint::IntegratorNordsieckAdamsBashforth::new(
            order,
            &problem,
            storage,
            odeint::AdaptiveStepSizeManager::new(order, size, 1e-2)
                .with_abs_tol(abs_tol_storage)
                .with_rel_tol(rel_tol_storage),
        );

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

    pub fn fields_snapshot(&self) -> FieldsSnapshot<B> {
        let odeint::Solution { t, y } = self.integrator.current_solution();
        FieldsSnapshot {
            t,
            fields: Fields::new_owned(self.problem.basis.clone(), y.to_owned()),
        }
    }

    pub fn fields_mut<F: FnOnce(Fields<nd::ViewRepr<&mut ComplexFloat>, B>)>(&mut self, f: F) {
        self.integrator.current_solution_mut(
            |current_solution| {
                let odeint::Solution { y, .. } = current_solution;
                f(Fields::new_mut(self.problem.basis.clone(), y))
            },
            &self.problem,
        );
    }

    pub fn step(&mut self) {
        self.integrator.step(&self.problem);
    }
}
