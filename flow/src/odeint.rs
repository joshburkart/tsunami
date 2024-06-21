#![allow(non_snake_case)]

use ndarray as nd;

use crate::{ComplexFloat, Float};

pub trait Value
where
    Self: Copy
        + Clone
        + nd::ScalarOperand
        + ndrustfft::Zero
        + std::ops::Mul<Self, Output = Self>
        + std::ops::Div<Self, Output = Self>
        + std::ops::MulAssign<Self>
        + std::ops::AddAssign<Self>
        + std::ops::SubAssign<Self>
        + std::ops::Sub<Self, Output = Self>
        + From<Float>
        + nd::LinalgScalar
        + NormSq
        + std::fmt::Display
        + std::fmt::Debug,
{
}

impl Value for Float {}
impl Value for ComplexFloat {}

pub struct Solution<V, S: nd::RawData<Elem = V>> {
    pub t: Float,
    pub y: nd::ArrayBase<S, nd::Ix1>,
}

pub trait Integrator<S: System> {
    fn step(&mut self, system: &S);

    fn current_solution(&self) -> Solution<S::Value, nd::ViewRepr<&S::Value>>;
    fn current_solution_mut<
        F: FnOnce(Solution<<S as System>::Value, ndarray::ViewRepr<&mut <S as System>::Value>>),
    >(
        &mut self,
        modifier: F,
        system: &S,
    );
}

pub trait System {
    type Value;

    fn system<S: nd::RawDataMut<Elem = Self::Value> + nd::Data + nd::DataMut>(
        &self,
        y: nd::ArrayView1<Self::Value>,
        dydt: nd::ArrayBase<S, nd::Ix1>,
    );
}

/// An explicit ODE integrator using Bulirsch-Stoer.
pub struct IntegratorBulirschStoer<S: System> {
    t: Float,
    y: nd::Array1<S::Value>,

    delta_t: Float,

    abs_tol: Float,
    rel_tol: Float,

    max_iterations: usize,
}

impl<S: System> IntegratorBulirschStoer<S>
where
    S::Value: Value,
{
    pub fn new(y_init: nd::Array1<S::Value>, delta_t: Float) -> Self {
        Self {
            t: 0.,
            y: y_init,
            delta_t,
            abs_tol: 1e-5,
            rel_tol: 1e-5,
            max_iterations: 10,
        }
    }

    pub fn with_abs_tol(self, abs_tol: Float) -> Self {
        Self { abs_tol, ..self }
    }

    pub fn with_rel_tol(self, rel_tol: Float) -> Self {
        Self { rel_tol, ..self }
    }

    pub fn with_max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }

    fn midpoint_step(
        &self,
        system: &S,
        n: usize,
        f_init: &nd::Array1<S::Value>,
    ) -> nd::Array1<S::Value> {
        let step_size = self.delta_t / n as Float;

        // 0    1    2    3    4    5    6    n
        //                  ..
        //           zi  zip1
        //           zip1 zi
        //                zi zip1
        //                  ..
        //                               zi  zip1
        let mut zi = self.y.clone();
        let mut zip1 = &zi + f_init * S::Value::from(step_size);
        let mut fi = f_init.clone();

        for _i in 1..n {
            std::mem::swap(&mut zi, &mut zip1);
            system.system(zi.view(), fi.view_mut());
            zip1 += &(&fi * S::Value::from(2. * step_size));
        }

        system.system(zip1.view(), fi.view_mut());
        (&zi + &zip1 + fi * S::Value::from(step_size)) * S::Value::from(0.5)
    }
}

impl<S: System> Integrator<S> for IntegratorBulirschStoer<S>
where
    S::Value: Value,
{
    fn current_solution(&self) -> Solution<S::Value, nd::ViewRepr<&S::Value>> {
        Solution {
            t: self.t,
            y: self.y.view(),
        }
    }

    fn current_solution_mut<
        F: FnOnce(Solution<<S as System>::Value, ndarray::ViewRepr<&mut <S as System>::Value>>),
    >(
        &mut self,
        modifier: F,
        _system: &S,
    ) {
        modifier(Solution {
            t: self.t,
            y: self.y.view_mut(),
        });
    }

    fn step(&mut self, system: &S) {
        let f_init = {
            let mut f_init = nd::Array1::zeros(self.y.raw_dim());
            system.system(self.y.view(), f_init.view_mut());
            f_init
        };

        let compute_n = |k| 2 * (k + 1);

        let mut T = Vec::<Vec<nd::Array1<S::Value>>>::new();
        for k in 0..self.max_iterations {
            let n = compute_n(k);
            let mut Tk = Vec::with_capacity(k + 1);
            Tk.push(self.midpoint_step(system, n, &f_init));
            for j in 0..k {
                Tk.push(
                    &Tk[j]
                        + (&Tk[j] - &T[k - 1][j])
                            / S::Value::from(
                                (n as Float / compute_n(k - j - 1) as Float).powi(2) - 1.,
                            ),
                );
            }

            if k > 0 {
                let last_two = Tk.last_chunk::<2>().unwrap();
                let error = compute_scaled_truncation_error(
                    last_two[0].view(),
                    last_two[1].view(),
                    self.abs_tol,
                    self.rel_tol,
                );
                if error <= 1. {
                    self.y = last_two[1].to_owned();
                    self.t += self.delta_t;
                    log::info!("Converged at step {k}, n={n}, last error was: {error:?}");
                    return;
                }
            }

            T.push(Tk);
        }

        let last_two = T.last().unwrap().last_chunk::<2>().unwrap();
        let error = compute_scaled_truncation_error(
            last_two[0].view(),
            last_two[1].view(),
            self.abs_tol,
            self.rel_tol,
        );
        panic!(
            "Failed to converge at step {}, n={}, last error was: {error:?}, T:\n{T:?}",
            self.max_iterations - 1,
            compute_n(self.max_iterations - 1)
        );
    }
}

/// An explicit, adaptive-stepsize ODE integrator using a Nordsieck (multivalue)
/// version Adams-Bashforth.
///
/// Uses the implementation/notation from:
/// <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/AdamsNordsieckFieldTransformer.html>
pub struct IntegratorNordsieckAdamsBashforth<S: System, SP: StepSizePolicy>
where
    S::Value: Value,
{
    t: Float,
    step_size_policy: SP,

    y_nordsieck: NordsieckVectorOwned<S::Value>,

    stepper: StepperNordsieckAdamsBashforth<S>,
}

impl<S: System, SP: StepSizePolicy> IntegratorNordsieckAdamsBashforth<S, SP>
where
    S::Value: Value,
{
    pub fn new(
        order: usize,
        system: &S,
        y_init: nd::Array1<S::Value>,
        step_size_policy: SP,
    ) -> Self {
        let stepper = StepperNordsieckAdamsBashforth::new(order);

        let mut y_nordsieck = stepper.initialize_y_nordsieck(
            system,
            order,
            step_size_policy.step_size(),
            y_init.view(),
        );
        y_nordsieck.change_step_size(step_size_policy.step_size());

        Self {
            t: 0.,
            step_size_policy,
            stepper,
            y_nordsieck,
        }
    }
}

impl<S: System, SP: StepSizePolicy> Integrator<S> for IntegratorNordsieckAdamsBashforth<S, SP>
where
    S::Value: Value,
{
    fn step(&mut self, system: &S) {
        let mut next_y_nordsieck = self.stepper.step(system, self.y_nordsieck.view());

        // After stepping forward, step back to the starting point, then compute the
        // error relative to the original value of `y`.
        let roundtrip_y = {
            next_y_nordsieck.change_step_size(-next_y_nordsieck.step_size);
            let roundtrip_y = next_y_nordsieck.next_y();
            next_y_nordsieck.change_step_size(-next_y_nordsieck.step_size);
            roundtrip_y
        };
        let prev_step_size = self.step_size_policy.step_size();
        match self
            .step_size_policy
            .process_step(self.y_nordsieck.y().view(), roundtrip_y.view())
        {
            StepResult::Accept => {
                // Use the error to adjust the step size.
                next_y_nordsieck.change_step_size(self.step_size_policy.step_size());
                self.y_nordsieck = next_y_nordsieck;
                // println!(
                //     "Step accepted at t={:.4}, new step size: {:.4}, new nordsieck: {:?}",
                //     self.t,
                //     self.step_size_policy.step_size(),
                //     self.y_nordsieck
                // );
                self.t += prev_step_size;
            }
            StepResult::Reject => {
                self.y_nordsieck
                    .change_step_size(self.step_size_policy.step_size());
                // println!(
                //     "Step rejected at t={:.4}, new step size: {:.4}, new nordsieck: {:?}",
                //     self.t,
                //     self.step_size_policy.step_size(),
                //     self.y_nordsieck
                // );
                self.step(system)
            }
        }
    }

    fn current_solution(
        &self,
    ) -> Solution<<S as System>::Value, ndarray::ViewRepr<&<S as System>::Value>> {
        Solution {
            t: self.t,
            y: self.y_nordsieck.y(),
        }
    }

    fn current_solution_mut<
        F: FnOnce(Solution<<S as System>::Value, ndarray::ViewRepr<&mut <S as System>::Value>>),
    >(
        &mut self,
        modifier: F,
        system: &S,
    ) {
        modifier(Solution {
            t: self.t,
            y: self.y_nordsieck.data.slice_mut(nd::s![0, ..]),
        });
        // Since the caller has modified the solution vector, the Nordsieck vector has
        // been invalidated and needs to be reinitialized.
        self.y_nordsieck = self.stepper.initialize_y_nordsieck(
            system,
            self.y_nordsieck.data.shape()[0] - 1,
            self.y_nordsieck.step_size,
            self.y_nordsieck.y(),
        );
    }
}

/// Uses the implementation/notation from:
/// <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/AdamsNordsieckFieldTransformer.html>
struct StepperNordsieckAdamsBashforth<S: System>
where
    S::Value: Value,
{
    P_forward_inv: nd::Array2<S::Value>,
    P_inv_A_P: nd::Array2<S::Value>,
    P_inv_u: nd::Array1<S::Value>,
}

impl<S: System> StepperNordsieckAdamsBashforth<S>
where
    S::Value: Value,
{
    fn new(order: usize) -> Self {
        let (P, P_inv, P_forward_inv) = {
            // Use `nalgebra` to compute matrix inverses, since `ndarray-linalg` is
            // incompatible with Wasm.
            let mut P = nalgebra::DMatrix::zeros(order - 1, order - 1);
            let mut P_forward = nalgebra::DMatrix::zeros(order - 1, order - 1);
            for i in 1i32..order as i32 {
                for j in 1i32..order as i32 {
                    P[((i - 1) as usize, (j - 1) as usize)] =
                        ((j + 1) as Float * (-i as Float).powi(j)) as Float;
                    P_forward[((i - 1) as usize, (j - 1) as usize)] =
                        ((j + 1) as Float * (i as Float).powi(j)) as Float;
                }
            }
            let P_inv = P.clone().try_inverse().unwrap();
            let P_forward_inv = P_forward.try_inverse().unwrap();

            // Note that `nalgebra` uses Fortran/column-major ordering, while `ndarray` uses
            // C/row-major ordering, hence the `reversed_axes()` calls.
            let P = nd::Array::from_shape_vec((order - 1, order - 1), P.iter().copied().collect())
                .unwrap()
                .reversed_axes();
            let P_inv =
                nd::Array::from_shape_vec((order - 1, order - 1), P_inv.iter().copied().collect())
                    .unwrap()
                    .reversed_axes();
            let P_forward_inv = nd::Array::from_shape_vec(
                (order - 1, order - 1),
                P_forward_inv.iter().copied().collect(),
            )
            .unwrap()
            .reversed_axes();

            (P, P_inv, P_forward_inv)
        };

        let mut A = nd::Array2::zeros((order - 1, order - 1));
        for i in 1i32..(order - 1) as i32 {
            A[[i as usize, (i - 1) as usize]] = 1.;
        }

        let P_inv_A_P = P_inv.dot(&A).dot(&P).mapv(S::Value::from);
        let u = nd::Array1::ones(order - 1);
        let P_inv_u = P_inv.dot(&u).mapv(S::Value::from);
        let P_forward_inv = P_forward_inv.mapv(S::Value::from);

        Self {
            P_forward_inv,
            P_inv_A_P,
            P_inv_u,
        }
    }

    fn step(
        &self,
        system: &S,
        y_nordsieck: NordsieckVectorView<'_, S::Value>,
    ) -> NordsieckVectorOwned<S::Value> {
        let mut next_y_nordsieck: nd::Array2<S::Value> =
            nd::Array2::zeros(y_nordsieck.data.raw_dim());

        // Compute next `y`.
        let next_y = y_nordsieck.next_y();
        next_y_nordsieck.slice_mut(nd::s![0, ..]).assign(&next_y);

        // Compute next `dy/dt`.
        let mut next_dydt_scaled = next_y_nordsieck.slice_mut(nd::s![1, ..]);
        system.system(next_y.view(), next_dydt_scaled.view_mut());
        next_dydt_scaled *= S::Value::from(y_nordsieck.step_size);
        let next_dydt_scaled = next_y_nordsieck.slice(nd::s![1, ..]).to_owned();

        // Compute next higher-order derivatives.
        let mut next_dnydtn_scaled = next_y_nordsieck.slice_mut(nd::s![2.., ..]);
        next_dnydtn_scaled.assign(&self.P_inv_A_P.dot(&y_nordsieck.data.slice(nd::s![2.., ..])));
        next_dnydtn_scaled += &(&(&y_nordsieck.data.slice(nd::s![1, ..]) - &next_dydt_scaled)
            .slice(nd::s![nd::NewAxis, ..])
            * &self.P_inv_u.slice(nd::s![.., nd::NewAxis]));

        NordsieckVectorOwned {
            data: next_y_nordsieck,
            step_size: y_nordsieck.step_size,
        }
    }

    fn initialize_y_nordsieck(
        &self,
        system: &S,
        order: usize,
        step_size: Float,
        y_init: nd::ArrayView1<'_, S::Value>,
    ) -> NordsieckVectorOwned<S::Value> {
        // Create a bootstrapping Nordsieck vector.
        let zeros = nd::Array::zeros(y_init.raw_dim());
        let mut dydt_scaled_init = zeros.clone();
        system.system(y_init.view(), dydt_scaled_init.view_mut());
        dydt_scaled_init *= S::Value::from(step_size);
        let mut y_nordsieck_bootstrap = vec![y_init.view(), dydt_scaled_init.view()];
        for _ in 2..=order {
            y_nordsieck_bootstrap.push(zeros.view());
        }
        let mut y_nordsieck_bootstrap = NordsieckVectorOwned {
            data: nd::stack(nd::Axis(0), &y_nordsieck_bootstrap).unwrap(),
            step_size,
        };

        // Take steps and compute the derivative using the differential equation after
        // each step. Record in `dydt_scaled_steps`.
        let mut dydt_scaled_steps = Vec::with_capacity(order - 1);
        for _ in 0..=order - 2 {
            y_nordsieck_bootstrap = self.step(system, y_nordsieck_bootstrap.view());
            dydt_scaled_steps.push(y_nordsieck_bootstrap.data.slice(nd::s![1, ..]).to_owned());
        }
        let dydt_scaled_steps = nd::stack(
            nd::Axis(0),
            &dydt_scaled_steps
                .iter()
                .map(|dydt_step| dydt_step.view())
                .collect::<Vec<_>>(),
        )
        .unwrap();

        // Convert `dydt_scaled_steps` into a vector of scaled higher derivatives using
        // the Taylor series matrix `P_forward_inv`.
        let y_nordsieck_upper: nd::Array2<S::Value> = self.P_forward_inv.dot(
            &(&dydt_scaled_steps
                - &(&dydt_scaled_init.slice(nd::s![nd::NewAxis, ..])
                    * &nd::Array1::ones(order - 1).slice(nd::s![.., nd::NewAxis]))),
        );

        // Pack into an initial Nordsieck vector.
        let y_nordsieck_lower =
            nd::stack(nd::Axis(0), &[y_init.view(), dydt_scaled_init.view()]).unwrap();
        let y_nordsieck = nd::concatenate(
            nd::Axis(0),
            &[y_nordsieck_lower.view(), y_nordsieck_upper.view()],
        )
        .unwrap();
        NordsieckVectorOwned {
            data: y_nordsieck,
            step_size,
        }
    }
}

#[derive(Debug)]
struct NordsieckVector<V: Value, SD: nd::RawData<Elem = V> + nd::Data<Elem = V>> {
    data: nd::ArrayBase<SD, nd::Ix2>,
    step_size: Float,
}

type NordsieckVectorOwned<V> = NordsieckVector<V, nd::OwnedRepr<V>>;
type NordsieckVectorView<'a, V> = NordsieckVector<V, nd::ViewRepr<&'a V>>;

impl<V: Value, SD: nd::RawData<Elem = V> + nd::Data<Elem = V>> NordsieckVector<V, SD> {
    pub fn y(&self) -> nd::ArrayView1<'_, V> {
        self.data.slice(nd::s![0, ..])
    }

    pub fn next_y(&self) -> nd::Array1<V> {
        self.data.sum_axis(nd::Axis(0))
    }
}

impl<V: Value, SD: nd::RawData<Elem = V> + nd::Data<Elem = V>> NordsieckVector<V, SD> {
    fn view(&self) -> NordsieckVectorView<'_, V> {
        NordsieckVectorView {
            data: self.data.view(),
            step_size: self.step_size,
        }
    }
}

impl<V: Value, SD: nd::RawData<Elem = V> + nd::DataMut> NordsieckVector<V, SD> {
    fn change_step_size(&mut self, new_step_size: Float) {
        self.data *= &(nd::Array1::from_iter(0..self.data.shape()[0])
            .mapv(|n| V::from((new_step_size / self.step_size).powi(n as i32)))
            .slice(nd::s![.., nd::NewAxis]));
        self.step_size = new_step_size;
    }
}

pub trait NormSq: Copy {
    fn norm_sq(self) -> Float;
}

impl NormSq for Float {
    fn norm_sq(self) -> Float {
        self.powi(2)
    }
}
impl NormSq for ComplexFloat {
    fn norm_sq(self) -> Float {
        self.norm_sqr()
    }
}

pub trait StepSizePolicy {
    fn step_size(&self) -> Float;
    fn process_step<V: Value>(
        &mut self,
        y: nd::ArrayView1<V>,
        y_alt: nd::ArrayView1<V>,
    ) -> StepResult;
}

pub enum StepResult {
    Accept,
    Reject,
}

pub struct AdaptiveStepSizeManager {
    alpha: Float,
    beta: Float,
    safety_factor: Float,
    multiplier_bounds: [Float; 2],
    max_error: Float,

    abs_tol: Float,
    rel_tol: Float,

    step_size: Float,
    prev_error: Option<Float>,

    step_size_bounds: [Float; 2],
}

impl AdaptiveStepSizeManager {
    pub fn new(order: usize, init_step_size: Float) -> Self {
        Self {
            alpha: 0.7 / order as Float,
            beta: 0.4 / order as Float,
            safety_factor: 0.95,
            multiplier_bounds: [1. / 5., 5.],
            max_error: 1.5,
            abs_tol: 1e-5,
            rel_tol: 1e-5,
            step_size: init_step_size,
            prev_error: None,
            step_size_bounds: [1e-7, Float::INFINITY],
        }
    }

    pub fn with_abs_tol(self, abs_tol: Float) -> Self {
        Self { abs_tol, ..self }
    }

    pub fn with_rel_tol(self, rel_tol: Float) -> Self {
        Self { rel_tol, ..self }
    }

    pub fn with_max_error(self, max_error: Float) -> Self {
        Self { max_error, ..self }
    }

    pub fn with_step_size_bounds(self, step_size_bounds: [Float; 2]) -> Self {
        Self {
            step_size_bounds,
            ..self
        }
    }
}

impl StepSizePolicy for AdaptiveStepSizeManager {
    fn step_size(&self) -> Float {
        self.step_size
    }

    fn process_step<V: Value>(
        &mut self,
        y: nd::ArrayView1<V>,
        y_alt: nd::ArrayView1<V>,
    ) -> StepResult {
        let error = compute_scaled_truncation_error(y, y_alt, self.abs_tol, self.rel_tol);
        let multiplier = if let Some(prev_error) = self.prev_error {
            let upper = if prev_error > 1. {
                1.
            } else {
                self.multiplier_bounds[1]
            };
            let lower = self.multiplier_bounds[0];
            (self.safety_factor * error.powf(-self.alpha) * prev_error.powf(self.beta))
                .clamp(lower, upper)
        } else {
            self.safety_factor
                * error
                    .powf(-self.alpha)
                    .clamp(self.multiplier_bounds[0], self.multiplier_bounds[1])
        };
        self.prev_error = Some(error);

        self.step_size = (self.step_size * multiplier).min(self.step_size_bounds[1]);
        if self.step_size < self.step_size_bounds[0] {
            panic!("Step size became too small: {:?}", self.step_size);
        }

        if error > self.max_error {
            StepResult::Reject
        } else {
            StepResult::Accept
        }
    }
}

#[derive(Clone)]
pub struct FixedStepSize {
    step_size: Float,
}

impl FixedStepSize {
    pub fn new(step_size: Float) -> Self {
        Self { step_size }
    }
}

impl StepSizePolicy for FixedStepSize {
    fn step_size(&self) -> Float {
        self.step_size
    }

    fn process_step<V: Value>(
        &mut self,
        _y: ndarray::ArrayView1<V>,
        _y_alt: ndarray::ArrayView1<V>,
    ) -> StepResult {
        StepResult::Accept
    }
}

fn compute_scaled_truncation_error<V: Value>(
    y: nd::ArrayView1<V>,
    y_alt: nd::ArrayView1<V>,
    abs_tol: Float,
    rel_tol: Float,
) -> Float {
    (y.iter()
        .zip(y_alt.iter())
        .map(|(&yi, &yi_alt)| {
            let scale = abs_tol + rel_tol * yi_alt.norm_sq().max(yi.norm_sq()).sqrt();
            (yi - yi_alt).norm_sq() / scale.powi(2)
        })
        .sum::<Float>()
        / y.len() as Float)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ComplexFloat;

    struct TrigSystem {
        num_evals: std::cell::RefCell<usize>,
        omega: Float,
    }

    impl System for TrigSystem {
        type Value = ComplexFloat;

        fn system<S: nd::RawDataMut<Elem = Self::Value> + nd::Data + nd::DataMut>(
            &self,
            y: nd::ArrayView1<Self::Value>,
            mut dydt: nd::ArrayBase<S, nd::Ix1>,
        ) {
            dydt.assign(&(&y * ComplexFloat::new(0., self.omega)));
            *self.num_evals.borrow_mut() += 1;
        }
    }

    struct ExpSystem {}

    impl System for ExpSystem {
        type Value = Float;

        fn system<S: nd::RawDataMut<Elem = Self::Value> + nd::Data + nd::DataMut>(
            &self,
            y: nd::ArrayView1<Self::Value>,
            mut dydt: nd::ArrayBase<S, nd::Ix1>,
        ) {
            dydt.assign(&y);
        }
    }

    fn test_exp_impl<I: Integrator<ExpSystem>, F: FnOnce(&ExpSystem) -> I>(integrator_factory: F) {
        let system = ExpSystem {};
        let mut integrator = integrator_factory(&system);
        while integrator.current_solution().t < 2. {
            integrator.step(&system);
        }
        approx::assert_relative_eq!(
            integrator.current_solution().t.exp(),
            integrator.current_solution().y[[0]],
            epsilon = 1e-4,
        );
    }

    #[test]
    fn test_exp() {
        test_exp_impl(|_system| {
            IntegratorBulirschStoer::<ExpSystem>::new(nd::array![1.], 2.)
                .with_abs_tol(1e-4)
                .with_rel_tol(1e-4)
        });

        for order in 2..=5 {
            test_exp_impl(|system| {
                let fixed_step_size = FixedStepSize::new(1e-3);
                let integrator_nordsieck = IntegratorNordsieckAdamsBashforth::new(
                    order,
                    system,
                    nd::array![1.],
                    fixed_step_size.clone(),
                );
                fn factorial(num: usize) -> usize {
                    match num {
                        0 => 1,
                        1 => 1,
                        _ => factorial(num - 1) * num,
                    }
                }
                crate::test_util::assert_all_close(
                    &integrator_nordsieck.y_nordsieck.data,
                    &nd::Array1::from_iter((0..=order).map(|k| {
                        fixed_step_size.step_size().powi(k as i32) / factorial(k) as Float
                    }))
                    .slice(nd::s![nd::NewAxis, ..])
                    .to_owned(),
                )
                .with_abs_tol(1e-2 * integrator_nordsieck.y_nordsieck.data[[2, 0]])
                .with_print_ratio(true);
                integrator_nordsieck
            });
        }
    }

    #[test]
    fn test_trig() {
        test_trig_impl(
            |_system, y_init| {
                IntegratorBulirschStoer::new(y_init, 1.)
                    .with_abs_tol(1e-6)
                    .with_rel_tol(0.)
            },
            1612,
        );
        for (order, expected_num_system_evals) in [(2, 3906), (3, 1390), (4, 766), (5, 525)] {
            test_trig_impl(
                |system, y_init| {
                    let integrator = IntegratorNordsieckAdamsBashforth::new(
                        order,
                        system,
                        y_init,
                        AdaptiveStepSizeManager::new(order, 1e-2)
                            .with_abs_tol(1e-6)
                            .with_rel_tol(1e-6),
                    );
                    println!("{:?}", integrator.y_nordsieck);
                    integrator
                },
                expected_num_system_evals,
            );
        }
    }

    fn test_trig_impl<
        I: Integrator<TrigSystem>,
        F: FnOnce(&TrigSystem, nd::Array1<ComplexFloat>) -> I,
    >(
        integrator_factory: F,
        expected_num_system_evals: usize,
    ) {
        use ndrustfft::Zero;

        let system = TrigSystem {
            omega: 1.2,
            num_evals: std::cell::RefCell::new(0),
        };
        let mut integrator = integrator_factory(
            &system,
            nd::array![ComplexFloat::new(1., 0.), ComplexFloat::zero()],
        );

        loop {
            integrator.step(&system);
            let Solution { t, y } = integrator.current_solution();
            if t >= 51.7 {
                assert!(t < 55., "{:?}", t);
                crate::test_util::assert_all_close(
                    &y.mapv(|z| z.re),
                    &nd::array![ComplexFloat::new(0., 1.2 * t).exp().re, 0.],
                )
                .with_abs_tol(1e-2);
                crate::test_util::assert_all_close(
                    &y.mapv(|z| z.im),
                    &nd::array![ComplexFloat::new(0., 1.2 * t).exp().im, 0.],
                )
                .with_abs_tol(1e-2);
                break;
            }
            // Don't run forever even if integration has ground to a halt.
            assert!(
                *system.num_evals.borrow() < 10000,
                "t: {:?}, delta_t: {:?}",
                t,
                0.1
            );
        }
        assert_eq!(*system.num_evals.borrow(), expected_num_system_evals);
    }
}
