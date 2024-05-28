use ndarray as nd;

use crate::{ComplexFloat, Float};

pub trait System {
    type Value;

    fn system<S: nd::RawDataMut<Elem = Self::Value> + nd::Data + nd::DataMut>(
        &self,
        y: &nd::Array1<Self::Value>,
        dydt: &mut nd::ArrayBase<S, nd::Ix1>,
    );
}

/// An explicit, adaptive-stepsize ODE integrator using RK4.
pub struct Integrator<S: System> {
    t_prev: Float,
    y_prev: nd::Array1<S::Value>,
    t_next: Float,
    y_next: nd::Array1<S::Value>,
    t_now: Float,

    step_size: Float,
    min_step_size: Float,

    abs_tol: Float,
    rel_tol: Float,

    // Scratch arrays.
    y_half_step: nd::Array1<S::Value>,
    y_full_step: nd::Array1<S::Value>,
    k1: nd::Array1<S::Value>,
    k2: nd::Array1<S::Value>,
    k3: nd::Array1<S::Value>,
    k4: nd::Array1<S::Value>,
}

impl<S: System> Integrator<S>
where
    S::Value: Copy
        + Clone
        + nd::ScalarOperand
        + ndrustfft::Zero
        + std::ops::Mul<S::Value, Output = S::Value>
        + std::ops::MulAssign<S::Value>
        + std::ops::AddAssign<S::Value>
        + std::ops::Sub<S::Value, Output = S::Value>
        + From<Float>
        + NormSq,
{
    pub fn new(y_init: nd::Array1<S::Value>, step_size_init: Float, min_step_size: Float) -> Self {
        let k1 = y_init.clone();
        let k2 = y_init.clone();
        let k3 = y_init.clone();
        let k4 = y_init.clone();
        let y_half_step = y_init.clone();
        let y_full_step = y_init.clone();

        Self {
            y_prev: y_init.clone(),
            y_next: y_init,
            t_prev: 0.,
            t_next: 0.,
            t_now: 0.,
            step_size: step_size_init,
            min_step_size,
            abs_tol: 1e-5,
            rel_tol: 1e-5,
            y_half_step,
            y_full_step,
            k1,
            k2,
            k3,
            k4,
        }
    }

    pub fn abs_tol(self, abs_tol: Float) -> Self {
        Self { abs_tol, ..self }
    }
    pub fn rel_tol(self, rel_tol: Float) -> Self {
        Self { rel_tol, ..self }
    }

    pub fn y(&self) -> nd::Array1<S::Value> {
        // `t_now` should be bracketed between `t_prev` and `t_next`. Linearly interpolate so we hit
        // `t_now`.
        if self.t_next > self.t_prev {
            let weight_prev =
                S::Value::from((self.t_next - self.t_now) / (self.t_next - self.t_prev));
            let weight_next =
                S::Value::from((self.t_now - self.t_prev) / (self.t_next - self.t_prev));
            let mut y = self.y_prev.clone();
            y *= weight_prev;
            y += &(&self.y_next * weight_next);
            y
        } else {
            self.y_prev.clone()
        }
    }

    pub fn integrate(&mut self, system: &S, delta_t: Float) {
        assert!(delta_t >= 0.);

        // Put dummy values for these arrays to avoid violating borrowing rules. We'll swap the
        // dummies out at the end.
        let mut y_prev = nd::Array::zeros(0);
        let mut y_next = nd::Array::zeros(0);
        let mut y_half_step = nd::Array::zeros(0);
        let mut y_full_step = nd::Array::zeros(0);
        std::mem::swap(&mut y_prev, &mut self.y_prev);
        std::mem::swap(&mut y_next, &mut self.y_next);
        std::mem::swap(&mut y_half_step, &mut self.y_half_step);
        std::mem::swap(&mut y_full_step, &mut self.y_full_step);

        self.t_now += delta_t;

        // Loop until we've passed `self.t_now`. At the end of this loop, `self.t_next` should be
        // `>= self.t_now`, `self.t_prev` should be `< self.t_now`, `y_next` should be the value of
        // `y` integrated to `self.t_next`, and `y_prev` should be the value of `y` integrated to
        // `self.t_prev`. This is used to linearly interpolate in `y()`.
        while self.t_next < self.t_now {
            self.t_prev = self.t_next;
            y_prev.assign(&y_next);

            // Full step.
            self.step(system, self.step_size, &y_prev, &mut y_full_step, true);

            // Two half steps.
            let half_step_size = self.step_size / 2.;
            self.step(system, half_step_size, &y_prev, &mut y_half_step, false);
            self.step(system, half_step_size, &y_half_step, &mut y_next, true);

            // Compute scaled truncation error between two half steps and full step.
            let error: Float = (y_full_step
                .iter()
                .copied()
                .zip(y_next.iter().copied())
                .map(|(yi_full_step, yi_next)| {
                    let scale = self.abs_tol
                        + self.rel_tol * yi_next.norm_sq().max(yi_full_step.norm_sq()).sqrt();
                    (yi_full_step - yi_next).norm_sq() / scale.powi(2)
                })
                .sum::<Float>()
                / y_next.len() as Float)
                .sqrt();

            if error > 5. {
                // Error is unacceptable. Halve the step size and try again.
                self.step_size /= 2.;
                assert!(self.step_size > self.min_step_size, "Error: {:?}", error,);

                y_next.assign(&y_prev);
                continue;
            }

            self.t_next += self.step_size;

            // Simple PI controller for step size.
            let step_size_multiplier = (0.9 * error.powf(-0.1)).min(6.).max(0.3);
            self.step_size *= step_size_multiplier;
            assert!(self.step_size > self.min_step_size, "Error: {:?}", error);
        }

        std::mem::swap(&mut y_prev, &mut self.y_prev);
        std::mem::swap(&mut y_next, &mut self.y_next);
        std::mem::swap(&mut y_half_step, &mut self.y_half_step);
        std::mem::swap(&mut y_full_step, &mut self.y_full_step);
    }

    /// Take a single step. Only mutable since it uses the scratch `ki` arrays.
    fn step(
        &mut self,
        system: &S,
        step_size: Float,
        y: &nd::Array1<S::Value>,
        y_next: &mut nd::Array1<S::Value>,
        compute_k1: bool,
    ) {
        // Compute `k1`.
        if compute_k1 {
            system.system(y, &mut self.k1);
        }

        // Compute `k2`.
        y_next.assign(&self.k1);
        *y_next *= S::Value::from(step_size / 2.);
        *y_next += y;
        system.system(&y_next, &mut self.k2);

        // Compute `k3`.
        y_next.assign(&self.k2);
        *y_next *= S::Value::from(step_size / 2.);
        *y_next += y;
        system.system(&y_next, &mut self.k3);

        // Compute `k4`.
        y_next.assign(&self.k3);
        *y_next *= S::Value::from(step_size);
        *y_next += y;
        system.system(&y_next, &mut self.k4);

        // Populate `y_next`.
        let weight_end = S::Value::from(step_size / 6.);
        let weight_mid = S::Value::from(step_size / 3.);
        y_next.assign(y);
        *y_next += &(&self.k1 * weight_end);
        *y_next += &(&self.k2 * weight_mid);
        *y_next += &(&self.k3 * weight_mid);
        *y_next += &(&self.k4 * weight_end);
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

#[cfg(test)]
mod tests {
    use super::*;

    struct TrigSystem {
        num_evals: std::cell::RefCell<usize>,
        omega: Float,
    }

    impl System for TrigSystem {
        type Value = ComplexFloat;

        fn system<S: nd::RawDataMut<Elem = Self::Value> + nd::Data + nd::DataMut>(
            &self,
            y: &nd::Array1<Self::Value>,
            dydt: &mut nd::ArrayBase<S, nd::Ix1>,
        ) {
            dydt.assign(&(y * ComplexFloat::new(0., self.omega)));
            *self.num_evals.borrow_mut() += 1;
        }
    }

    #[test]
    fn test_trig() {
        use ndrustfft::Zero;

        let system = TrigSystem {
            omega: 1.2,
            num_evals: std::cell::RefCell::new(0),
        };
        let mut integrator = Integrator::new(
            nd::array![ComplexFloat::new(1., 0.), ComplexFloat::zero()],
            1.,
            1e-10,
        )
        .abs_tol(1e-6)
        .rel_tol(0.);

        integrator.integrate(&system, 51.7);

        assert_eq!(*system.num_evals.borrow(), 4763);

        crate::test_util::assert_all_close(
            &integrator.y().mapv(|z| z.re),
            &nd::array![ComplexFloat::new(0., 1.2 * 51.7).exp().re, 0.],
        )
        .abs_tol(Some(1e-2));
        crate::test_util::assert_all_close(
            &integrator.y().mapv(|z| z.im),
            &nd::array![ComplexFloat::new(0., 1.2 * 51.7).exp().im, 0.],
        )
        .abs_tol(Some(1e-2));

        integrator.integrate(&system, 11.7);

        assert_eq!(*system.num_evals.borrow(), 5830);

        crate::test_util::assert_all_close(
            &integrator.y().mapv(|z| z.re),
            &nd::array![ComplexFloat::new(0., 1.2 * (51.7 + 11.7)).exp().re, 0.],
        )
        .abs_tol(Some(1e-2));
        crate::test_util::assert_all_close(
            &integrator.y().mapv(|z| z.im),
            &nd::array![ComplexFloat::new(0., 1.2 * (51.7 + 11.7)).exp().im, 0.],
        )
        .abs_tol(Some(1e-2));
    }
}
