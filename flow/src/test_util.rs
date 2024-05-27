use crate::Float;

use ndarray as nd;

#[cfg(test)]
pub struct AllCloseAssertion<'a, 'b, V, D: nd::Dimension>
where
    V: std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>,
{
    left: &'a nd::ArrayBase<nd::OwnedRepr<V>, D>,
    right: &'b nd::ArrayBase<nd::OwnedRepr<V>, D>,

    rel_tol: Option<Float>,
    abs_tol: Option<Float>,
}
#[cfg(test)]
impl<'a, 'b, V, D: nd::Dimension> AllCloseAssertion<'a, 'b, V, D>
where
    V: std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>,
{
    #[allow(dead_code)]
    pub fn rel_tol(&mut self, rel_tol: Option<Float>) -> &mut Self {
        self.rel_tol = rel_tol;
        self
    }

    pub fn abs_tol(&mut self, abs_tol: Option<Float>) -> &mut Self {
        self.abs_tol = abs_tol;
        self
    }
}

#[cfg(test)]
impl<'a, 'b, V, D: nd::Dimension> Drop for AllCloseAssertion<'a, 'b, V, D>
where
    V: std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>,
{
    #[track_caller]
    fn drop(&mut self) {
        if self.rel_tol.is_none() && self.abs_tol.is_none() {
            panic!("At least one tolerance must be specified");
        }
        let mut num_failures = 0;
        self.left
            .indexed_iter()
            .zip(self.right.iter())
            .for_each(|((index, left), right)| {
                let mut checker = approx::Relative::default();
                if let Some(rel_tol) = self.rel_tol {
                    checker = checker.max_relative(rel_tol);
                }
                if let Some(abs_tol) = self.abs_tol {
                    checker = checker.epsilon(abs_tol);
                }

                if !checker.eq(&left, &right) {
                    if num_failures < 20 {
                        eprintln!("At {index:?}, left = {left}, right = {right}");
                    }
                    num_failures += 1;
                }
            });
        if num_failures > 0 {
            panic!(
                "Didn't match at {num_failures}/{} elements",
                self.left.len()
            )
        }
    }
}

#[track_caller]
pub fn assert_all_close<
    'a,
    'b,
    V: std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>,
    D: nd::Dimension,
>(
    left: &'a nd::ArrayBase<nd::OwnedRepr<V>, D>,
    right: &'b nd::ArrayBase<nd::OwnedRepr<V>, D>,
) -> AllCloseAssertion<'a, 'b, V, D> {
    AllCloseAssertion {
        left,
        right,
        rel_tol: Some(1e-7),
        abs_tol: Some(0.),
    }
}
