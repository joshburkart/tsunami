use crate::Float;

use ndarray as nd;

#[cfg(test)]
pub struct AllCloseAssertion<'a, 'b, V, D: nd::Dimension>
where
    V: std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>
        + std::ops::Div<V, Output = V>
        + Copy,
{
    left: &'a nd::ArrayBase<nd::OwnedRepr<V>, D>,
    right: &'b nd::ArrayBase<nd::OwnedRepr<V>, D>,

    rel_tol: Float,
    abs_tol: Float,

    print_ratio: bool,
    max_mismatching_elements: usize,
}
#[cfg(test)]
impl<'a, 'b, V, D: nd::Dimension> AllCloseAssertion<'a, 'b, V, D>
where
    V: std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>
        + std::ops::Div<V, Output = V>
        + Copy,
{
    pub fn with_rel_tol(mut self, rel_tol: Float) -> Self {
        self.rel_tol = rel_tol;
        self
    }
    pub fn with_abs_tol(mut self, abs_tol: Float) -> Self {
        self.abs_tol = abs_tol;
        self
    }
    #[allow(dead_code)]
    pub fn with_max_mismatching_elements(mut self, max_mismatching_elements: usize) -> Self {
        self.max_mismatching_elements = max_mismatching_elements;
        self
    }
    pub fn with_print_ratio(mut self, print_ratio: bool) -> Self {
        self.print_ratio = print_ratio;
        self
    }
}

#[cfg(test)]
impl<'a, 'b, V, D: nd::Dimension> Drop for AllCloseAssertion<'a, 'b, V, D>
where
    V: std::fmt::Display
        + approx::AbsDiffEq<V, Epsilon = Float>
        + approx::RelativeEq<V, Epsilon = Float>
        + std::ops::Div<V, Output = V>
        + Copy,
{
    #[track_caller]
    fn drop(&mut self) {
        let mut mismatching_elements = 0;
        self.left
            .indexed_iter()
            .zip(self.right.iter())
            .for_each(|((index, left), right)| {
                let checker = approx::Relative::default()
                    .max_relative(self.rel_tol)
                    .epsilon(self.abs_tol);

                if !checker.eq(&left, &right) {
                    if mismatching_elements < self.max_mismatching_elements {
                        if self.print_ratio {
                            eprintln!(
                                "At {index:?}, left = {left}, right = {right}; ratio = {}",
                                *left / *right
                            );
                        } else {
                            eprintln!("At {index:?}, left = {left}, right = {right}");
                        }
                    }
                    mismatching_elements += 1;
                }
            });
        if mismatching_elements > 0 {
            panic!(
                "Didn't match at {mismatching_elements}/{} elements",
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
        + approx::RelativeEq<V, Epsilon = Float>
        + Copy,
    D: nd::Dimension,
>(
    left: &'a nd::ArrayBase<nd::OwnedRepr<V>, D>,
    right: &'b nd::ArrayBase<nd::OwnedRepr<V>, D>,
) -> AllCloseAssertion<'a, 'b, V, D>
where
    V: std::ops::Div<V, Output = V>,
{
    AllCloseAssertion {
        left,
        right,
        rel_tol: 1e-5,
        abs_tol: 0.,
        print_ratio: false,
        max_mismatching_elements: 20,
    }
}
