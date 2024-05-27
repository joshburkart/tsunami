use pyo3::prelude::*;

mod examples;

#[pymodule]
fn pyflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(examples::bump_2d_spectral, m)?)
}
