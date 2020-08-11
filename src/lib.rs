/*
Copyright 2020 Derrick W. Turk | terminus data science, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

use pyo3::{
    prelude::*,
    exceptions::ValueError,
};

use numpy::{
    PyArray1,
};

#[pymodule]
fn qed(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "solve_least_squares")]
    fn solve_least_squares(py: Python) -> PyResult<&PyAny> {
        let s_opt = PyModule::import(py, "scipy.optimize")?;
        s_opt.call1("least_squares", (Residuals { }, [0.0, 0.0]))
    }

    Ok(())
}

#[pyclass]
struct Residuals { }

#[pymethods]
impl Residuals {
    #[call]
    fn __call__<'py>(&self, py: Python<'py>, theta: &PyArray1<f64>
      ) -> PyResult<&'py PyArray1<f64>> {
        residuals(py, theta)
    }
}

const X: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];
const Y: [f64; 5] = [3.0, 7.0, 9.0, 21.0, 8.0];

/* evaluate residuals of f(x|theta) = theta_0*x^3 + theta_1*x^2 against a
 * fixed series of target values f(x) */
fn residuals<'py>(py: Python<'py>, theta: &PyArray1<f64>
  ) -> PyResult<&'py PyArray1<f64>> {
    match theta.readonly().as_slice()? {
        [theta0, theta1] => {
            let resid: Vec<_> = X.iter().zip(Y.iter())
                .map(|(x, y)| theta0 * x * x * x + theta1 * x * x - y)
                .collect();
            Ok(PyArray1::from_vec(py, resid))
        },

        _ => ValueError::into("expected two-element array"),
    }
}
