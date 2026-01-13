#include "../include/OLS_regressor.hpp"
#include <stdexcept>
#include <iostream>

OLS_Regressor::OLS_Regressor() : is_fitted(false) {}

void OLS_Regressor::fit(const Matrix& X, const std::vector<double>& y) {
    if (X.get_rows() != y.size()) {
        throw std::invalid_argument("Sample size mismatch between X and y");
    }

    // 1. Calculate X_transpose
    Matrix Xt = X.transpose();

    // 2. Compute Normal Equation components
    // A = Xt * X
    Matrix A = Xt.mat_mul(X);
    
    // b = Xt * y
    // We store this directly into 'coefficients' to solve in-place later
    coefficients = Xt.vec_mul(y);

    // 3. Solve for Beta using Cholesky
    // Decompose A (in-place)
    A.cholesky_inplace();
    
    // Solve system (in-place on coefficients vector)
    A.cholesky_solve(coefficients);

    is_fitted = true;
}

std::vector<double> OLS_Regressor::predict(const Matrix& X) const {
    if (!is_fitted) throw std::runtime_error("Model not fitted yet.");
    
    // Prediction is simply X * Beta
    return X.vec_mul(coefficients);
}

std::vector<double> OLS_Regressor::get_coeffs() const {
    return coefficients;
}