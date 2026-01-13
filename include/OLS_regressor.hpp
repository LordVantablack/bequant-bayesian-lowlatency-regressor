#ifndef OLS_REGRESSOR_HPP
#define OLS_REGRESSOR_HPP

#include "matrix.hpp"
#include <vector>

class OLS_Regressor {
private:

    std::vector<double> coefficients; 
    bool is_fitted;

public:
    OLS_Regressor();


    void fit(const Matrix& X, const std::vector<double>& y);

    std::vector<double> predict(const Matrix& X) const;


    std::vector<double> get_coeffs() const;
};

#endif