#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <cstddef> 

class Matrix {
private:
    std::vector<double> data;
    size_t rows;
    size_t cols;

public:
  
    Matrix(size_t r, size_t c, double initial_val = 0.0);
    Matrix(size_t r, size_t c, const std::vector<double>& input_data);

    double& operator()(size_t r, size_t c);
    const double& operator()(size_t r, size_t c) const;
    size_t get_rows() const;
    size_t get_cols() const;
    void print() const;


    Matrix transpose() const;
    Matrix mat_mul(const Matrix& B) const;
    std::vector<double> vec_mul(const std::vector<double>& vec) const;
    
 
    void cholesky_inplace();
    void cholesky_solve(std::vector<double>& b) const;
};

#endif