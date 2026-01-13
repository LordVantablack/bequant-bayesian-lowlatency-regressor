#include "../include/matrix.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <algorithm> 


Matrix::Matrix(size_t r, size_t c, double initial_val) 
    : rows(r), cols(c), data(r * c, initial_val) {}

Matrix::Matrix(size_t r, size_t c, const std::vector<double>& input_data) 
    : rows(r), cols(c), data(input_data) {
    if (input_data.size() != r * c) throw std::invalid_argument("Size mismatch");
}


double& Matrix::operator()(size_t r, size_t c) { return data[r * cols + c]; }
const double& Matrix::operator()(size_t r, size_t c) const { return data[r * cols + c]; }
size_t Matrix::get_rows() const { return rows; }
size_t Matrix::get_cols() const { return cols; }

void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) std::cout << (*this)(i, j) << "\t";
        std::cout << "\n";
    }
}


Matrix Matrix::transpose() const {
    Matrix T(cols, rows);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            T(j, i) = (*this)(i, j);
    return T;
}

Matrix Matrix::mat_mul(const Matrix& B) const {
    if (cols != B.rows) throw std::invalid_argument("Dimension mismatch");
    Matrix C(rows, B.cols, 0.0);
    const size_t TILE = 32;

    for (size_t i = 0; i < rows; i += TILE)
        for (size_t j = 0; j < B.cols; j += TILE)
            for (size_t k = 0; k < cols; k += TILE)
                for (size_t ii = i; ii < std::min(i + TILE, rows); ++ii)
                    for (size_t kk = k; kk < std::min(k + TILE, cols); ++kk) {
                        double val = (*this)(ii, kk);
                        for (size_t jj = j; jj < std::min(j + TILE, B.cols); ++jj)
                            C(ii, jj) += val * B(kk, jj);
                    }
    return C;
}

std::vector<double> Matrix::vec_mul(const std::vector<double>& vec) const {
    if (cols != vec.size()) throw std::invalid_argument("Vector size mismatch");
    std::vector<double> res(rows, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            res[i] += (*this)(i, j) * vec[j];
    return res;
}

void Matrix::cholesky_inplace() {
    if (rows != cols) throw std::invalid_argument("Not square");
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = (*this)(i, j);
            for (size_t k = 0; k < j; ++k) sum -= (*this)(i, k) * (*this)(j, k);
            
            if (i == j) {
                if (sum <= 0) throw std::runtime_error("Matrix not positive definite");
                (*this)(i, i) = std::sqrt(sum);
            } else {
                (*this)(i, j) = sum / (*this)(j, j);
            }
        }
    }
}

void Matrix::cholesky_solve(std::vector<double>& b) const {

    for (size_t i = 0; i < rows; ++i) {
        double sum = b[i];
        for (size_t j = 0; j < i; ++j) sum -= (*this)(i, j) * b[j];
        b[i] = sum / (*this)(i, i);
    }

    for (int i = rows - 1; i >= 0; --i) {
        double sum = b[i];
        for (size_t j = i + 1; j < rows; ++j) sum -= (*this)(j, i) * b[j];
        b[i] = sum / (*this)(i, i);
    }
}