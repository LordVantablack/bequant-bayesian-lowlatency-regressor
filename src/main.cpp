#include "../include/matrix.hpp"
#include "../include/OLS_regressor.hpp"
#include <iostream>
#include <vector>
#include <fstream>  // File I/O
#include <sstream>  // String parsing
#include <string>

// --- Helper: Read CSV ---
// Reads a CSV file where the LAST column is 'y' and the rest are 'X'.
// Returns pair: {Matrix X, Vector y}
std::pair<Matrix, std::vector<double>> load_data(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<double> all_data;
    std::string line;
    size_t rows = 0;
    size_t cols = 0;

    // 1. Read file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val_str;
        double val;
        size_t current_cols = 0;

        while (std::getline(ss, val_str, ',')) { // Split by comma
            all_data.push_back(std::stod(val_str));
            current_cols++;
        }
        
        if (rows == 0) cols = current_cols; // Set column count on first row
        rows++;
    }

    if (rows == 0) throw std::runtime_error("File is empty");

    // 2. Separate into X and y
    // We have 'rows' samples.
    // Total columns in CSV is 'cols'. 
    // Features = cols - 1.
    // Target = 1 (Last column).
    
    size_t num_features = cols - 1;
    
    // Create X Matrix (Rows x (Features + 1 for Intercept))
    // We add +1 col for the intercept (Bias term)
    Matrix X(rows, num_features + 1);
    std::vector<double> y(rows);

    for (size_t i = 0; i < rows; ++i) {
        // Set Intercept (First column is always 1.0)
        X(i, 0) = 1.0;

        for (size_t j = 0; j < num_features; ++j) {
            // CSV Data Index: i * cols + j
            // Matrix X Index: i, j+1 (because 0 is intercept)
            X(i, j + 1) = all_data[i * cols + j];
        }
        
        // Last column is y
        y[i] = all_data[i * cols + num_features];
    }

    return {X, y};
}

int main() {
    try {
        std::cout << "Loading data from data/data.csv..." << std::endl;
        
        // 1. Load Data
        auto [X, y] = load_data("data/data.csv"); // C++17 Structured Binding
        
        std::cout << "Data Loaded." << std::endl;
        std::cout << "Samples: " << X.get_rows() << std::endl;
        std::cout << "Features: " << X.get_cols() - 1 << " (+1 Intercept)" << std::endl;

        // 2. Train Model
        OLS_Regressor model;
        std::cout << "Training model..." << std::endl;
        
        // Simple timer
        auto start = std::chrono::high_resolution_clock::now();
        model.fit(X, y);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        std::cout << "Training complete in " << diff.count() << " s" << std::endl;

        // 3. Show First 5 Coefficients
        std::cout << "First 5 Coefficients (Beta): ";
        std::vector<double> coeffs = model.get_coeffs();
        for (size_t i = 0; i < std::min(coeffs.size(), (size_t)5); ++i) {
            std::cout << coeffs[i] << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}