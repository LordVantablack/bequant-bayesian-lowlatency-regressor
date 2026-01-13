
---


# Bequant Bayesian Low-Latency Regressor


## 📂 Directory Structure

| Directory/File | Description |
| :--- | :--- |
| `src/` | Source code (`.cpp`) for the regression logic and main application. |
| `include/` | Header files (`.h`/`.hpp`) defining the model interfaces. |
| `data/` | Directory for storing input datasets or generated test data. |
| `generate_data.py` | Python script to create synthetic datasets for testing the model. |
| `Makefile` / `CMakeLists.txt` | Build configuration files. |

## Prerequisites

Before building the project, ensure you have the following installed:

* **C++ Compiler**: GCC, Clang, or MSVC supporting C++17 or later.
* **CMake**: Version 3.10 or higher.
* **Python 3.x**: (Optional) Required if you wish to run the data generation scripts.
* *dependencies*: [List any libraries used here, e.g., Eigen3, Armadillo, etc., or remove this line if none]

## Building the Project

The project uses **CMake** for building. Follow these steps to compile the application:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/LordVantablack/bequant-bayesian-lowlatency-regressor.git](https://github.com/LordVantablack/bequant-bayesian-lowlatency-regressor.git)
    cd bequant-bayesian-lowlatency-regressor
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Configure and Compile:**
    ```bash
    cmake ..
    make
    ```

    *Alternatively, if you prefer using the provided Makefile directly in the root:*
    ```bash
    make
    ```

## 💻 Usage

### 1. Generating Test Data
To test the model with fresh data, use the provided Python script. This will populate the `data/` directory.

```bash
python3 generate_data.py

```

### 2. Running the Regressor

Once built, the binary `bayesian_regressor` will be located in the `build/` directory.

```bash
./build/bayesian_regressor

```

> **Note:** Currently, running the executable executes a hardcoded test case to verify the mathematical logic and performance of the regressor.



