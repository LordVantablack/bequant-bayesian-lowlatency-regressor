# Compiler
CXX = g++

# Flags: -Iinclude tells compiler where to look for .hpp files
CXXFLAGS = -std=c++17 -O3 -Iinclude -Wall

# Target executable name
TARGET = regressor

# Source files
SRCS = src/main.cpp src/matrix.cpp src/OLS_regressor.cpp

# Default rule
all:
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)