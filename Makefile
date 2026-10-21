# Compilers
NVCC        := nvcc
CXX         := g++

# Build Mode (default is normal/release)
MODE        ?= normal

# Base Compiler flags (Normal mode)
NVCCFLAGS   := -O3 -Xcompiler -Wall
CXXFLAGS    := -O3 -Wall -std=c++17 -fopenmp -march=native -mfma -ffast-math

# Perf/Debug mode additions
ifeq ($(MODE),perf)
    # -g: host debug, -G: device debug, -fno-omit-frame-pointer: perf call graphs
    NVCCFLAGS += -g -G -Xcompiler -fno-omit-frame-pointer
    CXXFLAGS  += -g -fno-omit-frame-pointer -fopenmp
    $(info Building in PERF mode (Debug symbols & frame pointers enabled))
else
    $(info Building in NORMAL mode (Optimized, no debug symbols))
endif

# Target executable name
TARGET      := my_program

# Source and Object files
CUDA_SRCS   := acceleration.cu
CXX_SRCS    := main.cpp

CUDA_OBJS   := $(CUDA_SRCS:.cu=.o)
CXX_OBJS    := $(CXX_SRCS:.cpp=.o)
OBJS        := $(CUDA_OBJS) $(CXX_OBJS)

# Libraries (Link CUDA runtime if needed by the C++ linker)
LIBS        := -lcudart

# Default rule
all: $(TARGET)

# Link step
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean

