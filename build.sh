#!/bin/bash
SOURCE=${1:-main.cc}
OUTPUT=${2:-main}
BACKEND_MODE=${3:-cpu}

# Detect CPU architecture (uname -m: machine hardware name)
ARCH=$(uname -m)

# Set the multiarch triplet directory based on architecture.
# On Debian/Ubuntu, system libs/headers live under /usr/{lib,include}/<triplet>.
# x86_64 (Intel/AMD 64-bit) -> x86_64-linux-gnu
# aarch64 (ARM 64-bit, e.g. Jetson Orin Nano) -> aarch64-linux-gnu
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    TRIPLET="aarch64-linux-gnu"
    # On Jetson, -march=native is not supported by GCC for aarch64.
    # Use -mcpu=native instead (selects CPU-specific tuning + ISA extensions).
    ARCH_FLAGS="-mcpu=native"
else
    TRIPLET="x86_64-linux-gnu"
    # -march=native: generate code for the host CPU's instruction set.
    ARCH_FLAGS="-march=native"
fi

COMMON_SOURCES=(
    "$SOURCE"
    inference/backend_factory.cc
    inference/cpp_frangi_backend.cc
    inference/tensorrt_backend.cc
)

PKG_FLAGS=($(pkg-config --cflags --libs opencv4))
CXXFLAGS=(-g -std=c++20 -fopenmp -O3 "$ARCH_FLAGS")
EXTRA_FLAGS=(-pthread)

case "$BACKEND_MODE" in
    cpu)
        ;;
    tensorrt)
        CXXFLAGS+=(-DUSE_TENSORRT)
        EXTRA_FLAGS+=(
            -isystem /usr/local/cuda/include
            -isystem /usr/include/$TRIPLET
            -L/usr/local/cuda/lib64
            -L/usr/lib/$TRIPLET
            -lnvinfer
            -lnvonnxparser
            -lcudart
        )
        ;;
    *)
        echo "Unknown backend mode '$BACKEND_MODE' (expected 'cpu' or 'tensorrt')"
        exit 1
        ;;
esac

g++ "${CXXFLAGS[@]}" -o "$OUTPUT" "${COMMON_SOURCES[@]}" "${PKG_FLAGS[@]}" \
    "${EXTRA_FLAGS[@]}"
