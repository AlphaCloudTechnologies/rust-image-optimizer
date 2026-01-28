#!/bin/bash
set -e

echo "🔧 Building WASM image optimizer..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build the WASM package
echo "📦 Compiling to WebAssembly..."
wasm-pack build --target web --release

echo "✅ Build complete! WASM files are in ./pkg/"
echo ""
echo "To start the web server, run: ./serve.sh"

