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

# Copy WASM files to web folder
echo "📁 Copying WASM files to web folder..."
cp pkg/image_optimizer.js web/
cp pkg/image_optimizer_bg.wasm web/

echo "✅ Build complete! All files are in ./web/"
echo ""
echo "To start the web server, run: ./serve.sh"

