# Rust Image Optimizer

A fast, browser-based image optimizer built with Rust and WebAssembly. Optimize JPEG and PNG images directly in your browser with no server uploads required.


## Features

- **100% Client-side** — Images never leave your browser
- **JPEG Optimization** — Adjustable quality (1-100) with fast encoding
- **PNG Optimization** — Multiple compression levels (0-9)
- **Batch Processing** — Upload and optimize multiple images at once
- **Drag & Drop** — Easy file upload interface
- **Individual Downloads** — Download optimized images one by one
- **ZIP Download** — Download all optimized images as a single ZIP file
- **Real-time Stats** — See original vs optimized size and total savings

## Building

### Prerequisites

- [Rust](https://rustup.rs/) (latest stable)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

### Build Steps

```bash
# Clone the repository
cd image-optimizer

# Build the WASM package
wasm-pack build --target web --release

# Serve the web directory (requires a local server due to WASM CORS)
# Option 1: Python
python3 -m http.server 8080

# Option 2: Node.js
npx serve .

# Option 3: Any static file server
```

Then open http://localhost:8080/web/ in your browser.

## Project Structure

```
image-optimizer/
├── Cargo.toml          # Rust dependencies
├── src/
│   └── lib.rs          # Rust WASM library
├── pkg/                # Generated WASM package (after build)
│   ├── image_optimizer.js
│   ├── image_optimizer_bg.wasm
│   └── ...
└── web/
    └── index.html      # Web interface
```

## How It Works

1. **JPEG Optimization**: Uses the `jpeg-encoder` crate to re-encode JPEG images at a specified quality level. Lower quality = smaller file size.

2. **PNG Optimization**: Uses the `png` crate with configurable compression levels. Higher compression = smaller file but slower processing.

3. **Metadata Stripping**: When enabled, EXIF and other metadata is removed during re-encoding (enabled by default).

## Browser Compatibility

Works in all modern browsers that support WebAssembly:
- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

## License

MIT

