// Web Worker for image optimization
// Runs WASM processing off the main thread to keep UI responsive

import init, { optimize_image, convert_png_to_jpeg } from './image_optimizer.js';

let wasmReady = false;

// Initialize WASM
async function initWasm() {
    try {
        await init();
        wasmReady = true;
        self.postMessage({ type: 'ready' });
    } catch (error) {
        self.postMessage({ type: 'error', error: 'Failed to initialize WASM: ' + error.message });
    }
}

// Process a single image
function processImage(data, filename, options, id) {
    console.log('📨 Worker received options:', JSON.stringify(options, null, 2));
    
    if (!wasmReady) {
        return {
            id,
            success: false,
            error: 'WASM not initialized'
        };
    }

    try {
        const result = optimize_image(data, filename, options);
        return {
            id,
            type: 'result',
            result: {
                success: result.success,
                error: result.error,
                format: result.format,
                original_size: result.original_size,
                optimized_size: result.optimized_size,
                original_width: result.original_width,
                original_height: result.original_height,
                new_width: result.new_width,
                new_height: result.new_height,
                original_palette: result.original_palette || [],
                optimized_palette: result.optimized_palette || [],
                original_colors: result.original_colors || 0,
                optimized_colors: result.optimized_colors || 0,
                // Transfer the data array
                data: result.success ? new Uint8Array(result.data) : null
            }
        };
    } catch (error) {
        return {
            id,
            type: 'result',
            result: {
                success: false,
                error: error.message
            }
        };
    }
}

// Convert PNG to JPEG
function convertPngToJpeg(data, filename, options, id) {
    if (!wasmReady) {
        return {
            id,
            success: false,
            error: 'WASM not initialized'
        };
    }

    try {
        const result = convert_png_to_jpeg(data, filename, options);
        return {
            id,
            type: 'convert_result',
            result: {
                success: result.success,
                error: result.error,
                format: result.format,
                filename: result.filename,
                original_size: result.original_size,
                optimized_size: result.optimized_size,
                original_width: result.original_width,
                original_height: result.original_height,
                new_width: result.new_width,
                new_height: result.new_height,
                original_palette: result.original_palette || [],
                optimized_palette: result.optimized_palette || [],
                original_colors: result.original_colors || 0,
                optimized_colors: result.optimized_colors || 0,
                data: result.success ? new Uint8Array(result.data) : null
            }
        };
    } catch (error) {
        return {
            id,
            type: 'convert_result',
            result: {
                success: false,
                error: error.message
            }
        };
    }
}

// Handle messages from main thread
self.onmessage = function(e) {
    const { type, id, data, filename, options } = e.data;

    switch (type) {
        case 'optimize':
            const optimizeResult = processImage(new Uint8Array(data), filename, options, id);
            self.postMessage(optimizeResult);
            break;

        case 'convert':
            const convertResult = convertPngToJpeg(new Uint8Array(data), filename, options, id);
            self.postMessage(convertResult);
            break;

        default:
            self.postMessage({ type: 'error', error: 'Unknown message type: ' + type });
    }
};

// Initialize on worker start
initWasm();

