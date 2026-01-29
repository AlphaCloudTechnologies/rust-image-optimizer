use color_quant::NeuQuant;
use image::{GenericImageView, ImageFormat};
use jpeg_encoder::{ColorType, Encoder};
use png::{BitDepth, ColorType as PngColorType, Compression, Filter};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OptimizeOptions {
    pub quality: u8,             // 1-100 for JPEG
    pub png_compression: u8,     // 0-9 for PNG compression level
    pub strip_metadata: bool,    // Remove EXIF and other metadata
    pub png_quantize: bool,      // Enable color quantization for PNG (lossy)
    pub png_colors: u16,         // Target colors for quantization (2-256)
    pub png_auto: bool,          // Auto mode: automatically find best settings
    pub png_auto_level: u8,      // Auto mode level: 1=Light, 2=Balanced, 3=Maximum
    pub resize_enabled: bool,    // Enable resizing
    pub max_width: u32,          // Maximum width
    pub max_height: u32,         // Maximum height
}

#[derive(Serialize, Deserialize)]
pub struct OptimizeResult {
    pub data: Vec<u8>,
    pub original_size: usize,
    pub optimized_size: usize,
    pub format: String,
    pub filename: String,
    pub success: bool,
    pub error: Option<String>,
    pub original_width: u32,
    pub original_height: u32,
    pub new_width: u32,
    pub new_height: u32,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            quality: 85,
            png_compression: 6,
            strip_metadata: true,
            png_quantize: false,
            png_colors: 256,
            png_auto: true,  // Auto mode enabled by default
            png_auto_level: 5,  // Default to balanced (1-10 scale)
            resize_enabled: false,
            max_width: 1200,
            max_height: 1200,
        }
    }
}

#[wasm_bindgen]
pub fn optimize_image(
    data: &[u8],
    filename: &str,
    options_js: JsValue,
) -> Result<JsValue, JsValue> {
    let options: OptimizeOptions = serde_wasm_bindgen::from_value(options_js)
        .unwrap_or_else(|_| OptimizeOptions::default());

    let result = optimize_image_internal(data, filename, &options);
    
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

struct OptimizeOutput {
    data: Vec<u8>,
    original_width: u32,
    original_height: u32,
    new_width: u32,
    new_height: u32,
}

fn optimize_image_internal(data: &[u8], filename: &str, options: &OptimizeOptions) -> OptimizeResult {
    let original_size = data.len();
    let lower_filename = filename.to_lowercase();
    
    let is_png = lower_filename.ends_with(".png");
    let is_jpeg = lower_filename.ends_with(".jpg") || lower_filename.ends_with(".jpeg");
    
    if is_png {
        match optimize_png(data, options) {
            Ok(output) => OptimizeResult {
                optimized_size: output.data.len(),
                data: output.data,
                original_size,
                format: "png".to_string(),
                filename: filename.to_string(),
                success: true,
                error: None,
                original_width: output.original_width,
                original_height: output.original_height,
                new_width: output.new_width,
                new_height: output.new_height,
            },
            Err(e) => OptimizeResult {
                data: vec![],
                original_size,
                optimized_size: 0,
                format: "png".to_string(),
                filename: filename.to_string(),
                success: false,
                error: Some(e),
                original_width: 0,
                original_height: 0,
                new_width: 0,
                new_height: 0,
            },
        }
    } else if is_jpeg {
        match optimize_jpeg(data, options) {
            Ok(output) => OptimizeResult {
                optimized_size: output.data.len(),
                data: output.data,
                original_size,
                format: "jpeg".to_string(),
                filename: filename.to_string(),
                success: true,
                error: None,
                original_width: output.original_width,
                original_height: output.original_height,
                new_width: output.new_width,
                new_height: output.new_height,
            },
            Err(e) => OptimizeResult {
                data: vec![],
                original_size,
                optimized_size: 0,
                format: "jpeg".to_string(),
                filename: filename.to_string(),
                success: false,
                error: Some(e),
                original_width: 0,
                original_height: 0,
                new_width: 0,
                new_height: 0,
            },
        }
    } else {
        OptimizeResult {
            data: vec![],
            original_size,
            optimized_size: 0,
            format: "unknown".to_string(),
            filename: filename.to_string(),
            success: false,
            error: Some("Unsupported format. Please use JPEG or PNG files.".to_string()),
            original_width: 0,
            original_height: 0,
            new_width: 0,
            new_height: 0,
        }
    }
}

struct ResizeResult {
    img: image::DynamicImage,
    original_width: u32,
    original_height: u32,
}

fn resize_image_if_needed(img: image::DynamicImage, options: &OptimizeOptions) -> ResizeResult {
    let (original_width, original_height) = img.dimensions();
    
    if !options.resize_enabled {
        return ResizeResult { img, original_width, original_height };
    }
    
    let max_w = options.max_width;
    let max_h = options.max_height;
    
    if original_width <= max_w && original_height <= max_h {
        return ResizeResult { img, original_width, original_height };
    }
    
    // Calculate new dimensions maintaining aspect ratio
    let ratio_w = max_w as f64 / original_width as f64;
    let ratio_h = max_h as f64 / original_height as f64;
    let ratio = ratio_w.min(ratio_h);
    
    let new_width = (original_width as f64 * ratio).round() as u32;
    let new_height = (original_height as f64 * ratio).round() as u32;
    
    let resized = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);
    ResizeResult { img: resized, original_width, original_height }
}

fn optimize_png(data: &[u8], options: &OptimizeOptions) -> Result<OptimizeOutput, String> {
    let img = image::load_from_memory_with_format(data, ImageFormat::Png)
        .map_err(|e| format!("Failed to decode PNG: {}", e))?;
    
    let resize_result = resize_image_if_needed(img, options);
    let original_width = resize_result.original_width;
    let original_height = resize_result.original_height;
    let img = resize_result.img;
    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8();
    
    // Collect all outputs to compare
    let mut outputs: Vec<Vec<u8>> = Vec::new();
    
    // Count unique colors for auto mode decisions
    let unique_colors = if options.png_auto {
        count_unique_colors(&rgba)
    } else {
        0
    };
    
    // Strategy 1: Try indexed PNG (palette mode) - lossless if ≤256 colors
    let target_colors = (options.png_colors as usize).clamp(2, 256);
    if let Ok(indexed) = try_indexed_png(&rgba, width, height, 256, options) {
        outputs.push(indexed);
    }
    
    // Strategy 2: Quantization
    if options.png_auto {
        // Auto mode: intelligently try different color counts based on image
        if unique_colors > 256 {
            // Try multiple quantization levels and pick the best
            let color_levels = auto_select_color_levels(unique_colors, options.png_auto_level);
            for colors in color_levels {
                if let Ok(quantized) = try_quantized_png(&rgba, width, height, colors, options) {
                    outputs.push(quantized);
                }
            }
        }
    } else if options.png_quantize {
        // Manual mode: use user-specified color count
        if let Ok(quantized) = try_quantized_png(&rgba, width, height, target_colors, options) {
            outputs.push(quantized);
        }
    }
    
    // Strategy 3: Optimal direct encoding (grayscale/RGB/RGBA)
    let (color_type, image_data) = determine_optimal_color_type(&rgba);
    if let Ok(direct) = encode_png(width, height, color_type, BitDepth::Eight, &image_data, options) {
        outputs.push(direct);
    }
    
    // Find the smallest output
    let best = outputs.into_iter().min_by_key(|o| o.len());
    
    let result_data = match best {
        Some(output) if output.len() < data.len() => output,
        _ => data.to_vec(), // Return original if we can't improve
    };
    
    Ok(OptimizeOutput {
        data: result_data,
        original_width,
        original_height,
        new_width: width,
        new_height: height,
    })
}

/// Count unique colors in the image (capped for performance)
fn count_unique_colors(rgba: &image::RgbaImage) -> usize {
    let mut colors: std::collections::HashSet<[u8; 4]> = std::collections::HashSet::new();
    for pixel in rgba.pixels() {
        colors.insert(pixel.0);
        // Cap at 1000 to avoid excessive memory/time for large images
        if colors.len() > 1000 {
            return colors.len();
        }
    }
    colors.len()
}

/// Auto-select color levels to try based on auto level
/// Higher effort = more aggressive color reduction = smaller files but potentially lower quality
fn auto_select_color_levels(unique_colors: usize, auto_level: u8) -> Vec<usize> {
    if unique_colors <= 256 {
        // Already under 256 colors, no quantization needed
        return vec![];
    }
    
    // Each effort level tries increasingly aggressive color reduction
    // The minimum color count decreases as effort increases
    match auto_level {
        1 => vec![256],                              // Lossless-ish only
        2 => vec![256, 224],                         // Very light
        3 => vec![256, 192, 160],                    // Light
        4 => vec![256, 192, 128],                    // Light-medium
        5 => vec![256, 192, 128, 96],                // Medium (default)
        6 => vec![256, 192, 128, 64],                // Medium-aggressive
        7 => vec![256, 192, 128, 64, 48],            // Aggressive
        8 => vec![256, 192, 128, 64, 32],            // More aggressive
        9 => vec![256, 192, 128, 64, 32, 24],        // Very aggressive
        10 => vec![256, 192, 128, 64, 32, 24, 16],   // Maximum compression
        _ => vec![256, 192, 128, 96],                // Default fallback
    }
}

/// Try to create an indexed PNG if the image has ≤256 unique colors (lossless)
fn try_indexed_png(
    rgba: &image::RgbaImage,
    width: u32,
    height: u32,
    max_colors: usize,
    options: &OptimizeOptions,
) -> Result<Vec<u8>, String> {
    // Count unique colors
    let mut color_to_index: HashMap<[u8; 4], u8> = HashMap::new();
    let mut palette: Vec<[u8; 4]> = Vec::new();
    
    for pixel in rgba.pixels() {
        let color = pixel.0;
        if !color_to_index.contains_key(&color) {
            if palette.len() >= max_colors {
                return Err("Too many colors for indexed PNG".to_string());
            }
            color_to_index.insert(color, palette.len() as u8);
            palette.push(color);
        }
    }
    
    // Create indexed image data
    let indices: Vec<u8> = rgba.pixels()
        .map(|p| color_to_index[&p.0])
        .collect();
    
    encode_indexed_png(width, height, &palette, &indices, options)
}

/// Use NeuQuant color quantization to reduce colors (lossy)
fn try_quantized_png(
    rgba: &image::RgbaImage,
    width: u32,
    height: u32,
    target_colors: usize,
    options: &OptimizeOptions,
) -> Result<Vec<u8>, String> {
    // Prepare pixel data for NeuQuant (expects RGBA bytes)
    let pixels: Vec<u8> = rgba.pixels().flat_map(|p| p.0).collect();
    
    // NeuQuant sample_faction: 1 = examine all pixels (best quality), 30 = sample 1/30 (faster)
    let sample_faction = match options.png_compression {
        0..=2 => 10,  // Faster
        3..=6 => 3,   // Balanced  
        _ => 1,       // Best quality
    };
    
    // Create quantizer
    let nq = NeuQuant::new(sample_faction, target_colors, &pixels);
    
    // Build palette from NeuQuant
    let color_map = nq.color_map_rgba();
    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(target_colors);
    for chunk in color_map.chunks(4) {
        if chunk.len() == 4 {
            palette.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
    
    // Map each pixel to its nearest palette index
    let indices: Vec<u8> = pixels
        .chunks(4)
        .map(|chunk| nq.index_of(chunk) as u8)
        .collect();
    
    encode_indexed_png(width, height, &palette, &indices, options)
}

/// Encode an indexed PNG with optimal bit depth
fn encode_indexed_png(
    width: u32,
    height: u32,
    palette: &[[u8; 4]],
    indices: &[u8],
    options: &OptimizeOptions,
) -> Result<Vec<u8>, String> {
    let num_colors = palette.len();
    
    // Determine optimal bit depth based on palette size
    let (bit_depth, packed_data) = if num_colors <= 2 {
        (BitDepth::One, pack_indices(indices, 1, width as usize))
    } else if num_colors <= 4 {
        (BitDepth::Two, pack_indices(indices, 2, width as usize))
    } else if num_colors <= 16 {
        (BitDepth::Four, pack_indices(indices, 4, width as usize))
    } else {
        (BitDepth::Eight, indices.to_vec())
    };
    
    // Build RGB palette
    let rgb_palette: Vec<u8> = palette.iter()
        .flat_map(|c| [c[0], c[1], c[2]])
        .collect();
    
    // Check for transparency
    let has_alpha = palette.iter().any(|c| c[3] < 255);
    let trns: Option<Vec<u8>> = if has_alpha {
        Some(palette.iter().map(|c| c[3]).collect())
    } else {
        None
    };
    
    // Encode
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut output, width, height);
        encoder.set_color(PngColorType::Indexed);
        encoder.set_depth(bit_depth);
        encoder.set_palette(rgb_palette);
        
        if let Some(t) = trns {
            encoder.set_trns(t);
        }
        
        let compression = match options.png_compression {
            0..=1 => Compression::Fastest,
            2..=3 => Compression::Fast,
            4..=6 => Compression::Balanced,
            _ => Compression::High,
        };
        encoder.set_compression(compression);
        encoder.set_filter(Filter::Adaptive);
        
        let mut writer = encoder.write_header()
            .map_err(|e| format!("Failed to write indexed PNG header: {}", e))?;
        
        writer.write_image_data(&packed_data)
            .map_err(|e| format!("Failed to write indexed PNG data: {}", e))?;
    }
    
    Ok(output)
}

/// Pack indices into fewer bits per pixel
fn pack_indices(indices: &[u8], bits: usize, width: usize) -> Vec<u8> {
    if bits == 8 {
        return indices.to_vec();
    }
    
    let pixels_per_byte = 8 / bits;
    let height = indices.len() / width;
    let row_bytes = (width + pixels_per_byte - 1) / pixels_per_byte;
    
    let mut packed = Vec::with_capacity(row_bytes * height);
    
    for row in 0..height {
        let row_start = row * width;
        for byte_idx in 0..row_bytes {
            let mut byte = 0u8;
            for i in 0..pixels_per_byte {
                let pixel_idx = byte_idx * pixels_per_byte + i;
                if pixel_idx < width {
                    let index = indices[row_start + pixel_idx];
                    let shift = (pixels_per_byte - 1 - i) * bits;
                    byte |= (index & ((1 << bits) - 1)) << shift;
                }
            }
            packed.push(byte);
        }
    }
    
    packed
}

/// Encode a standard PNG with given color type
fn encode_png(
    width: u32,
    height: u32,
    color_type: PngColorType,
    bit_depth: BitDepth,
    data: &[u8],
    options: &OptimizeOptions,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut output, width, height);
        encoder.set_color(color_type);
        encoder.set_depth(bit_depth);
        
        let compression = match options.png_compression {
            0..=1 => Compression::Fastest,
            2..=3 => Compression::Fast,
            4..=6 => Compression::Balanced,
            _ => Compression::High,
        };
        encoder.set_compression(compression);
        encoder.set_filter(Filter::Adaptive);
        
        let mut writer = encoder.write_header()
            .map_err(|e| format!("Failed to write PNG header: {}", e))?;
        
        writer.write_image_data(data)
            .map_err(|e| format!("Failed to write PNG data: {}", e))?;
    }
    
    Ok(output)
}

/// Determine the optimal color type (grayscale, RGB, or RGBA)
fn determine_optimal_color_type(rgba: &image::RgbaImage) -> (PngColorType, Vec<u8>) {
    let has_alpha = rgba.pixels().any(|p| p[3] < 255);
    let is_grayscale = rgba.pixels().all(|p| p[0] == p[1] && p[1] == p[2]);
    
    if is_grayscale && !has_alpha {
        let gray: Vec<u8> = rgba.pixels().map(|p| p[0]).collect();
        (PngColorType::Grayscale, gray)
    } else if is_grayscale && has_alpha {
        let gray_alpha: Vec<u8> = rgba.pixels().flat_map(|p| [p[0], p[3]]).collect();
        (PngColorType::GrayscaleAlpha, gray_alpha)
    } else if !has_alpha {
        let rgb: Vec<u8> = rgba.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
        (PngColorType::Rgb, rgb)
    } else {
        (PngColorType::Rgba, rgba.clone().into_raw())
    }
}

fn optimize_jpeg(data: &[u8], options: &OptimizeOptions) -> Result<OptimizeOutput, String> {
    let original_size = data.len();
    
    let img = image::load_from_memory_with_format(data, ImageFormat::Jpeg)
        .map_err(|e| format!("Failed to decode JPEG: {}", e))?;
    
    let resize_result = resize_image_if_needed(img, options);
    let original_width = resize_result.original_width;
    let original_height = resize_result.original_height;
    let img = resize_result.img;
    let rgb_image = img.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    
    // Try encoding at various quality levels
    let mut best_encoded: Option<Vec<u8>> = None;
    
    let qualities: Vec<u8> = if options.png_auto {
        // Auto mode: try multiple quality levels aggressively
        vec![
            options.quality,
            options.quality.saturating_sub(5),
            options.quality.saturating_sub(10),
            options.quality.saturating_sub(15),
            80, 75, 70, 65, 60, 55, 50,
        ]
    } else {
        // Manual mode: just use specified quality
        vec![options.quality]
    };
    
    for q in qualities {
        if q >= 30 {
            if let Ok(encoded) = encode_jpeg(&rgb_image, width, height, q) {
                let dominated = best_encoded.as_ref().map_or(false, |b| encoded.len() >= b.len());
                if !dominated {
                    best_encoded = Some(encoded);
                }
            }
        }
    }
    
    // CRITICAL: Only return encoded if it's STRICTLY smaller than original
    let result_data = match best_encoded {
        Some(encoded) if encoded.len() < original_size => encoded,
        _ => data.to_vec(),  // Return original unchanged
    };
    
    Ok(OptimizeOutput {
        data: result_data,
        original_width,
        original_height,
        new_width: width,
        new_height: height,
    })
}

fn encode_jpeg(rgb_image: &image::RgbImage, width: u32, height: u32, quality: u8) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let encoder = Encoder::new(&mut output, quality);
    
    encoder
        .encode(
            rgb_image.as_raw(),
            width as u16,
            height as u16,
            ColorType::Rgb,
        )
        .map_err(|e| format!("Failed to encode JPEG: {}", e))?;
    
    Ok(output)
}

#[wasm_bindgen]
pub fn get_default_options() -> JsValue {
    let options = OptimizeOptions::default();
    serde_wasm_bindgen::to_value(&options).unwrap()
}

#[wasm_bindgen]
pub fn get_supported_formats() -> JsValue {
    let formats = vec!["jpeg", "jpg", "png"];
    serde_wasm_bindgen::to_value(&formats).unwrap()
}
