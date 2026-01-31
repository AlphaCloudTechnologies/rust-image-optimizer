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
    pub png_dithering: bool,     // Enable Floyd-Steinberg dithering for quantized PNGs
    pub png_dithering_level: f32, // Dithering strength 0.0-1.0
    pub png_auto: bool,          // Auto mode: automatically find best settings
    pub png_auto_level: u8,      // Auto mode level: 1-10 scale
    pub jpeg_auto: bool,         // Auto mode for JPEG: find optimal quality/size tradeoff
    pub jpeg_auto_level: u8,     // Auto mode level for JPEG: 1=Light to 10=Maximum compression
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
    pub original_palette: Vec<String>,
    pub optimized_palette: Vec<String>,
    pub original_colors: usize,
    pub optimized_colors: usize,
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self {
            quality: 85,
            png_compression: 6,
            strip_metadata: true,
            png_quantize: false,
            png_colors: 256,
            png_dithering: true,  // Dithering enabled by default for better quality
            png_dithering_level: 0.8,  // 80% dithering strength
            png_auto: true,  // Auto mode enabled by default
            png_auto_level: 5,  // Default to balanced (1-10 scale)
            jpeg_auto: true,  // Auto mode enabled by default for JPEG
            jpeg_auto_level: 5,  // Default to balanced (1-10 scale)
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

/// Palette extraction result with colors and count
struct PaletteInfo {
    palette: Vec<String>,
    color_count: usize,
}

/// Extract dominant colors from an image and return them as hex strings along with total unique color count
fn extract_palette(rgba: &image::RgbaImage, max_colors: usize) -> PaletteInfo {
    // Sample pixels and count color occurrences
    let mut color_counts: HashMap<[u8; 3], u32> = HashMap::new();
    
    // Sample every Nth pixel for performance on large images
    let total_pixels = rgba.width() as usize * rgba.height() as usize;
    let sample_rate = if total_pixels > 100_000 { 10 } else if total_pixels > 10_000 { 5 } else { 1 };
    
    for (i, pixel) in rgba.pixels().enumerate() {
        if i % sample_rate != 0 {
            continue;
        }
        // Skip fully transparent pixels
        if pixel[3] < 128 {
            continue;
        }
        // Quantize colors to reduce noise (group similar colors)
        let r = (pixel[0] / 16) * 16;
        let g = (pixel[1] / 16) * 16;
        let b = (pixel[2] / 16) * 16;
        *color_counts.entry([r, g, b]).or_insert(0) += 1;
    }
    
    // Total unique colors (quantized)
    let color_count = color_counts.len();
    
    // Sort by frequency and take top colors
    let mut colors: Vec<_> = color_counts.into_iter().collect();
    colors.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Convert to hex strings
    let palette = colors.iter()
        .take(max_colors)
        .map(|([r, g, b], _)| format!("#{:02x}{:02x}{:02x}", r, g, b))
        .collect();
    
    PaletteInfo { palette, color_count }
}

/// Extract palette from raw image bytes
fn extract_palette_from_bytes(data: &[u8], format: ImageFormat) -> PaletteInfo {
    if let Ok(img) = image::load_from_memory_with_format(data, format) {
        extract_palette(&img.to_rgba8(), 8)
    } else {
        PaletteInfo { palette: vec![], color_count: 0 }
    }
}

fn optimize_image_internal(data: &[u8], filename: &str, options: &OptimizeOptions) -> OptimizeResult {
    let original_size = data.len();
    let lower_filename = filename.to_lowercase();
    
    let is_png = lower_filename.ends_with(".png");
    let is_jpeg = lower_filename.ends_with(".jpg") || lower_filename.ends_with(".jpeg");
    
    if is_png {
        // Extract original palette before optimization
        let original_info = extract_palette_from_bytes(data, ImageFormat::Png);
        
        match optimize_png(data, options) {
            Ok(output) => {
                // Extract optimized palette
                let optimized_info = extract_palette_from_bytes(&output.data, ImageFormat::Png);
                OptimizeResult {
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
                    original_palette: original_info.palette,
                    optimized_palette: optimized_info.palette,
                    original_colors: original_info.color_count,
                    optimized_colors: optimized_info.color_count,
                }
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
                original_palette: vec![],
                optimized_palette: vec![],
                original_colors: 0,
                optimized_colors: 0,
            },
        }
    } else if is_jpeg {
        // Extract original palette before optimization
        let original_info = extract_palette_from_bytes(data, ImageFormat::Jpeg);
        
        match optimize_jpeg(data, options) {
            Ok(output) => {
                // Extract optimized palette
                let optimized_info = extract_palette_from_bytes(&output.data, ImageFormat::Jpeg);
                OptimizeResult {
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
                    original_palette: original_info.palette,
                    optimized_palette: optimized_info.palette,
                    original_colors: original_info.color_count,
                    optimized_colors: optimized_info.color_count,
                }
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
                original_palette: vec![],
                optimized_palette: vec![],
                original_colors: 0,
                optimized_colors: 0,
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
            original_palette: vec![],
            optimized_palette: vec![],
            original_colors: 0,
            optimized_colors: 0,
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
    
    let target_colors = (options.png_colors as usize).clamp(2, 256);
    
    // Manual quantization mode: ONLY use the user's specified color count
    if options.png_quantize && !options.png_auto {
        if let Ok(quantized) = try_quantized_png(&rgba, width, height, target_colors, options) {
            outputs.push(quantized);
        }
        
        // Find the smallest output (should be just the quantized one)
        let best = outputs.into_iter().min_by_key(|o| o.len());
        
        let result_data = match best {
            Some(output) => output, // Always use quantized result in manual mode
            _ => data.to_vec(),
        };
        
        return Ok(OptimizeOutput {
            data: result_data,
            original_width,
            original_height,
            new_width: width,
            new_height: height,
        });
    }
    
    // Count unique colors for auto mode decisions
    let color_info = if options.png_auto {
        count_unique_colors(&rgba)
    } else {
        ColorCountResult { count: 0, is_capped: false, has_alpha: false, is_grayscale: false }
    };
    let unique_colors = color_info.count;
    
    // Auto-select compression effort based on color count when in auto mode
    let effective_options = if options.png_auto {
        let auto_compression = auto_select_compression_effort(unique_colors, options.png_compression);
        let mut opts = options.clone();
        opts.png_compression = auto_compression;
        opts
    } else {
        options.clone()
    };
    
    // Strategy 1: Try indexed PNG (palette mode) - lossless if ≤256 colors
    if let Ok(indexed) = try_indexed_png(&rgba, width, height, 256, &effective_options) {
        outputs.push(indexed);
    }
    
    // Strategy 2: Auto mode quantization
    if options.png_auto && unique_colors > 256 {
        // Try multiple quantization levels and pick the best
        let color_levels = auto_select_color_levels(unique_colors, options.png_auto_level);
        for colors in color_levels {
            if let Ok(quantized) = try_quantized_png(&rgba, width, height, colors, &effective_options) {
                outputs.push(quantized);
            }
        }
    }
    
    // Strategy 3: Optimal direct encoding (grayscale/RGB/RGBA)
    let (color_type, image_data) = determine_optimal_color_type(&rgba);
    if let Ok(direct) = encode_png(width, height, color_type, BitDepth::Eight, &image_data, &effective_options) {
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

/// Result of color counting with information about whether it was capped
#[allow(dead_code)]  // Fields reserved for future optimizations
struct ColorCountResult {
    count: usize,
    is_capped: bool,
    has_alpha: bool,
    is_grayscale: bool,
}

/// Count unique colors in the image with smart sampling for large images
fn count_unique_colors(rgba: &image::RgbaImage) -> ColorCountResult {
    use std::collections::HashSet;
    
    let total_pixels = rgba.pixels().len();
    let sample_rate = if total_pixels > 500_000 { 8 } 
                      else if total_pixels > 100_000 { 4 } 
                      else if total_pixels > 50_000 { 2 } 
                      else { 1 };
    
    let mut colors: HashSet<[u8; 4]> = HashSet::with_capacity(512);
    let mut has_alpha = false;
    let mut is_grayscale = true;
    
    for (i, pixel) in rgba.pixels().enumerate() {
        if i % sample_rate != 0 {
            continue;
        }
        
        // Track alpha and grayscale properties
        if pixel[3] < 255 {
            has_alpha = true;
        }
        if pixel[0] != pixel[1] || pixel[1] != pixel[2] {
            is_grayscale = false;
        }
        
        colors.insert(pixel.0);
        
        // Cap at 4096 to avoid excessive memory for complex images
        if colors.len() > 4096 {
            return ColorCountResult {
                count: colors.len(),
                is_capped: true,
                has_alpha,
                is_grayscale,
            };
        }
    }
    
    // If we sampled, estimate actual count
    let estimated_count = if sample_rate > 1 {
        // Colors scale sub-linearly with sampling, apply correction factor
        (colors.len() as f64 * (sample_rate as f64).sqrt()).ceil() as usize
    } else {
        colors.len()
    };
    
    ColorCountResult {
        count: estimated_count,
        is_capped: false,
        has_alpha,
        is_grayscale,
    }
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

/// Auto-select compression effort based on the number of unique colors in the image.
/// Images with fewer colors are simpler and compress well with less effort.
/// Images with many colors benefit from higher compression effort.
fn auto_select_compression_effort(unique_colors: usize, base_level: u8) -> u8 {
    // Calculate a color-based modifier
    // - Very few colors (≤16): reduce effort by 2
    // - Few colors (≤64): reduce effort by 1  
    // - Medium colors (≤256): keep base effort
    // - Many colors (≤1000): increase effort by 1
    // - Very many colors (>1000): increase effort by 2
    let color_modifier: i8 = if unique_colors <= 16 {
        -2  // Very simple images - low effort is fine
    } else if unique_colors <= 64 {
        -1  // Simple images - slightly lower effort
    } else if unique_colors <= 256 {
        0   // Medium complexity - base effort
    } else if unique_colors <= 1000 {
        1   // Complex images - higher effort helps
    } else {
        2   // Very complex images - maximum effort beneficial
    };
    
    // Apply modifier while keeping in valid range (0-9 for compression)
    let adjusted = (base_level as i8 + color_modifier).clamp(0, 9);
    adjusted as u8
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

/// Find the nearest color in the palette using weighted distance (perceptual)
fn find_nearest_color_index(pixel: &[f32; 4], palette: &[[u8; 4]]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    
    for (idx, pal_color) in palette.iter().enumerate() {
        // Weighted RGB distance (green is more perceptually important)
        let dr = pixel[0] - pal_color[0] as f32;
        let dg = pixel[1] - pal_color[1] as f32;
        let db = pixel[2] - pal_color[2] as f32;
        let da = pixel[3] - pal_color[3] as f32;
        
        // Perceptual weights: R=0.299, G=0.587, B=0.114 (ITU-R BT.601)
        let dist = dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114 + da * da * 0.1;
        
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    best_idx
}

/// Apply Floyd-Steinberg dithering to reduce color banding
/// Returns the dithered image with pixels mapped to palette indices
fn apply_floyd_steinberg_dithering(
    rgba: &image::RgbaImage,
    palette: &[[u8; 4]],
    strength: f32,
) -> Vec<u8> {
    let (width, height) = rgba.dimensions();
    let w = width as usize;
    let h = height as usize;
    
    // Working buffer with accumulated errors (using f32 for precision)
    let mut pixels: Vec<[f32; 4]> = rgba.pixels()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32])
        .collect();
    
    let mut indices = Vec::with_capacity(w * h);
    
    // Strength factor (0.0 = no dithering, 1.0 = full dithering)
    let s = strength.clamp(0.0, 1.0);
    
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let current = pixels[idx];
            
            // Clamp to valid range
            let clamped = [
                current[0].clamp(0.0, 255.0),
                current[1].clamp(0.0, 255.0),
                current[2].clamp(0.0, 255.0),
                current[3].clamp(0.0, 255.0),
            ];
            
            // Find nearest palette color
            let nearest_idx = find_nearest_color_index(&clamped, palette);
            indices.push(nearest_idx as u8);
            
            let nearest = palette[nearest_idx];
            
            // Calculate quantization error
            let error = [
                (clamped[0] - nearest[0] as f32) * s,
                (clamped[1] - nearest[1] as f32) * s,
                (clamped[2] - nearest[2] as f32) * s,
                (clamped[3] - nearest[3] as f32) * s,
            ];
            
            // Distribute error using Floyd-Steinberg coefficients
            // Current pixel: X
            //                7/16  (right)
            //   3/16  5/16  1/16   (next row: left, center, right)
            
            if x + 1 < w {
                let i = idx + 1;
                pixels[i][0] += error[0] * 7.0 / 16.0;
                pixels[i][1] += error[1] * 7.0 / 16.0;
                pixels[i][2] += error[2] * 7.0 / 16.0;
                pixels[i][3] += error[3] * 7.0 / 16.0;
            }
            
            if y + 1 < h {
                if x > 0 {
                    let i = (y + 1) * w + (x - 1);
                    pixels[i][0] += error[0] * 3.0 / 16.0;
                    pixels[i][1] += error[1] * 3.0 / 16.0;
                    pixels[i][2] += error[2] * 3.0 / 16.0;
                    pixels[i][3] += error[3] * 3.0 / 16.0;
                }
                
                let i = (y + 1) * w + x;
                pixels[i][0] += error[0] * 5.0 / 16.0;
                pixels[i][1] += error[1] * 5.0 / 16.0;
                pixels[i][2] += error[2] * 5.0 / 16.0;
                pixels[i][3] += error[3] * 5.0 / 16.0;
                
                if x + 1 < w {
                    let i = (y + 1) * w + (x + 1);
                    pixels[i][0] += error[0] * 1.0 / 16.0;
                    pixels[i][1] += error[1] * 1.0 / 16.0;
                    pixels[i][2] += error[2] * 1.0 / 16.0;
                    pixels[i][3] += error[3] * 1.0 / 16.0;
                }
            }
        }
    }
    
    indices
}

/// Use NeuQuant color quantization to reduce colors (lossy)
/// With optional Floyd-Steinberg dithering for better visual quality
fn try_quantized_png(
    rgba: &image::RgbaImage,
    width: u32,
    height: u32,
    target_colors: usize,
    options: &OptimizeOptions,
) -> Result<Vec<u8>, String> {
    // Filter out fully transparent pixels for better color selection
    // Transparent pixels shouldn't influence the palette
    let opaque_pixels: Vec<u8> = rgba.pixels()
        .filter(|p| p[3] > 0)  // Only include pixels with some opacity
        .flat_map(|p| p.0)
        .collect();
    
    // If all pixels are transparent, use original data
    let quantize_pixels = if opaque_pixels.is_empty() {
        rgba.pixels().flat_map(|p| p.0).collect()
    } else {
        opaque_pixels
    };
    
    // NeuQuant sample_faction: 1 = examine all pixels (best quality), 30 = sample 1/30 (faster)
    // Note: Lower values = better quality (more pixels examined)
    let sample_faction = match options.png_compression {
        0..=2 => 10,  // Faster, lower quality
        3..=5 => 3,   // Balanced  
        6..=7 => 2,   // Higher quality
        _ => 1,       // Best quality (examine all pixels)
    };
    
    // Create quantizer with filtered pixels
    let nq = NeuQuant::new(sample_faction, target_colors, &quantize_pixels);
    
    // Build palette from NeuQuant
    let color_map = nq.color_map_rgba();
    let mut palette: Vec<[u8; 4]> = Vec::with_capacity(target_colors);
    for chunk in color_map.chunks(4) {
        if chunk.len() == 4 {
            palette.push([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
    
    // Ensure we have a fully transparent color in palette if image has transparency
    let has_transparency = rgba.pixels().any(|p| p[3] == 0);
    if has_transparency {
        // Find or add a transparent color
        let transparent_exists = palette.iter().any(|c| c[3] == 0);
        if !transparent_exists && !palette.is_empty() {
            // Replace the least used color with transparent
            let last_idx = palette.len() - 1;
            palette[last_idx] = [0, 0, 0, 0];
        }
    }
    
    // Map pixels to palette indices - with or without dithering
    let indices = if options.png_dithering && options.png_dithering_level > 0.0 {
        apply_floyd_steinberg_dithering(rgba, &palette, options.png_dithering_level)
    } else {
        // Simple nearest-color mapping without dithering
        rgba.pixels()
            .map(|p| {
                let pixel = [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32];
                find_nearest_color_index(&pixel, &palette) as u8
            })
            .collect()
    };
    
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
    
    // Optimize tRNS chunk - only include alpha values up to the last non-opaque entry
    // PNG spec allows tRNS to be shorter than palette; remaining entries are assumed opaque
    let trns: Option<Vec<u8>> = {
        let alphas: Vec<u8> = palette.iter().map(|c| c[3]).collect();
        // Find the last non-opaque (< 255) entry
        if let Some(last_non_opaque) = alphas.iter().rposition(|&a| a < 255) {
            // Only include alpha values up to and including the last non-opaque
            Some(alphas[..=last_non_opaque].to_vec())
        } else {
            // All entries are fully opaque, no tRNS needed
            None
        }
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

/// Encode a PNG with a specific filter
fn encode_png_with_filter(
    width: u32,
    height: u32,
    color_type: PngColorType,
    bit_depth: BitDepth,
    data: &[u8],
    compression: Compression,
    filter: Filter,
) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut output, width, height);
        encoder.set_color(color_type);
        encoder.set_depth(bit_depth);
        encoder.set_compression(compression);
        encoder.set_filter(filter);
        
        let mut writer = encoder.write_header()
            .map_err(|e| format!("Failed to write PNG header: {}", e))?;
        
        writer.write_image_data(data)
            .map_err(|e| format!("Failed to write PNG data: {}", e))?;
    }
    
    Ok(output)
}

/// Encode a standard PNG with given color type, trying multiple filters to find the best
fn encode_png(
    width: u32,
    height: u32,
    color_type: PngColorType,
    bit_depth: BitDepth,
    data: &[u8],
    options: &OptimizeOptions,
) -> Result<Vec<u8>, String> {
    let compression = match options.png_compression {
        0..=1 => Compression::Fastest,
        2..=3 => Compression::Fast,
        4..=6 => Compression::Balanced,
        _ => Compression::High,
    };
    
    // For high compression levels, try multiple filters and pick the smallest
    if options.png_compression >= 6 {
        let filters = [
            Filter::NoFilter,
            Filter::Sub,
            Filter::Up,
            Filter::Avg,
            Filter::Paeth,
            Filter::Adaptive,
        ];
        
        let mut best_output: Option<Vec<u8>> = None;
        
        for &filter in &filters {
            if let Ok(output) = encode_png_with_filter(width, height, color_type, bit_depth, data, compression, filter) {
                let is_better = best_output.as_ref().map_or(true, |best| output.len() < best.len());
                if is_better {
                    best_output = Some(output);
                }
            }
        }
        
        best_output.ok_or_else(|| "All PNG filter strategies failed".to_string())
    } else {
        // For lower compression levels, just use Adaptive filter for speed
        encode_png_with_filter(width, height, color_type, bit_depth, data, compression, Filter::Adaptive)
    }
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

/// Use binary search to find the optimal JPEG quality for a target size
fn find_optimal_jpeg_quality(
    rgb_image: &image::RgbImage,
    width: u32,
    height: u32,
    target_size: usize,
    min_quality: u8,
    max_quality: u8,
) -> Option<(Vec<u8>, u8)> {
    let mut low = min_quality;
    let mut high = max_quality;
    let mut best_result: Option<(Vec<u8>, u8)> = None;
    
    // Binary search for the highest quality that fits target size
    while high - low > 2 {
        let mid = (low + high) / 2;
        
        match encode_jpeg(rgb_image, width, height, mid) {
            Ok(encoded) => {
                if encoded.len() <= target_size {
                    // This quality fits, try higher
                    best_result = Some((encoded, mid));
                    low = mid;
                } else {
                    // Too large, try lower quality
                    high = mid;
                }
            }
            Err(_) => {
                high = mid; // Encoding failed, try lower
            }
        }
    }
    
    // Final check at the upper bound
    if let Ok(encoded) = encode_jpeg(rgb_image, width, height, high) {
        if encoded.len() <= target_size {
            if best_result.as_ref().map_or(true, |(_, q)| high > *q) {
                best_result = Some((encoded, high));
            }
        }
    }
    
    best_result
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
    
    let mut best_encoded: Option<Vec<u8>> = None;
    
    if options.jpeg_auto {
        // Auto mode: Use binary search to find optimal quality
        // Target different compression ratios based on jpeg_auto_level
        let target_ratios: Vec<f32> = match options.jpeg_auto_level {
            1..=2 => vec![0.95, 0.90],           // Light: 5-10% reduction
            3..=4 => vec![0.85, 0.75, 0.65],     // Moderate: 15-35% reduction
            5..=6 => vec![0.75, 0.60, 0.50],     // Balanced: 25-50% reduction
            7..=8 => vec![0.60, 0.45, 0.35],     // Aggressive: 40-65% reduction
            _ => vec![0.50, 0.35, 0.25],         // Maximum: 50-75% reduction
        };
        
        // First, try the user-specified quality
        if let Ok(encoded) = encode_jpeg(&rgb_image, width, height, options.quality) {
            if encoded.len() < original_size {
                best_encoded = Some(encoded);
            }
        }
        
        // Then try binary search for each target ratio
        for ratio in target_ratios {
            let target_size = (original_size as f32 * ratio) as usize;
            
            // Only try if target is smaller than what we have
            if best_encoded.as_ref().map_or(true, |b| target_size < b.len()) {
                if let Some((encoded, _quality)) = find_optimal_jpeg_quality(
                    &rgb_image, width, height, target_size, 35, options.quality
                ) {
                    // Only keep if it's actually smaller
                    if best_encoded.as_ref().map_or(true, |b| encoded.len() < b.len()) {
                        best_encoded = Some(encoded);
                    }
                }
            }
        }
    } else {
        // Manual mode: just use specified quality
        if let Ok(encoded) = encode_jpeg(&rgb_image, width, height, options.quality) {
            best_encoded = Some(encoded);
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

/// Convert PNG to JPEG
#[wasm_bindgen]
pub fn convert_png_to_jpeg(
    data: &[u8],
    filename: &str,
    options_js: JsValue,
) -> Result<JsValue, JsValue> {
    let options: OptimizeOptions = serde_wasm_bindgen::from_value(options_js)
        .unwrap_or_else(|_| OptimizeOptions::default());

    let result = convert_png_to_jpeg_internal(data, filename, &options);
    
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

fn convert_png_to_jpeg_internal(data: &[u8], filename: &str, options: &OptimizeOptions) -> OptimizeResult {
    let original_size = data.len();
    
    // Load PNG
    let img = match image::load_from_memory_with_format(data, ImageFormat::Png) {
        Ok(img) => img,
        Err(e) => {
            return OptimizeResult {
                data: vec![],
                original_size,
                optimized_size: 0,
                format: "jpeg".to_string(),
                filename: filename.to_string(),
                success: false,
                error: Some(format!("Failed to decode PNG: {}", e)),
                original_width: 0,
                original_height: 0,
                new_width: 0,
                new_height: 0,
                original_palette: vec![],
                optimized_palette: vec![],
                original_colors: 0,
                optimized_colors: 0,
            };
        }
    };
    
    // Extract original palette info
    let original_info = extract_palette(&img.to_rgba8(), 8);
    
    // Resize if needed
    let resize_result = resize_image_if_needed(img, options);
    let original_width = resize_result.original_width;
    let original_height = resize_result.original_height;
    let img = resize_result.img;
    
    // Convert to RGB (JPEG doesn't support alpha)
    let rgb_image = img.to_rgb8();
    let (width, height) = rgb_image.dimensions();
    
    // Encode as JPEG
    match encode_jpeg(&rgb_image, width, height, options.quality) {
        Ok(jpeg_data) => {
            // Extract optimized palette info
            let optimized_info = extract_palette_from_bytes(&jpeg_data, ImageFormat::Jpeg);
            
            // Generate new filename with .jpg extension
            let new_filename = {
                let name = filename.trim_end_matches(".png").trim_end_matches(".PNG");
                format!("{}.jpg", name)
            };
            
            OptimizeResult {
                optimized_size: jpeg_data.len(),
                data: jpeg_data,
                original_size,
                format: "jpeg".to_string(),
                filename: new_filename,
                success: true,
                error: None,
                original_width,
                original_height,
                new_width: width,
                new_height: height,
                original_palette: original_info.palette,
                optimized_palette: optimized_info.palette,
                original_colors: original_info.color_count,
                optimized_colors: optimized_info.color_count,
            }
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
            original_palette: vec![],
            optimized_palette: vec![],
            original_colors: 0,
            optimized_colors: 0,
        },
    }
}
