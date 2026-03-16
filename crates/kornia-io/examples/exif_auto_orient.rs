use std::env;

use kornia_io::jpeg::{read_image_jpeg_rgb8, write_image_jpeg_rgb8};
use kornia_io::metadata::read_image_metadata;
use kornia_io::read_image_jpeg_auto_orient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = args
        .next()
        .unwrap_or_else(|| "/home/incharanew/Downloads/landscape_3.jpg".to_string());
    let before = args
        .next()
        .unwrap_or_else(|| "/tmp/exif_before_raw.jpg".to_string());
    let after = args
        .next()
        .unwrap_or_else(|| "/tmp/exif_after_auto_orient.jpg".to_string());

    let meta = read_image_metadata(&input)?;
    println!(
        "EXIF orientation: {:?}",
        meta.exif_orientation.map(|v| v.get())
    );

    let raw = read_image_jpeg_rgb8(&input)?;
    let fixed = read_image_jpeg_auto_orient(&input)?;

    println!("raw dims  : {}x{}", raw.cols(), raw.rows());
    println!("fixed dims: {}x{}", fixed.cols(), fixed.rows());

    let fixed_u8 = fixed
        .cast::<u8>()
        .map_err(kornia_io::error::IoError::ImageCreationError)?;

    write_image_jpeg_rgb8(before.as_str(), &raw, 95)?;
    write_image_jpeg_rgb8(after.as_str(), &fixed_u8, 95)?;

    println!("wrote raw  : {before}");
    println!("wrote fixed: {after}");
    Ok(())
}
