//! Smoke tests that do not require a working GPU.

#[cfg(feature = "vulkan")]
#[test]
fn vkfft_version_is_linked() {
    let v = gpufft_vulkan_sys::vkfft_runtime_version();
    assert!(v >= 10000, "VkFFT version suspiciously low: {v}");
}

#[test]
fn scalar_metadata() {
    use gpufft::{Precision, Scalar};
    use num_complex::Complex32;

    assert_eq!(<Complex32 as Scalar>::BYTES, 8);
    assert!(<Complex32 as Scalar>::IS_COMPLEX);
    assert_eq!(<Complex32 as Scalar>::PRECISION, Precision::F32);

    assert_eq!(<f32 as Scalar>::BYTES, 4);
    assert!(!<f32 as Scalar>::IS_COMPLEX);
    assert_eq!(<f32 as Scalar>::PRECISION, Precision::F32);
}

#[test]
fn shape_elements_and_rank() {
    use gpufft::Shape;
    assert_eq!(Shape::D1(128).elements(), 128);
    assert_eq!(Shape::D1(128).rank(), 1);
    assert_eq!(Shape::D2([16, 32]).elements(), 512);
    assert_eq!(Shape::D2([16, 32]).rank(), 2);
    assert_eq!(Shape::D3([4, 8, 16]).elements(), 512);
    assert_eq!(Shape::D3([4, 8, 16]).rank(), 3);
}
