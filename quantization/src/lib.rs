mod half;
mod q8_0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizeError {
    Indivisible,
    LengthMismatch,
}

pub trait QuantBlock: Sized {
    const BLOCK_SIZE: usize;

    fn size_of_array(n: usize) -> Result<usize, QuantizeError> {
        if n % Self::BLOCK_SIZE == 0 {
            Ok(n / Self::BLOCK_SIZE * size_of::<Self>())
        } else {
            Err(QuantizeError::Indivisible)
        }
    }
    fn quantize(dst: &mut [Self], src: &[f32]) -> Result<(), QuantizeError>;
    fn dequantize(dst: &mut [f32], src: &[Self]) -> Result<(), QuantizeError>;
}

pub use ::half::{bf16, f16};
pub use q8_0::Q8_0;
