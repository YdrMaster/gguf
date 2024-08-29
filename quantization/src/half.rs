use super::{QuantBlock, QuantizeError};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

macro_rules! impl_half {
    ($ty:ty) => {
        impl QuantBlock for $ty {
            const BLOCK_SIZE: usize = 1;

            fn quantize(dst: &mut [Self], src: &[f32]) -> Result<(), QuantizeError> {
                if dst.len() != src.len() {
                    return Err(QuantizeError::LengthMismatch);
                }

                dst.into_par_iter()
                    .zip(src)
                    .for_each(|(y, &x)| *y = Self::from_f32(x));
                Ok(())
            }

            fn dequantize(dst: &mut [f32], src: &[Self]) -> Result<(), QuantizeError> {
                if dst.len() != src.len() {
                    return Err(QuantizeError::LengthMismatch);
                }

                dst.into_par_iter()
                    .zip(src)
                    .for_each(|(y, x)| *y = x.to_f32());
                Ok(())
            }
        }
    };
}

impl_half!(half::f16);
impl_half!(half::bf16);
