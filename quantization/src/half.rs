use super::{QuantBlock, QuantizeError};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

impl QuantBlock for half::f16 {
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

    fn arr_len(n: usize) -> Result<usize, QuantizeError> {
        Ok(n)
    }

    fn quantize_f16(dst: &mut [Self], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() == src.len() {
            dst.copy_from_slice(src);
            Ok(())
        } else {
            Err(QuantizeError::LengthMismatch)
        }
    }

    fn dequantize_f16(dst: &mut [Self], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() == src.len() {
            dst.copy_from_slice(src);
            Ok(())
        } else {
            Err(QuantizeError::LengthMismatch)
        }
    }

    fn quantize_bf16(dst: &mut [Self], src: &[half::bf16]) -> Result<(), QuantizeError> {
        if dst.len() == src.len() {
            dst.into_par_iter().zip(src).for_each(|(y, &x)| {
                *y = Self::from_f32(x.to_f32());
            });
            Ok(())
        } else {
            Err(QuantizeError::LengthMismatch)
        }
    }

    fn dequantize_bf16(dst: &mut [half::bf16], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() == src.len() {
            dst.into_par_iter().zip(src).for_each(|(y, &x)| {
                *y = half::bf16::from_f32(x.to_f32());
            });
            Ok(())
        } else {
            Err(QuantizeError::LengthMismatch)
        }
    }
}

impl QuantBlock for half::bf16 {
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

    fn arr_len(n: usize) -> Result<usize, QuantizeError> {
        Ok(n)
    }

    fn quantize_bf16(dst: &mut [Self], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() == src.len() {
            dst.copy_from_slice(src);
            Ok(())
        } else {
            Err(QuantizeError::LengthMismatch)
        }
    }

    fn dequantize_bf16(dst: &mut [Self], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() == src.len() {
            dst.copy_from_slice(src);
            Ok(())
        } else {
            Err(QuantizeError::LengthMismatch)
        }
    }

    fn quantize_f16(dst: &mut [Self], src: &[half::f16]) -> Result<(), QuantizeError> {
        if dst.len() == src.len() {
            dst.into_par_iter().zip(src).for_each(|(y, &x)| {
                *y = Self::from_f32(x.to_f32());
            });
            Ok(())
        } else {
            Err(QuantizeError::LengthMismatch)
        }
    }

    fn dequantize_f16(dst: &mut [half::f16], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() == src.len() {
            dst.into_par_iter().zip(src).for_each(|(y, &x)| {
                *y = half::f16::from_f32(x.to_f32());
            });
            Ok(())
        } else {
            Err(QuantizeError::LengthMismatch)
        }
    }
}
