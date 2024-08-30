use super::{QuantBlock, QuantizeError};
use half::f16;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{iter::zip, slice::from_raw_parts_mut};

pub struct Q8_0 {
    delta: f16,
    quants: [i8; BLOCK_SIZE],
}

const BLOCK_SIZE: usize = 32;

impl QuantBlock for Q8_0 {
    const BLOCK_SIZE: usize = BLOCK_SIZE;

    fn quantize(dst: &mut [Self], src: &[f32]) -> Result<(), QuantizeError> {
        if Self::arr_len(src.len())? != dst.len() {
            return Err(QuantizeError::LengthMismatch);
        }
        dst.into_par_iter().enumerate().for_each(|(i, y)| {
            let x = &src[i * BLOCK_SIZE..][..BLOCK_SIZE];
            let amax = x.iter().fold(0., |acc, x| x.abs().max(acc));

            let delta = amax / i8::MAX as f32;
            let recip = if delta == 0. { 0. } else { delta.recip() };

            y.delta = f16::from_f32(delta);
            for (y, &x) in zip(&mut y.quants, x) {
                *y = (x * recip).round() as _;
            }
        });
        Ok(())
    }

    fn dequantize(dst: &mut [f32], src: &[Self]) -> Result<(), QuantizeError> {
        if Self::arr_len(dst.len())? != src.len() {
            return Err(QuantizeError::LengthMismatch);
        }
        let dst = dst.as_mut_ptr() as usize;
        src.into_par_iter().enumerate().for_each(|(i, x)| {
            let dst = unsafe { from_raw_parts_mut((dst + i * BLOCK_SIZE) as *mut f32, BLOCK_SIZE) };
            let delta = x.delta.to_f32();
            for (y, &x) in zip(dst, &x.quants) {
                *y = x as f32 * delta;
            }
        });
        Ok(())
    }
}
