use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub trait DataBlock: Sized + 'static {
    const COUNT: usize;
    const ZEROS: Self;
}

macro_rules! impl_data_block {
    ($ty:ty, $zero:expr) => {
        impl DataBlock for $ty {
            const COUNT: usize = 1;
            const ZEROS: Self = $zero;
        }
    };
}

impl_data_block!(u8, 0);
impl_data_block!(i8, 0);
impl_data_block!(u16, 0);
impl_data_block!(i16, 0);
impl_data_block!(u32, 0);
impl_data_block!(i32, 0);
impl_data_block!(f32, 0.);
impl_data_block!(u64, 0);
impl_data_block!(i64, 0);
impl_data_block!(f64, 0.);
impl_data_block!(u128, 0);
impl_data_block!(i128, 0);

pub trait Quantize<T, const N: usize>: DataBlock {
    fn quantize(data: &[T; N]) -> Self;
    fn dequantize(&self) -> [T; N];
}

impl<Blk, const N: usize> Quantize<f16, N> for Blk
where
    Blk: Quantize<f32, N>,
{
    #[inline]
    fn quantize(data: &[f16; N]) -> Self {
        Self::quantize(&data.map(f16::to_f32))
    }
    #[inline]
    fn dequantize(&self) -> [f16; N] {
        self.dequantize().map(f16::from_f32)
    }
}

impl<Blk, const N: usize> Quantize<bf16, N> for Blk
where
    Blk: Quantize<f32, N>,
{
    #[inline]
    fn quantize(data: &[bf16; N]) -> Self {
        Self::quantize(&data.map(bf16::to_f32))
    }
    #[inline]
    fn dequantize(&self) -> [bf16; N] {
        self.dequantize().map(bf16::from_f32)
    }
}

pub trait QuantExt<T, const N: usize>: Sized {
    fn quantize_slice(dst: &mut [Self], src: &[T]) -> Result<(), QuantizeError>;
    fn dequantize_slice(dst: &mut [T], src: &[Self]) -> Result<(), QuantizeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizeError {
    Indivisible,
    LengthMismatch,
}

impl<Blk, T, const N: usize> QuantExt<T, N> for Blk
where
    Blk: Quantize<T, N> + Send + Sync,
    T: Send + Sync,
{
    fn quantize_slice(dst: &mut [Self], src: &[T]) -> Result<(), QuantizeError> {
        if src.len() % N != 0 {
            return Err(QuantizeError::Indivisible);
        }
        if dst.len() != src.len() / N {
            return Err(QuantizeError::LengthMismatch);
        }
        let src = unsafe { from_raw_parts(src.as_ptr().cast::<[T; N]>(), dst.len()) };
        dst.into_par_iter()
            .zip(src)
            .for_each(|(dst, src)| *dst = Blk::quantize(src));
        Ok(())
    }

    fn dequantize_slice(dst: &mut [T], src: &[Self]) -> Result<(), QuantizeError> {
        if dst.len() % N != 0 {
            return Err(QuantizeError::Indivisible);
        }
        if src.len() != dst.len() / N {
            return Err(QuantizeError::LengthMismatch);
        }
        let dst = unsafe { from_raw_parts_mut(dst.as_mut_ptr().cast::<[T; N]>(), src.len()) };
        src.into_par_iter()
            .zip(dst)
            .for_each(|(src, dst)| *dst = Blk::dequantize(src));
        Ok(())
    }
}

mod structs;
pub use structs::*;

#[cfg(feature = "types")]
pub extern crate digit_layout;

#[cfg(feature = "types")]
pub mod types;
