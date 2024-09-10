use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub trait DataBlock: Sized + 'static {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout;
    const COUNT: usize;
    const ZEROS: Self;
}

macro_rules! impl_data_block {
    ($id:ident $ty:ty; $zero:expr ) => {
        impl DataBlock for $ty {
            #[cfg(feature = "types")]
            const ID: digit_layout::DigitLayout = digit_layout::types::$id;
            const COUNT: usize = 1;
            const ZEROS: Self = $zero;
        }
    };
}

impl_data_block!(U8  u8  ; 0 );
impl_data_block!(I8  i8  ; 0 );
impl_data_block!(U16 u16 ; 0 );
impl_data_block!(I16 i16 ; 0 );
impl_data_block!(U32 u32 ; 0 );
impl_data_block!(I32 i32 ; 0 );
impl_data_block!(F32 f32 ; 0.);
impl_data_block!(U64 u64 ; 0 );
impl_data_block!(I64 i64 ; 0 );
impl_data_block!(F64 f64 ; 0.);

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
