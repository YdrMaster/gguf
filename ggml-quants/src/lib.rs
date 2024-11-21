use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::slice::{from_raw_parts, from_raw_parts_mut};

pub trait DataBlock: Sized + 'static {
    #[cfg(feature = "types")]
    const ID: digit_layout::DigitLayout;
    const COUNT: usize;
    const ZEROS: Self;
}

macro_rules! impl_data_block {
    ($ty:ty = $id:expr; $zero:expr ) => {
        impl DataBlock for $ty {
            #[cfg(feature = "types")]
            const ID: digit_layout::DigitLayout = $id;
            const COUNT: usize = Self::ID.group_size();
            const ZEROS: Self = $zero;
        }
    };
}

use digit_layout::types as ty;
impl_data_block!(u8  = ty::U8 ; 0 );
impl_data_block!(i8  = ty::I8 ; 0 );
impl_data_block!(u16 = ty::U16; 0 );
impl_data_block!(i16 = ty::I16; 0 );
impl_data_block!(u32 = ty::U32; 0 );
impl_data_block!(i32 = ty::I32; 0 );
impl_data_block!(f32 = ty::F32; 0.);
impl_data_block!(u64 = ty::U64; 0 );
impl_data_block!(i64 = ty::I64; 0 );
impl_data_block!(f64 = ty::F64; 0.);

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

#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod test_utils {
    use crate::Quantize;
    use std::fmt;

    pub fn test<const N: usize, T: Quantize<f32, N>>(abs: f32, rel: f32) {
        use rand::Rng;
        use std::iter::zip;

        let mut data = [0.0f32; N];
        rand::thread_rng().fill(&mut data[..]);

        let quant = T::quantize(&data);
        let dequant = T::dequantize(&quant);

        let mut ec = ErrorCollector::new(abs, rel);
        for (a, b) in zip(data, dequant) {
            ec.push(Diff::new(a, b))
        }
        println!("{ec}");

        for &i in ec.outliers() {
            println!("{} vs {}", data[i], dequant[i]);
        }

        assert!(ec.outliers().is_empty());
    }

    struct Diff {
        pub abs: f32,
        pub rel: f32,
    }

    impl Diff {
        fn new(a: f32, b: f32) -> Self {
            let abs = (a - b).abs();
            let rel = abs / (a.abs() + b.abs() + f32::EPSILON);
            Self { abs, rel }
        }
    }

    struct ErrorCollector {
        threshold: Diff,
        max_diff: Diff,
        outliers: Vec<usize>,
        count: usize,
    }

    impl ErrorCollector {
        fn new(abs: f32, rel: f32) -> Self {
            Self {
                threshold: Diff { abs, rel },
                max_diff: Diff { abs: 0.0, rel: 0.0 },
                outliers: vec![],
                count: 0,
            }
        }

        fn push(&mut self, diff: Diff) {
            self.max_diff.abs = f32::max(self.max_diff.abs, diff.abs);
            self.max_diff.rel = f32::max(self.max_diff.rel, diff.rel);

            if diff.abs > self.threshold.abs && diff.rel > self.threshold.rel {
                self.outliers.push(self.count);
            }

            self.count += 1;
        }

        fn outliers(&self) -> &[usize] {
            &self.outliers
        }
    }

    impl fmt::Display for ErrorCollector {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(
                f,
                "abs: {:.3e}, rel: {:.3e}, outliers: {}/{}",
                self.max_diff.abs,
                self.max_diff.rel,
                self.outliers.len(),
                self.count,
            )
        }
    }
}
