use crate::{GGufReadError, GGufReader};
use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::{copy_nonoverlapping, NonNull},
    slice::from_raw_parts,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u32)]
pub enum GGmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    #[deprecated = "support removed"]
    Q4_2 = 4,
    #[deprecated = "support removed"]
    Q4_3 = 5,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
}

#[derive(Clone, Copy, Debug)]
pub struct GGmlTypeSize {
    pub block_size: u32,
    pub type_size: u32,
}

impl GGmlTypeSize {
    #[inline]
    const fn unit<T>() -> Self {
        Self {
            block_size: 1,
            type_size: size_of::<T>() as _,
        }
    }

    #[inline]
    const fn quants<T: ggml_quants::DataBlock>() -> Self {
        Self {
            block_size: T::COUNT as _,
            type_size: size_of::<T>() as _,
        }
    }

    #[inline]
    pub fn elements_to_bytes(&self, shape: &[u64]) -> usize {
        let blk = self.block_size as u64;
        let ele = self.type_size as u64;
        match shape {
            [] => {
                assert_eq!(blk, 1);
                ele as _
            }
            [last, others @ ..] => {
                assert_eq!(last % blk, 0);
                (others.iter().product::<u64>() * last / blk * ele) as _
            }
        }
    }
}

impl GGmlType {
    #[rustfmt::skip]
    pub const fn size(self) -> GGmlTypeSize {
        macro_rules! size {
            (t: $ty:ty) => { GGmlTypeSize::  unit::<$ty>() };
            (q: $ty:ty) => { GGmlTypeSize::quants::<$ty>() };
        }

        use ggml_quants::*;
        match self {
            Self::F32      => size!(t: f32   ),
            Self::F16      => size!(q: f16   ),
            Self::Q4_0     => size!(q: Q4_0  ),
            Self::Q4_1     => size!(q: Q4_1  ),
            Self::Q5_0     => size!(q: Q5_0  ),
            Self::Q5_1     => size!(q: Q5_1  ),
            Self::Q8_0     => size!(q: Q8_0  ),
            Self::Q8_1     => size!(q: Q8_1  ),
            Self::Q2K      => size!(q: Q2K   ),
            Self::Q3K      => size!(q: Q3K   ),
            Self::Q4K      => size!(q: Q4K   ),
            Self::Q5K      => size!(q: Q5K   ),
            Self::Q6K      => size!(q: Q6K   ),
            Self::Q8K      => size!(q: Q8K   ),
            Self::IQ2XXS   => size!(q: IQ2XXS),
            Self::IQ2XS    => size!(q: IQ2XS ),
            Self::IQ3XXS   => size!(q: IQ3XXS),
            Self::IQ1S     => size!(q: IQ1S  ),
            Self::IQ4NL    => size!(q: IQ4NL ),
            Self::IQ3S     => size!(q: IQ3S  ),
            Self::IQ2S     => size!(q: IQ2S  ),
            Self::IQ4XS    => size!(q: IQ4XS ),
            Self::I8       => size!(t: i8    ),
            Self::I16      => size!(t: i16   ),
            Self::I32      => size!(t: i32   ),
            Self::I64      => size!(t: i64   ),
            Self::F64      => size!(t: f64   ),
            Self::IQ1M     => size!(q: IQ1M  ),
            Self::BF16     => size!(q: bf16   ),
            Self::Q4_0_4_4 |
            Self::Q4_0_4_8 |
            Self::Q4_0_8_8 => todo!(),
            _              => unimplemented!(),
        }
    }

    #[cfg(feature = "types")]
    pub const fn to_digit_layout(self) -> ggml_quants::digit_layout::DigitLayout {
        use ggml_quants::{digit_layout::types as primitive, types as quantized};
        #[rustfmt::skip]
        let ans = match self {
            Self::F32    => primitive::F32   ,
            Self::F16    => primitive::F16   ,
            Self::BF16   => primitive::BF16  ,
            Self::Q8_0   => quantized::Q8_0  ,
            Self::Q8_1   => quantized::Q8_1  ,
            Self::Q4_0   => quantized::Q4_0  ,
            Self::Q4_1   => quantized::Q4_1  ,
            Self::Q5_0   => quantized::Q5_0  ,
            Self::Q5_1   => quantized::Q5_1  ,
            Self::Q2K    => quantized::Q2K   ,
            Self::Q3K    => quantized::Q3K   ,
            Self::Q4K    => quantized::Q4K   ,
            Self::Q5K    => quantized::Q5K   ,
            Self::Q6K    => quantized::Q6K   ,
            Self::Q8K    => quantized::Q8K   ,
            Self::IQ2XXS => quantized::IQ2XXS,
            Self::IQ2XS  => quantized::IQ2XS ,
            Self::IQ3XXS => quantized::IQ3XXS,
            Self::IQ1S   => quantized::IQ1S  ,
            Self::IQ4NL  => quantized::IQ4NL ,
            Self::IQ3S   => quantized::IQ3S  ,
            Self::IQ2S   => quantized::IQ2S  ,
            Self::IQ4XS  => quantized::IQ4XS ,
            Self::IQ1M   => quantized::IQ1M  ,
            Self::I8     => primitive::I8    ,
            Self::I16    => primitive::I16   ,
            Self::I32    => primitive::I32   ,
            Self::I64    => primitive::I64   ,
            Self::F64    => primitive::F64   ,
            _            => todo!()          ,
        };
        ans
    }
}

#[repr(transparent)]
pub struct GGufTensorMeta<'a>(&'a [u8]);

impl<'a> GGufReader<'a> {
    pub fn read_tensor_meta(&mut self) -> Result<GGufTensorMeta<'a>, GGufReadError> {
        let data = self.remaining();

        let _ = self.read_str()?;
        let ndim: u32 = self.read()?;
        self.skip::<u64>(ndim as _)?
            .skip::<GGmlType>(1)?
            .skip::<u64>(1)?;

        let data = &data[..data.len() - self.remaining().len()];
        Ok(unsafe { GGufTensorMeta::new_unchecked(data) })
    }
}

impl<'a> GGufTensorMeta<'a> {
    /// Creates a new [GGufTensorMeta] instance without performing any validation on the input data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the input data is valid for the [GGufTensorMeta] type.
    #[inline]
    pub const unsafe fn new_unchecked(data: &'a [u8]) -> Self {
        Self(data)
    }

    #[inline]
    pub fn new(data: &'a [u8]) -> Result<Self, GGufReadError> {
        GGufReader::new(data).read_tensor_meta()
    }

    #[inline]
    pub fn name(&self) -> &'a str {
        let mut reader = GGufReader::new(self.0);
        unsafe { reader.read_str_unchecked() }
    }

    #[inline]
    pub fn to_info(&self) -> GGufTensorInfo {
        let mut reader = GGufReader::new(self.0);
        let ndim: u32 = reader.skip_str().unwrap().read().unwrap();
        let layout = Layout::array::<u64>(ndim as _).unwrap();
        let shape = unsafe {
            let dst = alloc(layout);
            copy_nonoverlapping(reader.remaining().as_ptr(), dst, layout.size());
            NonNull::new_unchecked(dst).cast()
        };
        let ty = reader.skip::<u64>(ndim as _).unwrap().read().unwrap();
        let offset = reader.read().unwrap();

        GGufTensorInfo {
            ty,
            ndim,
            shape,
            offset,
        }
    }
}

pub struct GGufTensorInfo {
    ty: GGmlType,
    ndim: u32,
    shape: NonNull<u64>,
    offset: u64,
}

impl GGufTensorInfo {
    #[inline]
    pub const fn ty(&self) -> GGmlType {
        self.ty
    }

    #[inline]
    pub const fn shape(&self) -> &[u64] {
        unsafe { from_raw_parts(self.shape.as_ptr(), self.ndim as _) }
    }

    #[inline]
    pub const fn offset(&self) -> usize {
        self.offset as _
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.ty.size().elements_to_bytes(self.shape())
    }
}

impl Drop for GGufTensorInfo {
    fn drop(&mut self) {
        let ptr = self.shape.as_ptr().cast();
        let layout = Layout::array::<u64>(self.ndim as _).unwrap();
        unsafe { dealloc(ptr, layout) }
    }
}
