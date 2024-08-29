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
    pub const fn size(self) -> GGmlTypeSize {
        // See: GGML_QUANT_SIZES in <https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/constants.py>
        const QK_K: u32 = 256;
        const fn unit<T>() -> (u32, u32) {
            (1, size_of::<T>() as _)
        }
        #[rustfmt::skip]
        let (block_size, type_size) = match self {
            Self::F32      => unit::<f32>(),
            Self::F16      => unit::<u16>(),
            Self::Q4_0     => ( 32, 2 + 16),
            Self::Q4_1     => ( 32, 2 + 2 + 16),
            Self::Q5_0     => ( 32, 2 + 4 + 16),
            Self::Q5_1     => ( 32, 2 + 2 + 4 + 16),
            Self::Q8_0     => ( 32, 2 + 32),
            Self::Q8_1     => ( 32, 4 + 4 + 32),
            Self::Q2K      => (256, 2 + 2 + QK_K / 16 + QK_K / 4),
            Self::Q3K      => (256, 2 + QK_K / 4 + QK_K / 8 + 12),
            Self::Q4K      => (256, 2 + 2 + QK_K / 2 + 12),
            Self::Q5K      => (256, 2 + 2 + QK_K / 2 + QK_K / 8 + 12),
            Self::Q6K      => (256, 2 + QK_K / 2 + QK_K / 4 + QK_K / 16),
            Self::Q8K      => (256, 4 + QK_K + QK_K / 8),
            Self::IQ2XXS   => (256, 2 + QK_K / 4),
            Self::IQ2XS    => (256, 2 + QK_K / 4 + QK_K / 32),
            Self::IQ3XXS   => (256, 2 + QK_K / 4 + QK_K / 8),
            Self::IQ1S     => (256, 2 + QK_K / 8 + QK_K / 16),
            Self::IQ4NL    => ( 32, 2 + 16),
            Self::IQ3S     => (256, 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4),
            Self::IQ2S     => (256, 2 + QK_K / 4 + QK_K / 16),
            Self::IQ4XS    => (256, 2 + 2 + QK_K / 2 + QK_K / 64),
            Self::I8       => unit::<i8 >(),
            Self::I16      => unit::<i16>(),
            Self::I32      => unit::<i32>(),
            Self::I64      => unit::<i64>(),
            Self::F64      => unit::<f64>(),
            Self::IQ1M     => (256, QK_K / 8 + QK_K / 16  + QK_K / 32),
            Self::BF16     => unit::<u16>(),
            Self::Q4_0_4_4 => (32, 2 + 16),
            Self::Q4_0_4_8 => (32, 2 + 16),
            Self::Q4_0_8_8 => (32, 2 + 16),
            _              => unimplemented!(),
        };
        GGmlTypeSize {
            block_size,
            type_size,
        }
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
