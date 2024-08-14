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
}

impl GGmlType {
    pub const fn nbytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F16 => 2,
            Self::Q4_0 => todo!(),
            Self::Q4_1 => todo!(),
            Self::Q5_0 => todo!(),
            Self::Q5_1 => todo!(),
            Self::Q8_0 => todo!(),
            Self::Q8_1 => todo!(),
            Self::Q2K => todo!(),
            Self::Q3K => todo!(),
            Self::Q4K => todo!(),
            Self::Q5K => todo!(),
            Self::Q6K => todo!(),
            Self::Q8K => todo!(),
            Self::IQ2XXS => todo!(),
            Self::IQ2XS => todo!(),
            Self::IQ3XXS => todo!(),
            Self::IQ1S => todo!(),
            Self::IQ4NL => todo!(),
            Self::IQ3S => todo!(),
            Self::IQ2S => todo!(),
            Self::IQ4XS => todo!(),
            Self::I8 => size_of::<i8>(),
            Self::I16 => size_of::<i16>(),
            Self::I32 => size_of::<i32>(),
            Self::I64 => size_of::<i64>(),
            Self::F64 => size_of::<f64>(),
            Self::IQ1M => todo!(),
            _ => unimplemented!(),
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
    pub const fn nbytes(&self) -> usize {
        let mut ans = self.ty.nbytes();
        let mut i = 0;
        while i < self.ndim {
            ans *= unsafe { self.shape.as_ptr().add(i as _).read() as usize };
            i += 1;
        }
        ans
    }
}

impl Drop for GGufTensorInfo {
    fn drop(&mut self) {
        let ptr = self.shape.as_ptr().cast();
        let layout = Layout::array::<u64>(self.ndim as _).unwrap();
        unsafe { dealloc(ptr, layout) }
    }
}
