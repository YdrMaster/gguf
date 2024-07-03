use indexmap::IndexMap;

use crate::{
    reader::{GGufReadError, GGufReader},
    sizeof,
};
use std::{
    alloc::{alloc, dealloc, Layout},
    hash::Hash,
    marker::PhantomData,
    ptr::{copy_nonoverlapping, NonNull},
    slice::from_raw_parts,
    str::from_utf8_unchecked,
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
    fn nbytes(self) -> usize {
        match self {
            Self::F32 => sizeof!(f32),
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
            Self::I8 => sizeof!(i8),
            Self::I16 => sizeof!(i16),
            Self::I32 => sizeof!(i32),
            Self::I64 => sizeof!(i64),
            Self::F64 => sizeof!(f64),
            Self::IQ1M => todo!(),
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GGufTensors<'a> {
    indices: IndexMap<&'a str, ()>,
    nbytes: usize,
}

impl<'a> GGufTensors<'a> {
    pub fn scan(count: u64, data: &'a [u8]) -> Result<Self, GGufReadError<'a>> {
        let mut reader = GGufReader::new(data);
        let mut indices = IndexMap::with_capacity(count as _);
        for _ in 0..count {
            let name = reader.read_str()?;
            let ndim = reader.read::<u32>()? as usize;
            reader.skip::<u64>(ndim)?;
            reader.skip::<u32>(1)?;
            reader.skip::<u64>(1)?;
            if indices.insert(name, ()).is_some() {
                return Err(GGufReadError::DuplicatedKey(name));
            }
        }
        Ok(Self {
            indices,
            nbytes: reader.cursor(),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[inline]
    pub fn names<'s>(&'s self) -> impl Iterator<Item = &'a str> + 's {
        self.indices.keys().copied()
    }

    #[inline]
    pub fn iter<'s>(&'s self) -> impl Iterator<Item = GGufTensorInfo<'a>> + 's {
        self.indices.keys().map(|name| GGufTensorInfo::new(name))
    }

    #[inline]
    pub const fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn get(&self, name: &str) -> Option<GGufTensorInfo<'a>> {
        self.indices
            .get_key_value(name)
            .map(|(name, ())| GGufTensorInfo::new(name))
    }
}

// | name::ptr | name::len | offset | ggml_type;ndim | shape ..
#[repr(transparent)]
pub struct GGufTensorInfo<'a>(NonNull<u64>, PhantomData<&'a ()>);

impl PartialEq for GGufTensorInfo<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for GGufTensorInfo<'_> {}

impl Hash for GGufTensorInfo<'_> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state)
    }
}

impl Drop for GGufTensorInfo<'_> {
    #[inline]
    fn drop(&mut self) {
        unsafe { dealloc(self.0.as_ptr().cast(), layout(self.ndim())) }
    }
}

impl<'a> GGufTensorInfo<'a> {
    fn new(name: &'a str) -> Self {
        unsafe {
            let ptr = name.as_ptr().add(name.len());
            let ndim = ptr.cast::<u32>().read_unaligned() as usize;

            let ptr = ptr.add(sizeof!(u32));
            let shape = ptr;

            let ptr = ptr.add(ndim * sizeof!(u64));
            let ggml_type = ptr.cast::<GGmlType>().read_unaligned();

            let ptr = ptr.add(sizeof!(u32));
            let offset = ptr.cast::<u64>().read_unaligned();

            let body = alloc(layout(ndim)).cast::<u64>();
            body.write(name.as_ptr() as _);
            body.add(1).write(name.len() as _);
            body.add(2).write(offset);
            body.add(3).cast::<GGmlType>().write(ggml_type);
            body.add(3).cast::<u32>().add(1).write(ndim as _);
            copy_nonoverlapping(shape, body.add(4).cast(), ndim * sizeof!(u64));
            Self(NonNull::new_unchecked(body), PhantomData)
        }
    }

    #[inline]
    pub fn name(&self) -> &'a str {
        unsafe {
            let ptr = self.0.as_ptr().read();
            let len = self.0.as_ptr().add(1).read();
            from_utf8_unchecked(from_raw_parts(ptr as _, len as _))
        }
    }

    #[inline]
    fn ndim(&self) -> usize {
        unsafe { self.0.as_ptr().add(3).cast::<u32>().add(1).read() as _ }
    }

    #[inline]
    pub fn shape(&self) -> &[u64] {
        unsafe { from_raw_parts(self.0.as_ptr().add(4), self.ndim()) }
    }

    #[inline]
    pub fn ggml_type(&self) -> GGmlType {
        unsafe { self.0.as_ptr().add(3).cast::<GGmlType>().read() }
    }

    #[inline]
    pub fn offset(&self) -> usize {
        unsafe { self.0.as_ptr().add(2).read() as _ }
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.shape().iter().product::<u64>() as usize * self.ggml_type().nbytes()
    }
}

#[inline(always)]
fn layout(ndim: usize) -> Layout {
    Layout::array::<u64>(4 + ndim).unwrap()
}
