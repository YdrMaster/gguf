use crate::{
    reader::{GGmlReadError, GGmlReader},
    sizeof,
};
use std::{
    alloc::{alloc, dealloc, Layout},
    collections::HashMap,
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

#[derive(Clone, Debug)]
pub struct GGufTensors<'a> {
    indices: HashMap<&'a str, ()>,
    nbytes: usize,
}

impl<'a> GGufTensors<'a> {
    pub fn scan(count: u64, data: &'a [u8]) -> Result<Self, GGmlReadError<'a>> {
        let mut reader = GGmlReader::new(data);
        let mut indices = HashMap::with_capacity(count as _);
        for _ in 0..count {
            let name = reader.read_str()?;
            let ndim = reader.read::<u32>()? as usize;
            reader.skip::<u64>(ndim)?;
            reader.skip::<u32>(1)?;
            reader.skip::<u64>(1)?;
            if indices.insert(name, ()).is_some() {
                return Err(GGmlReadError::DuplicatedKey(name));
            }
        }
        Ok(Self {
            indices,
            nbytes: reader.cursor(),
        })
    }

    #[inline]
    pub fn names<'s>(&'s self) -> impl Iterator<Item = &'a str> + 's {
        self.indices.keys().copied()
    }

    #[inline]
    pub fn iter<'s>(&'s self) -> impl Iterator<Item = TensorInfo<'a>> + 's {
        self.indices.iter().map(|(name, _)| TensorInfo::new(name))
    }

    #[inline]
    pub const fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn get(&self, name: &str) -> Option<TensorInfo<'a>> {
        self.indices
            .get_key_value(name)
            .map(|(name, ())| TensorInfo::new(name))
    }
}

// | name::ptr | name::len | offset | ggml_type;ndim | shape ..
#[repr(transparent)]
pub struct TensorInfo<'a>(NonNull<u64>, PhantomData<&'a ()>);

impl Drop for TensorInfo<'_> {
    #[inline]
    fn drop(&mut self) {
        unsafe { dealloc(self.0.as_ptr().cast(), layout(self.ndim())) }
    }
}

impl<'a> TensorInfo<'a> {
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
}

#[inline(always)]
fn layout(ndim: usize) -> Layout {
    Layout::array::<u64>(4 + ndim).unwrap()
}
