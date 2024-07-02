use crate::{
    reader::{GGmlReadError, GGmlReader},
    sizeof,
};
use std::{
    alloc::{alloc, dealloc, Layout},
    collections::HashMap,
    ptr::NonNull,
    slice::from_raw_parts,
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
    pub fn scan(count: usize, data: &'a [u8]) -> Result<Self, GGmlReadError<'a>> {
        let mut reader = GGmlReader::new(data);
        let mut indices = HashMap::with_capacity(count);
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
    pub const fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn get(&self, name: &str) -> Option<TensorInfo<'a>> {
        self.indices.get_key_value(name).map(|(name, ())| unsafe {
            let ptr = name.as_ptr().add(name.len());
            let ndim = ptr.cast::<u32>().read_unaligned() as usize;
            let layout = body_layout(ndim);
            let body = alloc(layout).cast::<u64>();
            body.cast::<u32>().write(ndim as _);

            let ptr = ptr.add(sizeof!(u32));
            std::ptr::copy_nonoverlapping(ptr, body.add(2).cast(), ndim * sizeof!(u64));

            let ptr = ptr.add(ndim * sizeof!(u64));
            std::ptr::copy_nonoverlapping(ptr, body.cast::<u32>().add(1).cast(), sizeof!(u32));

            let ptr = ptr.add(sizeof!(u32));
            std::ptr::copy_nonoverlapping(ptr, body.add(1).cast(), sizeof!(u64));

            TensorInfo {
                name,
                body: NonNull::new_unchecked(body),
            }
        })
    }
}

pub struct TensorInfo<'a> {
    name: &'a str,
    body: NonNull<u64>,
}

impl Drop for TensorInfo<'_> {
    #[inline]
    fn drop(&mut self) {
        let layout = body_layout(self.ndim());
        unsafe { dealloc(self.body.as_ptr().cast(), layout) };
    }
}

impl<'a> TensorInfo<'a> {
    #[inline]
    pub const fn name(&self) -> &'a str {
        self.name
    }

    #[inline]
    fn ndim(&self) -> usize {
        unsafe { *self.body.cast::<u32>().as_ptr() as _ }
    }

    #[inline]
    pub fn shape(&self) -> &[u64] {
        unsafe { from_raw_parts(self.body.as_ptr().add(2), self.ndim()) }
    }

    #[inline]
    pub fn ggml_type(&self) -> GGmlType {
        unsafe { *self.body.cast::<GGmlType>().as_ptr().add(1) }
    }

    #[inline]
    pub fn offset(&self) -> usize {
        unsafe { *self.body.as_ptr().add(1) as _ }
    }
}

#[inline(always)]
fn body_layout(ndim: usize) -> Layout {
    Layout::array::<u64>(ndim + 2).unwrap()
}
