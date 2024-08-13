use crate::{GGufReadError, GGufReader};
use indexmap::IndexMap;
use std::{
    alloc::{alloc, dealloc, Layout},
    hash::Hash,
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
    pub fn nbytes(self) -> usize {
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

#[derive(Clone, Debug)]
pub struct GGufTensors<'a> {
    indices: IndexMap<&'a str, ()>,
    nbytes: usize,
}

impl<'a> GGufTensors<'a> {
    pub fn scan(count: u64, data: &'a [u8]) -> Result<Self, GGufReadError> {
        let mut reader = GGufReader::new(data);
        let mut indices = IndexMap::with_capacity(count as _);
        for _ in 0..count {
            let name = reader.read_str()?;
            let ndim = reader.read::<u32>()? as usize;
            reader.skip::<u64>(ndim)?;
            reader.skip::<u32>(1)?;
            reader.skip::<u64>(1)?;
            if indices.insert(name, ()).is_some() {
                return Err(GGufReadError::DuplicatedKey(name.into()));
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

pub struct GGufTensorInfo<'a> {
    name: &'a str,
    ty: GGmlType,
    ndim: u32,
    offset: usize,
    shape: NonNull<u64>,
}

unsafe impl Sync for GGufTensorInfo<'_> {}

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
        unsafe {
            dealloc(
                self.shape.as_ptr().cast(),
                Layout::array::<u64>(self.ndim as _).unwrap(),
            )
        }
    }
}

impl<'a> GGufTensorInfo<'a> {
    fn new(name: &'a str) -> Self {
        unsafe {
            let ptr = name.as_ptr().add(name.len());
            let ndim = ptr.cast::<u32>().read_unaligned();
            let layout = Layout::array::<u64>(ndim as _).unwrap();

            let ptr = ptr.add(size_of::<u32>());
            let shape_ = ptr;

            let ptr = ptr.add(layout.size());
            let ty = ptr.cast::<GGmlType>().read_unaligned();

            let ptr = ptr.add(size_of::<u32>());
            let offset = ptr.cast::<u64>().read_unaligned() as _;

            let shape = alloc(layout).cast::<u64>();
            copy_nonoverlapping(shape_, shape.cast(), layout.size());

            Self {
                name,
                ty,
                ndim,
                offset,
                shape: NonNull::new_unchecked(shape),
            }
        }
    }

    #[inline]
    pub fn name(&self) -> &'a str {
        self.name
    }

    #[inline]
    pub fn shape(&self) -> &[u64] {
        unsafe { from_raw_parts(self.shape.as_ptr(), self.ndim as _) }
    }

    #[inline]
    pub fn ggml_type(&self) -> GGmlType {
        self.ty
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.shape().iter().product::<u64>() as usize * self.ggml_type().nbytes()
    }
}
