use crate::{sizeof, tensor::GGmlType, GGufFileHeader, GGufMetaDataValueType};
use std::{
    io::{BufWriter, Result, Write},
    mem::size_of_val,
    slice::from_raw_parts,
};

pub struct GGufWriter<T: Write>(BufWriter<T>, usize);

impl<T: Write> GGufWriter<T> {
    #[inline]
    pub fn new(writer: T, header: GGufFileHeader) -> Result<Self> {
        let mut ans = Self(BufWriter::new(writer), 0);
        ans.write_bytes(as_slice(&header))?;
        Ok(ans)
    }

    #[inline]
    pub const fn written_bytes(&self) -> usize {
        self.1
    }

    #[inline]
    pub fn write_bytes(&mut self, val: &[u8]) -> Result<()> {
        self.1 += val.len();
        self.0.write_all(val.as_ref())
    }

    #[inline]
    pub fn write<U: Copy>(&mut self, val: U) -> Result<()> {
        self.write_bytes(as_slice(&val))
    }

    #[inline]
    pub fn write_bool(&mut self, val: bool) -> Result<()> {
        self.write_bytes(if val { &[1] } else { &[0] })
    }

    #[inline]
    pub fn write_str(&mut self, val: impl AsRef<str>) -> Result<()> {
        let val = val.as_ref();
        self.write_bytes(as_slice(&(val.len() as u64)))?;
        self.write_bytes(val.as_bytes())
    }

    #[inline]
    pub fn write_meta_kv(
        &mut self,
        key: impl AsRef<str>,
        ty: GGufMetaDataValueType,
        val: impl AsRef<[u8]>,
    ) -> Result<()> {
        self.write_str(key)?;
        self.write(ty)?;
        self.write_bytes(val.as_ref())
    }

    #[inline]
    pub fn write_tensor_info(
        &mut self,
        name: impl AsRef<str>,
        shape: &[u64],
        ty: GGmlType,
        offset: usize,
    ) -> Result<()> {
        self.write_str(name)?;
        self.write(shape.len() as u32)?;
        self.write_bytes(unsafe { from_raw_parts(shape.as_ptr().cast(), size_of_val(shape)) })?;
        self.write(ty)?;
        self.write(offset as u64)
    }

    #[inline]
    pub fn flush(&mut self) -> Result<()> {
        self.0.flush()
    }
}

#[inline(always)]
fn as_slice<T: Sized>(val: &T) -> &[u8] {
    unsafe { from_raw_parts(val as *const _ as *const _, sizeof!(T)) }
}
