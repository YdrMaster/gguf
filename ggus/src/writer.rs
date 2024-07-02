use crate::{sizeof, GGufFileHeader};
use std::{
    io::{BufWriter, Result, Write},
    slice::from_raw_parts,
};

pub struct GGufWriter<T: Write>(BufWriter<T>);

impl<T: Write> GGufWriter<T> {
    #[inline]
    pub fn new(writer: T, header: GGufFileHeader) -> Result<Self> {
        let mut buf = BufWriter::new(writer);
        buf.write_all(as_slice(&header))?;
        Ok(Self(buf))
    }

    #[inline]
    pub fn write<U: Copy>(&mut self, val: U) -> Result<()> {
        self.0.write_all(as_slice(&val))
    }

    #[inline]
    pub fn read_bool(&mut self, val: bool) -> Result<()> {
        self.0.write_all(if val { &[1] } else { &[0] })
    }

    #[inline]
    pub fn read_str(&mut self, val: impl AsRef<str>) -> Result<()> {
        let val = val.as_ref();
        self.0.write_all(as_slice(&(val.len() as u64)))?;
        self.0.write_all(val.as_bytes())
    }
}

#[inline(always)]
fn as_slice<T: Sized>(val: &T) -> &[u8] {
    unsafe { from_raw_parts(val as *const _ as *const _, sizeof!(T)) }
}
