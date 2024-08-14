use crate::metadata::GGufMetaDataValueType;
use std::{
    alloc::Layout,
    str::{from_utf8, from_utf8_unchecked, Utf8Error},
};

#[derive(Clone)]
#[repr(transparent)]
pub struct GGufReader<'a>(&'a [u8]);

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum GGufReadError {
    Eos,
    Utf8(Utf8Error),
    Bool(u8),
}

impl<'a> GGufReader<'a> {
    #[inline]
    pub const fn new(data: &'a [u8]) -> Self {
        Self(data)
    }

    #[inline]
    pub const fn remaining(&self) -> &'a [u8] {
        self.0
    }

    pub(crate) fn skip<T>(&mut self, len: usize) -> Result<&mut Self, GGufReadError> {
        let len = Layout::array::<T>(len).unwrap().size();
        let (_, tail) = self.0.split_at_checked(len).ok_or(GGufReadError::Eos)?;
        self.0 = tail;
        Ok(self)
    }

    pub(crate) fn skip_str(&mut self) -> Result<&mut Self, GGufReadError> {
        let len = self.read::<u64>()?;
        self.skip::<u8>(len as _)
    }

    pub fn read<T: Copy + 'static>(&mut self) -> Result<T, GGufReadError> {
        let ptr = self.0.as_ptr().cast::<T>();
        self.skip::<T>(1)?;
        Ok(unsafe { ptr.read_unaligned() })
    }

    pub fn read_bool(&mut self) -> Result<bool, GGufReadError> {
        match self.read::<u8>()? {
            0 => Ok(false),
            1 => Ok(true),
            e => Err(GGufReadError::Bool(e)),
        }
    }

    pub fn read_str(&mut self) -> Result<&'a str, GGufReadError> {
        let len = self.read::<u64>()? as _;
        let (s, tail) = self.0.split_at_checked(len).ok_or(GGufReadError::Eos)?;
        let ans = from_utf8(s).map_err(GGufReadError::Utf8)?;
        self.0 = tail;
        Ok(ans)
    }

    /// Read a string without checking if it is valid utf8.
    ///
    /// # Safety
    ///
    /// This function does not check if the data is valid utf8.
    pub unsafe fn read_str_unchecked(&mut self) -> &'a str {
        let len = self.read::<u64>().unwrap() as _;
        let (s, tail) = self.0.split_at(len);
        self.0 = tail;
        from_utf8_unchecked(s)
    }

    pub fn read_arr_header(&mut self) -> Result<(GGufMetaDataValueType, usize), GGufReadError> {
        Ok((self.read()?, self.read::<u64>()? as _))
    }
}
