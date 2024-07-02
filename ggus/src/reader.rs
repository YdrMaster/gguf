use crate::{sizeof, GGufMetaDataValueType};

pub struct GGufReader<'a> {
    data: &'a [u8],
    cursor: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GGufReadError<'a> {
    Eos,
    DuplicatedKey(&'a str),
    Utf8(std::str::Utf8Error),
    Bool(u8),
}

impl<'a> GGufReader<'a> {
    #[inline]
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Self { data, cursor: 0 }
    }

    #[inline]
    pub(crate) const fn cursor(&self) -> usize {
        self.cursor
    }

    pub(crate) fn skip<T: Copy>(&mut self, len: usize) -> Result<(), GGufReadError<'a>> {
        let len = len * sizeof!(T);
        let data = &self.data[self.cursor..];
        if data.len() >= len {
            self.cursor += len;
            Ok(())
        } else {
            Err(GGufReadError::Eos)
        }
    }

    pub fn read<T: Copy>(&mut self) -> Result<T, GGufReadError<'a>> {
        let ptr = self.data[self.cursor..].as_ptr().cast::<T>();
        self.skip::<T>(1)?;
        Ok(unsafe { ptr.read_unaligned() })
    }

    pub fn read_bool(&mut self) -> Result<bool, GGufReadError<'a>> {
        match self.read::<u8>()? {
            0 => Ok(false),
            1 => Ok(true),
            e => Err(GGufReadError::Bool(e)),
        }
    }

    pub fn read_str(&mut self) -> Result<&'a str, GGufReadError<'a>> {
        let len = self.read::<u64>()? as usize;
        let tail = &self.data[self.cursor..];
        self.skip::<u8>(len)?;
        std::str::from_utf8(&tail[..len]).map_err(GGufReadError::Utf8)
    }

    pub fn read_arr_header(&mut self) -> Result<(GGufMetaDataValueType, usize), GGufReadError<'a>> {
        Ok((self.read()?, self.read::<u64>()? as _))
    }
}
