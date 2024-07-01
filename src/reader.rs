use crate::sizeof;

pub struct GGmlReader<'a> {
    data: &'a [u8],
    cursor: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GGmlReadError<'a> {
    Eos,
    DuplicatedKey(&'a str),
    Utf8(std::str::Utf8Error),
    Bool(u8),
}

impl<'a> GGmlReader<'a> {
    #[inline]
    pub(crate) const fn new(data: &'a [u8]) -> Self {
        Self { data, cursor: 0 }
    }

    #[inline]
    pub(crate) const fn cursor(&self) -> usize {
        self.cursor
    }

    pub(crate) fn skip<U: Copy>(&mut self, len: usize) -> Result<(), GGmlReadError<'a>> {
        let len = len * sizeof!(U);
        let data = &self.data[self.cursor..];
        if data.len() >= len {
            self.cursor += len;
            Ok(())
        } else {
            Err(GGmlReadError::Eos)
        }
    }

    pub fn read<U: Copy>(&mut self) -> Result<U, GGmlReadError<'a>> {
        let ptr = self.data[self.cursor..].as_ptr().cast::<U>();
        self.skip::<U>(1)?;
        Ok(unsafe { ptr.read_unaligned() })
    }

    pub fn read_str(&mut self) -> Result<&'a str, GGmlReadError<'a>> {
        let len = self.read::<u64>()? as usize;
        let tail = &self.data[self.cursor..];
        self.skip::<u8>(len)?;
        std::str::from_utf8(&tail[..len]).map_err(GGmlReadError::Utf8)
    }
}
