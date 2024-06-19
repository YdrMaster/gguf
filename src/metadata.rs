#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum GGufMetaDataValueType {
    /// The value is a 8-bit unsigned integer.
    U8 = 0,
    /// The value is a 8-bit signed integer.
    I8 = 1,
    /// The value is a 16-bit unsigned little-endian integer.
    U16 = 2,
    /// The value is a 16-bit signed little-endian integer.
    I16 = 3,
    /// The value is a 32-bit unsigned little-endian integer.
    U32 = 4,
    /// The value is a 32-bit signed little-endian integer.
    I32 = 5,
    /// The value is a 32-bit IEEE754 floating point number.
    F32 = 6,
    /// The value is a boolean.
    ///
    /// 1-byte value where 0 is false and 1 is true.
    /// Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    BOOL = 7,
    /// The value is a UTF-8 non-null-terminated string, with length prepended.
    STRING = 8,
    /// The value is an array of other values, with the length and type prepended.
    ///
    /// Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    ARRAY = 9,
    /// The value is a 64-bit unsigned little-endian integer.
    U64 = 10,
    /// The value is a 64-bit signed little-endian integer.
    I64 = 11,
    /// The value is a 64-bit IEEE754 floating point number.
    F64 = 12,
}

pub struct MetaReader<'a> {
    data: &'a [u8],
    cursor: usize,
}

impl<'a> MetaReader<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, cursor: 0 }
    }

    pub fn read<U: Copy>(&mut self) -> Result<U, ()> {
        let len = std::mem::size_of::<U>();
        let data = &self.data[self.cursor..];
        if data.len() >= len {
            self.cursor += len;
            let ptr = data.as_ptr().cast::<U>();
            Ok(unsafe { ptr.read_unaligned() })
        } else {
            Err(())
        }
    }

    pub fn read_str(&mut self) -> Result<&'a str, ()> {
        let len = self.read::<u64>()? as usize;
        let data = &self.data[self.cursor..];
        if data.len() >= len {
            self.cursor += len;
            std::str::from_utf8(&data[..len]).map_err(|_| ())
        } else {
            Err(())
        }
    }

    #[inline]
    pub fn read_kv_header(&mut self) -> Result<(&'a str, GGufMetaDataValueType), ()> {
        let id = self.read_str()?;
        let ty = self.read()?;
        Ok((id, ty))
    }

    #[inline]
    pub fn read_arr_header(&mut self) -> Result<(GGufMetaDataValueType, usize), ()> {
        let ty = self.read()?;
        let len = self.read()?;
        Ok((ty, len))
    }
}
