use std::{borrow::Borrow, ops::Deref, str::Utf8Error};

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

#[inline]
pub fn scan<'a>(
    count: usize,
    data: &'a [u8],
) -> Result<(Vec<MetaDataKV<'a>>, usize), MetaDataError> {
    let mut reader = MetaReader::new(data);
    let mut ans = Vec::with_capacity(count);
    for _ in 0..count {
        let (key, ty) = reader.read_kv_header()?;
        skip_value(ty, &mut reader)?;
        ans.push(MetaDataKV(key));
    }
    Ok((ans, reader.cursor))
}

fn skip_value(ty: GGufMetaDataValueType, reader: &mut MetaReader) -> Result<(), MetaDataError> {
    use GGufMetaDataValueType as Ty;
    match ty {
        Ty::U8 => reader.read::<u8>().map(|_| ()),
        Ty::I8 => reader.read::<i8>().map(|_| ()),
        Ty::U16 => reader.read::<u16>().map(|_| ()),
        Ty::I16 => reader.read::<i16>().map(|_| ()),
        Ty::U32 => reader.read::<u32>().map(|_| ()),
        Ty::I32 => reader.read::<i32>().map(|_| ()),
        Ty::F32 => reader.read::<f32>().map(|_| ()),
        Ty::BOOL => match reader.read::<u8>()? {
            0 | 1 => Ok(()),
            e => Err(MetaDataError::Bool(e)),
        },
        Ty::STRING => reader.read_str().map(|_| ()),
        Ty::ARRAY => {
            let (dt, len) = reader.read_arr_header()?;
            for _ in 0..len {
                skip_value(dt, reader)?;
            }
            Ok(())
        }
        Ty::U64 => reader.read::<u64>().map(|_| ()),
        Ty::I64 => reader.read::<i64>().map(|_| ()),
        Ty::F64 => reader.read::<f64>().map(|_| ()),
    }
}

#[repr(transparent)]
pub struct MetaDataKV<'a>(&'a str);

impl<'a> MetaDataKV<'a> {
    #[inline]
    pub fn key(&self) -> &'a str {
        self.0
    }

    #[inline]
    pub fn ty(&self) -> GGufMetaDataValueType {
        unsafe {
            self.0
                .as_ptr()
                .add(self.0.len())
                .cast::<GGufMetaDataValueType>()
                .read_unaligned()
        }
    }
}

impl AsRef<str> for MetaDataKV<'_> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl Borrow<str> for MetaDataKV<'_> {
    #[inline]
    fn borrow(&self) -> &str {
        self.0
    }
}

impl Deref for MetaDataKV<'_> {
    type Target = str;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

pub struct MetaReader<'a> {
    data: &'a [u8],
    cursor: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetaDataError {
    Eos,
    Utf8(Utf8Error),
    Bool(u8),
}

impl<'a> MetaReader<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, cursor: 0 }
    }

    pub fn read<U: Copy>(&mut self) -> Result<U, MetaDataError> {
        let len = std::mem::size_of::<U>();
        let data = &self.data[self.cursor..];
        if data.len() >= len {
            self.cursor += len;
            let ptr = data.as_ptr().cast::<U>();
            Ok(unsafe { ptr.read_unaligned() })
        } else {
            Err(MetaDataError::Eos)
        }
    }

    pub fn read_str(&mut self) -> Result<&'a str, MetaDataError> {
        let len = self.read::<u64>()? as usize;
        let data = &self.data[self.cursor..];
        if data.len() >= len {
            self.cursor += len;
            std::str::from_utf8(&data[..len]).map_err(MetaDataError::Utf8)
        } else {
            Err(MetaDataError::Eos)
        }
    }

    #[inline]
    pub fn read_kv_header(&mut self) -> Result<(&'a str, GGufMetaDataValueType), MetaDataError> {
        let id = self.read_str()?;
        let ty = self.read()?;
        Ok((id, ty))
    }

    #[inline]
    pub fn read_arr_header(&mut self) -> Result<(GGufMetaDataValueType, usize), MetaDataError> {
        let ty = self.read()?;
        let len = self.read()?;
        Ok((ty, len))
    }
}
