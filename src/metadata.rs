use std::{collections::HashMap, mem::size_of, slice::from_raw_parts, str::Utf8Error};

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

pub struct MetaDataPairs<'a> {
    indices: HashMap<&'a str, usize>,
    len: usize,
}

impl<'a> MetaDataPairs<'a> {
    pub fn scan(count: usize, data: &'a [u8]) -> Result<Self, MetaDataError<'a>> {
        let mut reader = MetaReader { data, cursor: 0 };
        let mut indices = HashMap::with_capacity(count);
        for _ in 0..count {
            let (key, ty) = reader.read_kv_header()?;
            let begin = reader.cursor;
            skip_value(ty, &mut reader)?;
            if indices.insert(key, reader.cursor - begin).is_some() {
                return Err(MetaDataError::DuplicatedKey(key));
            }
        }
        Ok(Self {
            indices,
            len: reader.cursor,
        })
    }

    #[inline]
    pub fn keys<'s>(&'s self) -> impl Iterator<Item = &'a str> + 's {
        self.indices.keys().copied()
    }

    #[inline]
    pub const fn nbytes(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<MetaDataKV<'a>> {
        self.indices
            .get_key_value(key)
            .map(|(&key, &len)| MetaDataKV { key, len })
    }
}

fn skip_value<'a>(
    ty: GGufMetaDataValueType,
    reader: &mut MetaReader<'a>,
) -> Result<(), MetaDataError<'a>> {
    use GGufMetaDataValueType as Ty;
    match ty {
        Ty::U8 => reader.read::<u8>().map(drop),
        Ty::I8 => reader.read::<i8>().map(drop),
        Ty::U16 => reader.read::<u16>().map(drop),
        Ty::I16 => reader.read::<i16>().map(drop),
        Ty::U32 => reader.read::<u32>().map(drop),
        Ty::I32 => reader.read::<i32>().map(drop),
        Ty::F32 => reader.read::<f32>().map(drop),
        Ty::BOOL => match reader.read::<u8>()? {
            0 | 1 => Ok(()),
            e => Err(MetaDataError::Bool(e)),
        },
        Ty::STRING => reader.read_str().map(drop),
        Ty::ARRAY => {
            let (dt, len) = reader.read_arr_header()?;
            for _ in 0..len {
                skip_value(dt, reader)?;
            }
            Ok(())
        }
        Ty::U64 => reader.read::<u64>().map(drop),
        Ty::I64 => reader.read::<i64>().map(drop),
        Ty::F64 => reader.read::<f64>().map(drop),
    }
}

pub struct MetaDataKV<'a> {
    key: &'a str,
    len: usize,
}

impl<'a> MetaDataKV<'a> {
    #[inline]
    pub fn key(&self) -> &'a str {
        self.key
    }

    #[inline]
    pub fn ty(&self) -> GGufMetaDataValueType {
        unsafe {
            self.key
                .as_ptr()
                .add(self.key.len())
                .cast::<GGufMetaDataValueType>()
                .read_unaligned()
        }
    }

    #[inline]
    pub fn value_reader(&self) -> MetaReader<'a> {
        MetaReader {
            data: unsafe {
                from_raw_parts(
                    self.key
                        .as_ptr()
                        .add(self.key.len())
                        .add(size_of::<GGufMetaDataValueType>()),
                    self.len,
                )
            },
            cursor: 0,
        }
    }
}

pub struct MetaReader<'a> {
    data: &'a [u8],
    cursor: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetaDataError<'a> {
    Eos,
    DuplicatedKey(&'a str),
    Utf8(Utf8Error),
    Bool(u8),
}

impl<'a> MetaReader<'a> {
    pub fn read<U: Copy>(&mut self) -> Result<U, MetaDataError<'a>> {
        let len = size_of::<U>();
        let data = &self.data[self.cursor..];
        if data.len() >= len {
            self.cursor += len;
            let ptr = data.as_ptr().cast::<U>();
            Ok(unsafe { ptr.read_unaligned() })
        } else {
            Err(MetaDataError::Eos)
        }
    }

    pub fn read_str(&mut self) -> Result<&'a str, MetaDataError<'a>> {
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
    fn read_kv_header(&mut self) -> Result<(&'a str, GGufMetaDataValueType), MetaDataError<'a>> {
        let id = self.read_str()?;
        let ty = self.read()?;
        Ok((id, ty))
    }

    #[inline]
    pub fn read_arr_header(&mut self) -> Result<(GGufMetaDataValueType, usize), MetaDataError<'a>> {
        let ty = self.read()?;
        let len = self.read()?;
        Ok((ty, len))
    }
}
