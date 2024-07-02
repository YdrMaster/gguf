use crate::{
    reader::{GGmlReadError, GGmlReader},
    sizeof,
};
use std::{collections::HashMap, slice::from_raw_parts};

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

pub struct GGufMetaKVPairs<'a> {
    indices: HashMap<&'a str, usize>,
    nbytes: usize,
}

impl<'a> GGufMetaKVPairs<'a> {
    pub fn scan(count: usize, data: &'a [u8]) -> Result<Self, GGmlReadError<'a>> {
        let mut reader = GGmlReader::new(data);
        let mut indices = HashMap::with_capacity(count);
        for _ in 0..count {
            let key = reader.read_str()?;
            let ty = reader.read()?;
            let begin = reader.cursor();
            skip_value(ty, &mut reader)?;
            if indices.insert(key, reader.cursor() - begin).is_some() {
                return Err(GGmlReadError::DuplicatedKey(key));
            }
        }
        Ok(Self {
            indices,
            nbytes: reader.cursor(),
        })
    }

    #[inline]
    pub fn keys<'s>(&'s self) -> impl Iterator<Item = &'a str> + 's {
        self.indices.keys().copied()
    }

    #[inline]
    pub const fn nbytes(&self) -> usize {
        self.nbytes
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
    reader: &mut GGmlReader<'a>,
) -> Result<(), GGmlReadError<'a>> {
    use GGufMetaDataValueType as Ty;
    match ty {
        Ty::U8 => reader.skip::<u8>(1),
        Ty::I8 => reader.skip::<i8>(1),
        Ty::U16 => reader.skip::<u16>(1),
        Ty::I16 => reader.skip::<i16>(1),
        Ty::U32 => reader.skip::<u32>(1),
        Ty::I32 => reader.skip::<i32>(1),
        Ty::F32 => reader.skip::<f32>(1),
        Ty::BOOL => match reader.read::<u8>()? {
            0 | 1 => Ok(()),
            e => Err(GGmlReadError::Bool(e)),
        },
        Ty::STRING => reader.read_str().map(drop),
        Ty::ARRAY => {
            let ty = reader.read()?;
            let len = reader.read::<u64>()?;
            for _ in 0..len {
                skip_value(ty, reader)?;
            }
            Ok(())
        }
        Ty::U64 => reader.skip::<u64>(1),
        Ty::I64 => reader.skip::<i64>(1),
        Ty::F64 => reader.skip::<f64>(1),
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
    pub fn value_reader(&self) -> GGmlReader<'a> {
        GGmlReader::new(unsafe {
            from_raw_parts(
                self.key
                    .as_ptr()
                    .add(self.key.len())
                    .add(sizeof!(GGufMetaDataValueType)),
                self.len,
            )
        })
    }
}
