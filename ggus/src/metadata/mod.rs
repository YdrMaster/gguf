//! See <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-key-value-pairs>.

mod general;
mod llm;
mod tokenizer;

use crate::{
    reader::{GGufReadError, GGufReader},
    sizeof,
};
use std::{collections::HashMap, slice::from_raw_parts};

pub use tokenizer::{utok, GGufArray};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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
    Bool = 7,
    /// The value is a UTF-8 non-null-terminated string, with length prepended.
    String = 8,
    /// The value is an array of other values, with the length and type prepended.
    ///
    /// Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array = 9,
    /// The value is a 64-bit unsigned little-endian integer.
    U64 = 10,
    /// The value is a 64-bit signed little-endian integer.
    I64 = 11,
    /// The value is a 64-bit IEEE754 floating point number.
    F64 = 12,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u32)]
pub enum GGufFileType {
    AllF32 = 0,
    MostlyF16 = 1,
    MostlyQ4_0 = 2,
    MostlyQ4_1 = 3,
    MostlyQ4_1SomeF16 = 4,
    #[deprecated = "support removed"]
    MostlyQ4_2 = 5,
    #[deprecated = "support removed"]
    MostlyQ4_3 = 6,
    MostlyQ8_0 = 7,
    MostlyQ5_0 = 8,
    MostlyQ51 = 9,
    MostlyQ2K = 10,
    MostlyQ3KS = 11,
    MostlyQ3KM = 12,
    MostlyQ3KL = 13,
    MostlyQ4KS = 14,
    MostlyQ4KM = 15,
    MostlyQ5KS = 16,
    MostlyQ5KM = 17,
    MostlyQ6K = 18,
}

#[derive(Clone, Debug)]
pub struct GGufMetaKVPairs<'a> {
    indices: HashMap<&'a str, usize>,
    nbytes: usize,
}

impl<'a> GGufMetaKVPairs<'a> {
    pub fn scan(count: u64, data: &'a [u8]) -> Result<Self, GGufReadError<'a>> {
        let mut reader = GGufReader::new(data);
        let mut indices = HashMap::with_capacity(count as _);
        for _ in 0..count {
            let key = reader.read_str()?;
            let ty = reader.read()?;
            let begin = reader.cursor();
            skip_value(ty, &mut reader, 1)?;
            if indices.insert(key, reader.cursor() - begin).is_some() {
                return Err(GGufReadError::DuplicatedKey(key));
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
    pub fn kvs<'s>(&'s self) -> impl Iterator<Item = GGufMetaKV<'a>> + 's {
        self.indices
            .iter()
            .map(|(&key, &len)| GGufMetaKV { key, len })
    }

    #[inline]
    pub const fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn get(&self, key: impl AsRef<str>) -> Option<GGufMetaKV<'a>> {
        self.indices
            .get_key_value(key.as_ref())
            .map(|(&key, &len)| GGufMetaKV { key, len })
    }

    fn get_typed(
        &self,
        name: impl AsRef<str>,
        ty: GGufMetaDataValueType,
    ) -> Option<GGufReader<'a>> {
        self.get(name).map(|kv| {
            assert_eq!(kv.ty(), ty);
            kv.value_reader()
        })
    }
}

fn skip_value<'a>(
    ty: GGufMetaDataValueType,
    reader: &mut GGufReader<'a>,
    len: usize,
) -> Result<(), GGufReadError<'a>> {
    use GGufMetaDataValueType as Ty;
    match ty {
        Ty::U8 => reader.skip::<u8>(len),
        Ty::I8 => reader.skip::<i8>(len),
        Ty::U16 => reader.skip::<u16>(len),
        Ty::I16 => reader.skip::<i16>(len),
        Ty::U32 => reader.skip::<u32>(len),
        Ty::I32 => reader.skip::<i32>(len),
        Ty::F32 => reader.skip::<f32>(len),
        Ty::U64 => reader.skip::<u64>(len),
        Ty::I64 => reader.skip::<i64>(len),
        Ty::F64 => reader.skip::<f64>(len),

        Ty::Bool => {
            for _ in 0..len {
                reader.read_bool()?;
            }
            Ok(())
        }
        Ty::String => {
            for _ in 0..len {
                reader.read_str()?;
            }
            Ok(())
        }
        Ty::Array => {
            let (ty, len) = reader.read_arr_header()?;
            skip_value(ty, reader, len)
        }
    }
}

pub struct GGufMetaKV<'a> {
    key: &'a str,
    len: usize,
}

impl<'a> GGufMetaKV<'a> {
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
    pub fn value_reader(&self) -> GGufReader<'a> {
        GGufReader::new(unsafe {
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
