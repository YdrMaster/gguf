use crate::{
    reader::{GGmlReadError, GGmlReader},
    sizeof,
};
use std::{collections::HashMap, slice::from_raw_parts};

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

    pub fn get(&self, key: impl AsRef<str>) -> Option<MetaDataKV<'a>> {
        self.indices
            .get_key_value(key.as_ref())
            .map(|(&key, &len)| MetaDataKV { key, len })
    }

    pub fn architecture(&self) -> &'a str {
        self.get_typed("general.architecture", GGufMetaDataValueType::STRING)
            .expect("required key `general.architecture` not exist")
            .read_str()
            .unwrap()
    }

    pub fn quantization_version(&self) -> u32 {
        self.get_typed("general.quantization_version", GGufMetaDataValueType::U32)
            .expect("required key `general.quantization_version` not exist")
            .read()
            .unwrap()
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.get_typed("general.alignment", GGufMetaDataValueType::U32)
            .map_or(32, |mut reader| reader.read::<u32>().unwrap() as _)
    }

    #[inline]
    pub fn name(&self) -> Option<&'a str> {
        self.get_typed("general.name", GGufMetaDataValueType::STRING)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn author(&self) -> Option<&'a str> {
        self.get_typed("general.author", GGufMetaDataValueType::STRING)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn url(&self) -> Option<&'a str> {
        self.get_typed("general.url", GGufMetaDataValueType::STRING)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn description(&self) -> Option<&'a str> {
        self.get_typed("general.description", GGufMetaDataValueType::STRING)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn license(&self) -> Option<&'a str> {
        self.get_typed("general.license", GGufMetaDataValueType::STRING)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn file_type(&self) -> Option<GGufFileType> {
        self.get_typed("general.license", GGufMetaDataValueType::U32)
            .map(|mut reader| {
                let val = reader.read::<u32>().unwrap();
                assert!(val <= GGufFileType::MostlyQ6K as _);
                unsafe { std::mem::transmute(val) }
            })
    }

    #[inline]
    pub fn source_url(&self) -> Option<&'a str> {
        self.get_typed("general.source.url", GGufMetaDataValueType::STRING)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn source_hf_repo(&self) -> Option<&'a str> {
        self.get_typed(
            "general.source.huggingface.repository",
            GGufMetaDataValueType::STRING,
        )
        .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    fn get_typed(&self, name: &str, ty: GGufMetaDataValueType) -> Option<GGmlReader<'a>> {
        self.get(name).map(|kv| {
            assert_eq!(kv.ty(), ty);
            kv.value_reader()
        })
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
