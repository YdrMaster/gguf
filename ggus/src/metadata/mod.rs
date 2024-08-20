//! See <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-key-value-pairs>.

mod meta_kv;

use crate::GGmlType;

pub use meta_kv::GGufMetaKV;

pub const GENERAL_ALIGNMENT: &str = "general.alignment";
pub const DEFAULT_ALIGNMENT: usize = 32;

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

impl GGufMetaDataValueType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "str",
            Self::Array => "arr",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::U64 => "u64",
        }
    }
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

impl GGufFileType {
    pub fn to_ggml_type(self) -> GGmlType {
        match self {
            Self::AllF32 => GGmlType::F32,
            Self::MostlyF16 => GGmlType::F16,
            Self::MostlyQ4_0 => GGmlType::Q4_0,
            Self::MostlyQ4_1 => GGmlType::Q4_1,
            #[allow(deprecated)]
            Self::MostlyQ4_2 => GGmlType::Q4_2,
            #[allow(deprecated)]
            Self::MostlyQ4_3 => GGmlType::Q4_3,
            Self::MostlyQ8_0 => GGmlType::Q8_0,
            Self::MostlyQ5_0 => GGmlType::Q5_0,
            Self::MostlyQ51 => GGmlType::Q5_1,
            Self::MostlyQ2K => GGmlType::Q2K,
            Self::MostlyQ6K => GGmlType::Q6K,
            _ => unimplemented!(),
        }
    }
}
