//! See <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-key-value-pairs>.

mod meta_kv;
pub(crate) mod standard;

pub use meta_kv::GGufMetaKV;

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
    MostlyIQ2XXS = 19,
    MostlyIQ2XS = 20,
    MostlyQ2KS = 21,
    MostlyIQ3XS = 22,
    MostlyIQ3XXS = 23,
    MostlyIQ1S = 24,
    MostlyIQ4NL = 25,
    MostlyIQ3S = 26,
    MostlyIQ3M = 27,
    MostlyIQ2S = 28,
    MostlyIQ2M = 29,
    MostlyIQ4XS = 30,
    MostlyIQ1M = 31,
    MostlyBF16 = 32,
    MostlyQ4_0_4_4 = 33,
    MostlyQ4_0_4_8 = 34,
    MostlyQ4_0_8_8 = 35,
    // GUESSED = 1024  # not specified in the model file
}

impl GGufFileType {
    #[rustfmt::skip]
    pub fn to_ggml_type(self) -> crate::GGmlType {
        type Ty = crate::GGmlType;
        match self {
            Self::AllF32         => Ty::F32,
            Self::MostlyF16      => Ty::F16,
            Self::MostlyQ4_0     => Ty::Q4_0,
            Self::MostlyQ4_1     => Ty::Q4_1,
            #[allow(deprecated)]
            Self::MostlyQ4_2     => Ty::Q4_2,
            #[allow(deprecated)]
            Self::MostlyQ4_3     => Ty::Q4_3,
            Self::MostlyQ8_0     => Ty::Q8_0,
            Self::MostlyQ5_0     => Ty::Q5_0,
            Self::MostlyQ51      => Ty::Q5_1,
            Self::MostlyQ2K      => Ty::Q2K,
            Self::MostlyQ6K      => Ty::Q6K,
            Self::MostlyIQ2XXS   => Ty::IQ2XXS,
            Self::MostlyIQ2XS    => Ty::IQ2XS,
            Self::MostlyIQ3XXS   => Ty::IQ3XXS,
            Self::MostlyIQ1S     => Ty::IQ1S,
            Self::MostlyIQ4NL    => Ty::IQ4NL,
            Self::MostlyIQ3S     => Ty::IQ3S,
            Self::MostlyIQ2S     => Ty::IQ2S,
            Self::MostlyIQ4XS    => Ty::IQ4XS,
            Self::MostlyIQ1M     => Ty::IQ1M,
            Self::MostlyBF16     => Ty::BF16,
            Self::MostlyQ4_0_4_4 => Ty::Q4_0_4_4,
            Self::MostlyQ4_0_4_8 => Ty::Q4_0_4_8,
            Self::MostlyQ4_0_8_8 => Ty::Q4_0_8_8,

            Self::MostlyQ4_1SomeF16 |
            Self::MostlyQ3KS        |
            Self::MostlyQ3KM        |
            Self::MostlyQ3KL        |
            Self::MostlyQ4KS        |
            Self::MostlyQ4KM        |
            Self::MostlyQ5KS        |
            Self::MostlyQ5KM        |
            Self::MostlyQ2KS        |
            Self::MostlyIQ3XS       |
            Self::MostlyIQ3M        |
            Self::MostlyIQ2M => todo!(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(i32)]
pub enum GGmlTokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    User = 4,
    Unused = 5,
    Byte = 6,
}
