use crate::GGmlType;

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
