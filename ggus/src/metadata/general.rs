use crate::{GGufFileType, GGufMetaDataValueType as Ty, GGufMetaKVPairs};

pub const GENERAL_ALIGNMENT: &str = "general.alignment";
pub const DEFAULT_ALIGNMENT: usize = 32;

impl<'a> GGufMetaKVPairs<'a> {
    pub fn architecture(&self) -> Option<&'a str> {
        self.get_typed("general.architecture", Ty::String)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn quantization_version(&self) -> Option<u32> {
        self.get_typed("general.quantization_version", Ty::U32)
            .map(|mut reader| reader.read().unwrap())
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.get_typed(GENERAL_ALIGNMENT, Ty::U32)
            .map_or(DEFAULT_ALIGNMENT, |mut reader| {
                reader.read::<u32>().unwrap() as _
            })
    }

    #[inline]
    pub fn name(&self) -> Option<&'a str> {
        self.get_typed("general.name", Ty::String)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn author(&self) -> Option<&'a str> {
        self.get_typed("general.author", Ty::String)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn url(&self) -> Option<&'a str> {
        self.get_typed("general.url", Ty::String)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn description(&self) -> Option<&'a str> {
        self.get_typed("general.description", Ty::String)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn license(&self) -> Option<&'a str> {
        self.get_typed("general.license", Ty::String)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn file_type(&self) -> Option<GGufFileType> {
        self.get_typed("general.license", Ty::U32)
            .map(|mut reader| {
                let val = reader.read::<u32>().unwrap();
                assert!(val <= GGufFileType::MostlyQ6K as _);
                unsafe { std::mem::transmute(val) }
            })
    }

    #[inline]
    pub fn source_url(&self) -> Option<&'a str> {
        self.get_typed("general.source.url", Ty::String)
            .map(|mut reader| reader.read_str().unwrap())
    }

    #[inline]
    pub fn source_hf_repo(&self) -> Option<&'a str> {
        self.get_typed("general.source.huggingface.repository", Ty::String)
            .map(|mut reader| reader.read_str().unwrap())
    }
}
