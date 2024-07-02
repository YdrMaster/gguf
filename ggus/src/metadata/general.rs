use super::{GGufFileType, GGufMetaDataValueType as Ty, GGufMetaKVPairs};

impl<'a> GGufMetaKVPairs<'a> {
    pub fn architecture(&self) -> &'a str {
        self.get_typed("general.architecture", Ty::String)
            .expect("required key `general.architecture` not exist")
            .read_str()
            .unwrap()
    }

    pub fn quantization_version(&self) -> u32 {
        self.get_typed("general.quantization_version", Ty::U32)
            .expect("required key `general.quantization_version` not exist")
            .read()
            .unwrap()
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.get_typed("general.alignment", Ty::U32)
            .map_or(32, |mut reader| reader.read::<u32>().unwrap() as _)
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
