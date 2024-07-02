use super::{GGufMetaDataValueType, GGufMetaKVPairs};
use crate::reader::GGmlReader;
use std::marker::PhantomData;

#[repr(transparent)]
pub struct TokenizerMeta<'a>(GGufMetaKVPairs<'a>);

#[allow(non_camel_case_types)]
pub type utok = u32;

impl<'a> GGufMetaKVPairs<'a> {
    #[inline]
    pub const fn tokenizer(self) -> TokenizerMeta<'a> {
        TokenizerMeta(self)
    }
}

impl<'a> TokenizerMeta<'a> {
    #[inline]
    pub fn all(self) -> GGufMetaKVPairs<'a> {
        self.0
    }
}

macro_rules! get {
    ($self:expr, $key:literal @ $ty:ident) => {
        $self.0.get_typed(concat!("tokenizer.ggml.", $key), GGufMetaDataValueType::$ty)
    };

    ($self:expr, $key:literal(u32)) => {
        get!($self, $key @ U32).map(|mut reader| reader.read().unwrap())
    };

    ($self:expr, $key:literal(str)) => {
        get!($self, $key @ String).map(|mut reader| reader.read_str().unwrap())
    };
}

pub struct GGufArray<'a, T: ?Sized>(GGmlReader<'a>, u64, PhantomData<T>);

impl<'a, T: ?Sized> GGufArray<'a, T> {
    pub fn new_typed(mut reader: GGmlReader<'a>, ty: GGufMetaDataValueType) -> Self {
        assert_eq!(reader.read::<GGufMetaDataValueType>(), Ok(ty));
        let len = reader.read().unwrap();
        Self(reader, len, PhantomData)
    }
}

impl<'a> GGufArray<'a, i32> {
    #[inline]
    pub fn new(reader: GGmlReader<'a>) -> Self {
        Self::new_typed(reader, GGufMetaDataValueType::I32)
    }
}

impl<'a> GGufArray<'a, f32> {
    #[inline]
    pub fn new(reader: GGmlReader<'a>) -> Self {
        Self::new_typed(reader, GGufMetaDataValueType::F32)
    }
}

impl<'a> GGufArray<'a, str> {
    #[inline]
    pub fn new(reader: GGmlReader<'a>) -> Self {
        Self::new_typed(reader, GGufMetaDataValueType::String)
    }
}

impl<'a, T: Copy> Iterator for GGufArray<'a, T> {
    type Item = T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.1 > 0 {
            self.1 -= 1;
            Some(self.0.read::<T>().unwrap())
        } else {
            None
        }
    }
}

impl<'a> Iterator for GGufArray<'a, str> {
    type Item = &'a str;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.1 > 0 {
            self.1 -= 1;
            Some(self.0.read_str().unwrap())
        } else {
            None
        }
    }
}

impl<'a> TokenizerMeta<'a> {
    #[inline]
    pub fn model(&self) -> Option<&'a str> {
        get!(self, "model"(str))
    }

    #[inline]
    pub fn tokens(&self) -> Option<GGufArray<'a, str>> {
        get!(self, "tokens" @ Array).map(GGufArray::<str>::new)
    }

    #[inline]
    pub fn scores(&self) -> Option<GGufArray<'a, f32>> {
        get!(self, "scores" @ Array).map(GGufArray::<f32>::new)
    }

    #[inline]
    pub fn token_type(&self) -> Option<GGufArray<'a, i32>> {
        get!(self, "token_type" @ Array).map(GGufArray::<i32>::new)
    }

    #[inline]
    pub fn merges(&self) -> Option<GGufArray<'a, str>> {
        get!(self, "merges" @ Array).map(GGufArray::<str>::new)
    }

    #[inline]
    pub fn added_tokens(&self) -> Option<GGufArray<'a, str>> {
        get!(self, "added_tokens" @ Array).map(GGufArray::<str>::new)
    }

    #[inline]
    pub fn bos(&self) -> Option<utok> {
        get!(self, "bos_token_id"(u32))
    }

    #[inline]
    pub fn eos(&self) -> Option<utok> {
        get!(self, "eos_token_id"(u32))
    }

    #[inline]
    pub fn unknown(&self) -> Option<utok> {
        get!(self, "unknown_token_id"(u32))
    }

    #[inline]
    pub fn separator(&self) -> Option<utok> {
        get!(self, "separator_token_id"(u32))
    }

    #[inline]
    pub fn padding(&self) -> Option<utok> {
        get!(self, "padding_token_id"(u32))
    }
}

impl<'a> TokenizerMeta<'a> {
    #[inline]
    pub fn hf_json(&self) -> Option<&'a str> {
        get!(self, "huggingface.json"(str))
    }
}

impl<'a> TokenizerMeta<'a> {
    #[inline]
    pub fn rwkv_world(&self) -> Option<&'a str> {
        get!(self, "rwkv.world"(str))
    }

    #[inline]
    pub fn chat_template(&self) -> Option<&'a str> {
        get!(self, "chat_template"(str))
    }
}
