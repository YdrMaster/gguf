use crate::{
    pad, GGufFileHeader, GGufMetaKV, GGufReadError, GGufReader, GGufTensorMeta, DEFAULT_ALIGNMENT,
    GENERAL_ALIGNMENT,
};
use indexmap::IndexMap;
use std::{error::Error, fmt};

pub struct GGuf<'a> {
    pub header: GGufFileHeader,
    pub alignment: usize,
    pub meta_kvs: IndexMap<&'a str, GGufMetaKV<'a>>,
    pub tensors: IndexMap<&'a str, GGufTensorMeta<'a>>,
    pub data: &'a [u8],
}

#[derive(Debug)]
pub enum GGufError {
    Reading(GGufReadError),
    MagicMismatch,
    EndianNotSupport,
    VersionNotSupport,
    DuplicateMetaKey(String),
    DuplicateTensorName(String),
}

impl fmt::Display for GGufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reading(e) => write!(f, "reading error: {e:?}"),
            Self::MagicMismatch => f.write_str("magic mismatch"),
            Self::EndianNotSupport => f.write_str("endian not support"),
            Self::VersionNotSupport => f.write_str("version not support"),
            Self::DuplicateMetaKey(key) => write!(f, "duplicate meta key: {key}"),
            Self::DuplicateTensorName(name) => write!(f, "duplicate tensor name: {name}"),
        }
    }
}

impl Error for GGufError {}

impl<'a> GGuf<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, GGufError> {
        use GGufError::*;

        let mut reader = GGufReader::new(data);

        let header = reader.read_header().map_err(Reading)?;
        if !header.is_magic_correct() {
            return Err(MagicMismatch);
        }
        if !header.is_native_endian() {
            return Err(EndianNotSupport);
        }
        if header.version != 3 {
            return Err(VersionNotSupport);
        }

        let mut alignment = DEFAULT_ALIGNMENT;
        let mut meta_kvs = IndexMap::with_capacity(header.metadata_kv_count as _);
        for _ in 0..header.metadata_kv_count {
            let kv = reader.read_meta_kv().map_err(Reading)?;
            let k = kv.key();
            if k == GENERAL_ALIGNMENT {
                alignment = kv.value_reader().read::<u32>().map_err(Reading)? as _;
            }
            if meta_kvs.insert(k, kv).is_some() {
                return Err(DuplicateMetaKey(k.into()));
            }
        }

        let mut data_len = 0;
        let mut tensors = IndexMap::with_capacity(header.tensor_count as _);
        for _ in 0..header.tensor_count {
            let tensor = reader.read_tensor_meta().map_err(Reading)?;
            let name = tensor.name();
            let info = tensor.to_info();
            let end = info.offset() + info.nbytes();
            if end > data_len {
                data_len = end;
            }
            if tensors.insert(name, tensor).is_some() {
                return Err(DuplicateTensorName(name.into()));
            }
        }

        let cursor = data.len() - reader.remaining().len();
        let padding = if tensors.is_empty() {
            0
        } else {
            pad(cursor, alignment)
        };
        reader.skip::<u8>(padding).map_err(Reading)?;
        let data = reader.remaining();
        if data.len() != data_len {
            return Err(Reading(GGufReadError::Eos));
        }

        Ok(Self {
            header,
            alignment,
            meta_kvs,
            tensors,
            data,
        })
    }
}
