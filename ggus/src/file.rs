use crate::{pad, GGufFileHeader, GGufMetaKVPairs, GGufReadError, GGufTensors};
use std::{error::Error, fmt};

#[derive(Clone)]
pub struct GGuf<'a> {
    pub header: GGufFileHeader,
    pub meta_kvs: GGufMetaKVPairs<'a>,
    pub tensors: GGufTensors<'a>,
    pub data: &'a [u8],
}

#[derive(Debug)]
pub enum GGufError {
    MagicMismatch,
    EndianNotSupport,
    VersionNotSupport,
    Reading(GGufReadError),
}

impl fmt::Display for GGufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MagicMismatch => f.write_str("magic mismatch"),
            Self::EndianNotSupport => f.write_str("endian not support"),
            Self::VersionNotSupport => f.write_str("version not support"),
            Self::Reading(e) => write!(f, "reading error: {e:?}"),
        }
    }
}

impl Error for GGufError {}

impl<'a> GGuf<'a> {
    pub fn scan(data: &'a [u8]) -> Result<Self, GGufError> {
        let header = unsafe { data.as_ptr().cast::<GGufFileHeader>().read() };
        if !header.is_magic_correct() {
            return Err(GGufError::MagicMismatch);
        }
        if !header.is_native_endian() {
            return Err(GGufError::EndianNotSupport);
        }
        if header.version != 3 {
            return Err(GGufError::VersionNotSupport);
        }

        let cursor = header.nbytes();
        let meta_kvs = GGufMetaKVPairs::scan(header.metadata_kv_count, &data[cursor..])
            .map_err(GGufError::Reading)?;

        let cursor = cursor + meta_kvs.nbytes();
        let tensors =
            GGufTensors::scan(header.tensor_count, &data[cursor..]).map_err(GGufError::Reading)?;

        let cursor = cursor + tensors.nbytes();
        let padding = if tensors.is_empty() {
            0
        } else {
            pad(cursor, meta_kvs.alignment())
        };
        Ok(Self {
            header,
            meta_kvs,
            tensors,
            data: &data[cursor + padding..],
        })
    }
}
