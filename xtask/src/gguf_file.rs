use ggus::{GGufFileHeader, GGufMetaKVPairs, GGufReadError, GGufTensors};
use indexmap::IndexMap;

#[derive(Clone)]
pub(crate) struct GGufFile<'a> {
    header: GGufFileHeader,
    meta_kvs: GGufMetaKVPairs<'a>,
    tensors: GGufTensors<'a>,
    data: &'a [u8],
}

#[derive(Debug)]
pub(crate) enum GGufError<'a> {
    MagicMismatch,
    EndianNotSupport,
    VersionNotSupport,
    #[allow(dead_code)]
    Reading(GGufReadError<'a>),
    FileSizeError,
    SplitModeRepeated,
}

impl<'a> GGufFile<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Result<Self, GGufError<'a>> {
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

        let align = meta_kvs.alignment();
        let cursor = (cursor + tensors.nbytes() + align - 1) / align * align;
        Ok(Self {
            header,
            meta_kvs,
            tensors,
            data: &data[cursor..],
        })
    }

    pub fn header(&self) -> &GGufFileHeader {
        &self.header
    }

    pub fn meta_kvs(&self) -> &GGufMetaKVPairs<'a> {
        &self.meta_kvs
    }

    pub fn tensors_as_indexmap(&self) -> IndexMap<ggus::GGufTensorInfo, &[u8]> {
        self.tensors
            .iter()
            .map(move |t| (t, self.data))
            .collect::<IndexMap<_, _>>()
    }
}

#[inline(always)]
pub(crate) const fn pad(pos: usize, align: usize) -> usize {
    (align - pos % align) % align
}
