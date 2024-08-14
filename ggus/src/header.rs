use crate::{GGufReadError, GGufReader};
use std::str::{from_utf8, Utf8Error};

#[derive(Clone, Default, Debug)]
#[repr(C)]
pub struct GGufFileHeader {
    magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

const MAGIC: [u8; 4] = *b"GGUF";

impl GGufReader<'_> {
    #[inline]
    pub fn read_header(&mut self) -> Result<GGufFileHeader, GGufReadError> {
        let ptr = self.remaining().as_ptr().cast::<GGufFileHeader>();
        self.skip::<GGufFileHeader>(1)?;
        Ok(unsafe { ptr.read() })
    }
}

impl GGufFileHeader {
    #[inline]
    pub const fn new(version: u32, tensor_count: u64, metadata_kv_count: u64) -> Self {
        Self {
            magic: MAGIC,
            version,
            tensor_count,
            metadata_kv_count,
        }
    }

    #[inline]
    pub fn is_magic_correct(&self) -> bool {
        self.magic == MAGIC
    }

    #[inline]
    pub const fn is_native_endian(&self) -> bool {
        // 先判断 native endian 再判断 file endian
        if u32::from_ne_bytes(MAGIC) == u32::from_le_bytes(MAGIC) {
            self.version == u32::from_le(self.version)
        } else {
            self.version == u32::from_be(self.version)
        }
    }

    #[inline]
    pub const fn magic(&self) -> Result<&str, Utf8Error> {
        from_utf8(&self.magic)
    }
}
