use crate::sizeof;
use std::str::Utf8Error;

#[derive(Default, Debug)]
#[repr(C)]
pub struct GGufFileHeader {
    magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

const MAGIC: [u8; 4] = *b"GGUF";

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
        std::str::from_utf8(&self.magic)
    }

    #[inline]
    pub const fn nbytes(&self) -> usize {
        sizeof!(Self)
    }
}
