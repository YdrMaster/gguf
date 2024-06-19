#[derive(Default, Debug)]
#[repr(C)]
pub struct GGufFileHeader {
    magic: [u8; 4],
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

const MAGIC: [u8; 4] = *b"GGUF";

impl GGufFileHeader {
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
    pub const fn version(&self) -> u32 {
        self.version
    }

    #[inline]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count as _
    }

    #[inline]
    pub const fn metadata_kv_count(&self) -> usize {
        self.metadata_kv_count as _
    }
}
