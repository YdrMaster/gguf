use super::internal::GGufWriter;
use crate::{pad, GGufFileHeader, GGufMetaDataValueType, GGufTensorInfo, DEFAULT_ALIGNMENT};
use std::io::{Result, Write};

pub struct GGufSimulator {
    writer: GGufWriter<NWrite>,
    alignment: usize,
    data: Vec<usize>,
    offset: usize,
}

impl Default for GGufSimulator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl GGufSimulator {
    #[inline]
    pub fn new() -> Self {
        Self {
            writer: GGufWriter::new(NWrite, GGufFileHeader::default()).unwrap(),
            alignment: DEFAULT_ALIGNMENT,
            data: Vec::new(),
            offset: 0,
        }
    }

    #[inline]
    pub fn with_alignment(align: usize) -> Self {
        let mut ans = Self::new();
        ans.write_alignment(align);
        ans
    }

    #[inline]
    pub fn write_alignment(&mut self, align: usize) {
        self.alignment = self.writer.write_alignment(align).unwrap();
    }

    #[inline]
    pub fn write_meta_kv(
        &mut self,
        key: impl AsRef<str>,
        ty: GGufMetaDataValueType,
        val: impl AsRef<[u8]>,
    ) {
        if let Some(align) = self.writer.write_meta_kv(key, ty, val).unwrap() {
            self.alignment = align;
        }
    }

    pub fn write_tensor(&mut self, info: &GGufTensorInfo) {
        self.offset += pad(self.offset, self.alignment);
        self.writer
            .write_tensor_info(
                info.name(),
                info.shape(),
                info.ggml_type(),
                self.offset as _,
            )
            .unwrap();

        let len = info.nbytes();
        self.offset += len;
        self.data.push(len);
    }

    pub fn written_bytes(&self) -> usize {
        let mut total = self.writer.written_bytes();
        for len in &self.data {
            total += pad(total, self.alignment);
            total += len;
        }
        total
    }
}

struct NWrite;

impl Write for NWrite {
    #[inline(always)]
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        Ok(buf.len())
    }
    #[inline(always)]
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}
