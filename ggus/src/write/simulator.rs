use super::internal::GGufWriter;
use crate::{pad, GGmlType, GGufFileHeader, GGufMetaDataValueType, DEFAULT_ALIGNMENT};
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
    pub fn with_alignment(alignment: usize) -> Self {
        let mut ans = Self::new();
        ans.write_alignment(alignment);
        ans
    }

    #[inline]
    pub fn write_alignment(&mut self, alignment: usize) {
        self.alignment = self.writer.write_alignment(alignment).unwrap();
    }

    #[inline]
    pub fn write_meta_kv(&mut self, key: &str, ty: GGufMetaDataValueType, val: &[u8]) {
        if let Some(alignment) = self.writer.write_meta_kv(key, ty, val).unwrap() {
            self.alignment = alignment;
        }
    }

    pub fn write_tensor(&mut self, name: &str, ty: GGmlType, shape: &[u64]) {
        self.offset += pad(self.offset, self.alignment);
        self.writer
            .write_tensor_info(name, shape, ty, self.offset as _)
            .unwrap();

        let len = shape.iter().product::<u64>() as usize * ty.nbytes();
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
