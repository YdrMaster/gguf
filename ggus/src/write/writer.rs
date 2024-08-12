use super::internal::GGufWriter;
use crate::{pad, GGufFileHeader, GGufMetaDataValueType, GGufTensorInfo, DEFAULT_ALIGNMENT};
use std::io::{Result, Write};

pub struct GGufMetaWriter<T: Write> {
    writer: GGufWriter<T>,
    alignment: usize,
}

pub struct GGufTensorWriter<'t, T: Write> {
    writer: GGufWriter<T>,
    alignment: usize,
    data: Vec<&'t [u8]>,
    offset: usize,
}

impl<T: Write> GGufMetaWriter<T> {
    #[inline]
    pub fn new(writer: T, header: GGufFileHeader) -> Result<Self> {
        Ok(Self {
            writer: GGufWriter::new(writer, header)?,
            alignment: DEFAULT_ALIGNMENT,
        })
    }

    #[inline]
    pub fn write_alignment(&mut self, align: usize) -> Result<()> {
        self.alignment = self.writer.write_alignment(align)?;
        Ok(())
    }

    #[inline]
    pub fn write_meta_kv(
        &mut self,
        key: impl AsRef<str>,
        ty: GGufMetaDataValueType,
        val: impl AsRef<[u8]>,
    ) -> Result<()> {
        if let Some(align) = self.writer.write_meta_kv(key, ty, val)? {
            self.alignment = align;
        }
        Ok(())
    }

    #[inline]
    pub fn finish<'t>(self) -> GGufTensorWriter<'t, T> {
        GGufTensorWriter {
            writer: self.writer,
            alignment: self.alignment,
            data: Vec::new(),
            offset: 0,
        }
    }
}

impl<'t, T: Write> GGufTensorWriter<'t, T> {
    pub fn write_tensor(&mut self, info: &GGufTensorInfo, data: &'t [u8]) -> Result<()> {
        self.offset += pad(self.offset, self.alignment);
        self.writer.write_tensor_info(
            info.name(),
            info.shape(),
            info.ggml_type(),
            self.offset as _,
        )?;

        let data = &data[info.offset()..][..info.nbytes()];
        self.offset += data.len();
        self.data.push(data);

        Ok(())
    }

    pub fn finish(self) -> Result<usize> {
        let Self {
            mut writer,
            alignment,
            data,
            ..
        } = self;

        for data in data {
            writer.write_data(data, alignment)?;
        }
        Ok(writer.written_bytes())
    }
}
