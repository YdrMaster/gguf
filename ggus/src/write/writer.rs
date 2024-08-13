use super::internal::GGufWriter;
use crate::{pad, GGmlType, GGufFileHeader, GGufMetaDataValueType, DEFAULT_ALIGNMENT};
use std::{
    io::{Result, Write},
    ops::Deref,
};

pub struct GGufMetaWriter<T: Write> {
    writer: GGufWriter<T>,
    alignment: usize,
}

pub struct GGufTensorWriter<T: Write, U> {
    writer: GGufWriter<T>,
    alignment: usize,
    data: Vec<U>,
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
    pub fn write_alignment(&mut self, alignment: usize) -> Result<()> {
        self.alignment = self.writer.write_alignment(alignment)?;
        Ok(())
    }

    #[inline]
    pub fn write_meta_kv(
        &mut self,
        key: &str,
        ty: GGufMetaDataValueType,
        val: &[u8],
    ) -> Result<()> {
        if let Some(alignment) = self.writer.write_meta_kv(key, ty, val)? {
            self.alignment = alignment;
        }
        Ok(())
    }

    #[inline]
    pub fn finish<U>(self) -> GGufTensorWriter<T, U> {
        GGufTensorWriter {
            writer: self.writer,
            alignment: self.alignment,
            data: Vec::new(),
            offset: 0,
        }
    }
}

impl<T: Write, U: Deref<Target = [u8]>> GGufTensorWriter<T, U> {
    pub fn write_tensor(&mut self, name: &str, ty: GGmlType, shape: &[u64], data: U) -> Result<()> {
        self.offset += pad(self.offset, self.alignment);
        self.writer
            .write_tensor_info(name, shape, ty, self.offset as _)
            .unwrap();

        let len = shape.iter().product::<u64>() as usize * ty.nbytes();
        self.offset += len;
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
            writer.write_data(&*data, alignment)?;
        }
        Ok(writer.written_bytes())
    }
}
