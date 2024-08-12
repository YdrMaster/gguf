use crate::{
    GGufFileHeader, GGufMetaDataValueType, GGufTensorInfo, DEFAULT_ALIGNMENT, GENERAL_ALIGNMENT,
};
use internal::Internal;
use std::{
    io::{Result, Write},
    mem::size_of_val,
    slice::from_raw_parts,
};

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

pub struct GGufSimulator {
    writer: GGufWriter<NWrite>,
    alignment: usize,
    direct: usize,
    data: Vec<usize>,
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
        self.writer.write_str(info.name())?;

        let shape = info.shape();
        self.writer.write(shape.len() as u32)?;
        self.writer.write_bytes(as_slice(shape))?;
        self.writer.write(info.ggml_type())?;

        self.offset += pad(self.offset, self.alignment);
        self.writer.write(self.offset as u64)?;

        let len = info.nbytes();
        self.offset += len;
        self.data.push(&data[info.offset()..][..len]);

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
            for _ in 0..pad(writer.written_bytes(), alignment) {
                writer.write(0u8)?;
            }
            writer.write_bytes(data)?;
        }
        Ok(writer.written_bytes())
    }
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
            direct: 0,
            data: Vec::new(),
        }
    }

    #[inline]
    pub fn with_alignment(align: usize) -> Self {
        let mut ans = Self::new();
        ans.write_alignment(align);
        ans
    }

    #[inline]
    pub fn write(&mut self, n: usize) {
        self.direct += n;
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
        self.writer.write_str(info.name()).unwrap();

        let shape = info.shape();
        self.writer.write(shape.len() as u32).unwrap();
        self.writer.write_bytes(as_slice(shape)).unwrap();

        self.writer.write(info.ggml_type()).unwrap();
        self.writer.write(0u64).unwrap();

        self.data.push(info.nbytes());
    }

    pub fn written_bytes(&self) -> usize {
        let mut total = self.writer.written_bytes() + self.direct;
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

struct GGufWriter<T: Write>(Internal<T>);

impl<T: Write> GGufWriter<T> {
    #[inline]
    pub fn new(writer: T, header: GGufFileHeader) -> Result<Self> {
        let mut ans = Self(Internal::new(writer));
        ans.write_bytes(as_slice(&header))?;
        Ok(ans)
    }

    #[inline]
    pub const fn written_bytes(&self) -> usize {
        self.0.written_bytes()
    }

    #[inline]
    pub fn write_bytes(&mut self, val: &[u8]) -> Result<()> {
        self.0.write_bytes(val)
    }

    #[inline]
    pub fn write<U: Copy>(&mut self, val: U) -> Result<()> {
        self.write_bytes(as_slice(&val))
    }

    #[inline]
    pub fn write_str(&mut self, val: impl AsRef<str>) -> Result<()> {
        let val = val.as_ref();
        self.write_bytes(as_slice(&(val.len() as u64)))?;
        self.write_bytes(val.as_bytes())
    }

    #[inline]
    pub fn write_alignment(&mut self, align: usize) -> Result<usize> {
        self.write_meta_kv(
            GENERAL_ALIGNMENT,
            GGufMetaDataValueType::U32,
            (align as u32).to_le_bytes(),
        )
        .map(Option::unwrap)
    }

    pub fn write_meta_kv(
        &mut self,
        key: impl AsRef<str>,
        ty: GGufMetaDataValueType,
        val: impl AsRef<[u8]>,
    ) -> Result<Option<usize>> {
        let key = key.as_ref();
        let val = val.as_ref();

        self.write_str(key)?;
        self.write(ty)?;
        self.write_bytes(val)?;

        Ok(if key == GENERAL_ALIGNMENT {
            let &[a, b, c, d] = val else {
                panic!("general.alignment must be an u32")
            };
            Some(u32::from_le_bytes([a, b, c, d]) as _)
        } else {
            None
        })
    }
}

#[inline(always)]
fn as_slice<T: ?Sized>(val: &T) -> &[u8] {
    unsafe { from_raw_parts(val as *const _ as *const _, size_of_val(val)) }
}

#[inline(always)]
const fn pad(pos: usize, alignment: usize) -> usize {
    (alignment - pos % alignment) % alignment
}

mod internal {
    use std::io::{BufWriter, Result, Write};

    pub(super) struct Internal<T: Write>(BufWriter<T>, usize);

    impl<T: Write> Internal<T> {
        #[inline]
        pub fn new(writer: T) -> Self {
            Self(BufWriter::new(writer), 0)
        }

        #[inline]
        pub const fn written_bytes(&self) -> usize {
            self.1
        }

        #[inline]
        pub fn write_bytes(&mut self, val: &[u8]) -> Result<()> {
            self.1 += val.len();
            self.0.write_all(val.as_ref())
        }
    }
}
