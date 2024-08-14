use super::GGufMetaDataValueType as Ty;
use crate::{GGufReadError, GGufReader};

#[repr(transparent)]
pub struct GGufMetaKV<'a>(&'a [u8]);

impl<'a> GGufReader<'a> {
    pub fn read_meta_kv(&mut self) -> Result<GGufMetaKV<'a>, GGufReadError> {
        let data = self.remaining();

        let _k = self.read_str()?;
        let ty = self.read()?;
        self.skip_meta_value(ty, 1)?;

        let data = &data[..data.len() - self.remaining().len()];
        Ok(unsafe { GGufMetaKV::new_unchecked(data) })
    }

    fn skip_meta_value(&mut self, ty: Ty, len: usize) -> Result<&mut Self, GGufReadError> {
        match ty {
            Ty::U8 => self.skip::<u8>(len),
            Ty::I8 => self.skip::<i8>(len),
            Ty::U16 => self.skip::<u16>(len),
            Ty::I16 => self.skip::<i16>(len),
            Ty::U32 => self.skip::<u32>(len),
            Ty::I32 => self.skip::<i32>(len),
            Ty::F32 => self.skip::<f32>(len),
            Ty::U64 => self.skip::<u64>(len),
            Ty::I64 => self.skip::<i64>(len),
            Ty::F64 => self.skip::<f64>(len),
            Ty::Bool => {
                for _ in 0..len {
                    self.read_bool()?;
                }
                Ok(self)
            }
            Ty::String => {
                for _ in 0..len {
                    self.read_str()?;
                }
                Ok(self)
            }
            Ty::Array => {
                let (ty, len) = self.read_arr_header()?;
                self.skip_meta_value(ty, len)
            }
        }
    }
}

impl<'a> GGufMetaKV<'a> {
    /// Creates a new [GGufMetaKV] instance without performing any validation on the input data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the input data is valid for the [GGufMetaKV] type.
    #[inline]
    pub const unsafe fn new_unchecked(data: &'a [u8]) -> Self {
        Self(data)
    }

    #[inline]
    pub fn new(data: &'a [u8]) -> Result<Self, GGufReadError> {
        GGufReader::new(data).read_meta_kv()
    }

    #[inline]
    pub fn key(&self) -> &'a str {
        let mut reader = self.reader();
        unsafe { reader.read_str_unchecked() }
    }

    pub fn ty(&self) -> Ty {
        self.reader().skip_str().unwrap().read::<Ty>().unwrap()
    }

    pub fn value_bytes(&self) -> &'a [u8] {
        self.reader()
            .skip_str()
            .unwrap()
            .skip::<Ty>(1)
            .unwrap()
            .remaining()
    }

    pub fn value_reader(&self) -> GGufReader<'a> {
        let mut reader = self.reader();
        reader.skip_str().unwrap().skip::<Ty>(1).unwrap();
        reader
    }

    #[inline]
    fn reader(&self) -> GGufReader<'a> {
        GGufReader::new(self.0)
    }
}
