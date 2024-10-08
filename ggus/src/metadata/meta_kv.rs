﻿use super::GGufMetaDataValueType as Ty;
use crate::{GGufReadError, GGufReader};
use std::marker::PhantomData;

#[derive(Clone)]
#[repr(transparent)]
pub struct GGufMetaKV<'a>(&'a [u8]);

impl<'a> GGufReader<'a> {
    pub fn read_meta_kv(&mut self) -> Result<GGufMetaKV<'a>, GGufReadError> {
        let data = self.remaining();

        let _k = self.read_str()?;
        let ty = self.read()?;
        self.read_meta_value(ty, 1)?;

        let data = &data[..data.len() - self.remaining().len()];
        Ok(unsafe { GGufMetaKV::new_unchecked(data) })
    }

    fn read_meta_value(&mut self, ty: Ty, len: usize) -> Result<&mut Self, GGufReadError> {
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
                self.read_meta_value(ty, len)
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

    #[inline]
    pub fn ty(&self) -> Ty {
        self.reader().skip_str().unwrap().read().unwrap()
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

    pub fn read_integer(&self) -> isize {
        let mut reader = self.reader();
        let ty = reader.skip_str().unwrap().read::<Ty>().unwrap();
        match ty {
            Ty::Bool | Ty::U8 => reader.read::<u8>().unwrap().into(),
            Ty::I8 => reader.read::<i8>().unwrap().into(),
            Ty::U16 => reader.read::<u16>().unwrap().try_into().unwrap(),
            Ty::I16 => reader.read::<i16>().unwrap().into(),
            Ty::U32 => reader.read::<u32>().unwrap().try_into().unwrap(),
            Ty::I32 => reader.read::<i32>().unwrap().try_into().unwrap(),
            Ty::U64 => reader.read::<u64>().unwrap().try_into().unwrap(),
            Ty::I64 => reader.read::<i64>().unwrap().try_into().unwrap(),
            Ty::Array | Ty::String | Ty::F32 | Ty::F64 => panic!("not an integer type"),
        }
    }

    pub fn read_unsigned(&self) -> usize {
        let mut reader = self.reader();
        let ty = reader.skip_str().unwrap().read::<Ty>().unwrap();
        match ty {
            Ty::Bool | Ty::U8 => reader.read::<u8>().unwrap().into(),
            Ty::U16 => reader.read::<u16>().unwrap().into(),
            Ty::U32 => reader.read::<u32>().unwrap().try_into().unwrap(),
            Ty::U64 => reader.read::<u64>().unwrap().try_into().unwrap(),
            Ty::I8 => reader.read::<i8>().unwrap().try_into().unwrap(),
            Ty::I16 => reader.read::<i16>().unwrap().try_into().unwrap(),
            Ty::I32 => reader.read::<i32>().unwrap().try_into().unwrap(),
            Ty::I64 => reader.read::<i64>().unwrap().try_into().unwrap(),
            Ty::Array | Ty::String | Ty::F32 | Ty::F64 => panic!("not an integer type"),
        }
    }

    #[inline]
    fn reader(&self) -> GGufReader<'a> {
        GGufReader::new(self.0)
    }
}

pub struct GGufMetaValueArray<'a, T: ?Sized> {
    reader: GGufReader<'a>,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: ?Sized> GGufMetaValueArray<'a, T> {
    pub fn new(reader: GGufReader<'a>, len: usize) -> Self {
        Self {
            reader,
            len,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }
}

impl<'a> Iterator for GGufMetaValueArray<'a, str> {
    type Item = Result<&'a str, GGufReadError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            self.len -= 1;
            Some(self.reader.read_str())
        } else {
            None
        }
    }
}

impl<T: Copy> Iterator for GGufMetaValueArray<'_, T> {
    type Item = Result<T, GGufReadError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            self.len -= 1;
            Some(self.reader.read())
        } else {
            None
        }
    }
}
