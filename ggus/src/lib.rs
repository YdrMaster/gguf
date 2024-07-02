#![doc = include_str!("../README.md")]
#![deny(warnings)]

mod header;
mod metadata;
mod name;
mod reader;
mod tensor;
mod writer;

pub use header::GGufFileHeader;
pub use metadata::{
    utok, GGufArray, GGufFileType, GGufMetaDataValueType, GGufMetaKV, GGufMetaKVPairs,
};
pub use name::GGufFileName;
pub use reader::{GGufReadError, GGufReader};
pub use tensor::{GGufTensorInfo, GGufTensors};
pub use writer::GGufWriter;

macro_rules! sizeof {
    ($ty:ty) => {
        std::mem::size_of::<$ty>()
    };
}

use sizeof;
