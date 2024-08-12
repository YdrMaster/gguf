#![doc = include_str!("../README.md")]
#![deny(warnings)]

mod file;
mod header;
mod metadata;
mod name;
mod read;
mod tensor;
mod write;

pub use file::{GGuf, GGufError};
pub use header::GGufFileHeader;
pub use metadata::{
    utok, GGufArray, GGufFileType, GGufMetaDataValueType, GGufMetaKV, GGufMetaKVPairs,
    GGufTokenType, DEFAULT_ALIGNMENT, GENERAL_ALIGNMENT,
};
pub use name::GGufFileName;
pub use read::{GGufReadError, GGufReader};
pub use tensor::{GGmlType, GGufTensorInfo, GGufTensors};
pub use write::{GGufMetaWriter, GGufSimulator, GGufTensorWriter};

#[inline(always)]
const fn pad(pos: usize, align: usize) -> usize {
    (align - pos % align) % align
}
