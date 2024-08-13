mod operator;
mod read;
mod write;

use crate::file_info::FileInfo;
use ggus::{GGmlType, GGufError, GGufMetaDataValueType};
use indexmap::IndexMap;
use memmap2::Mmap;
use read::read_files;
use std::{
    fs::File,
    io,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, LazyLock},
};

pub use operator::Operator;

pub struct ConvertArgs {
    pub input_files: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub output_name: String,
    pub operations: Vec<Operator>,
    pub split_tensor_count: usize,
    pub split_file_size: usize,
    pub split_no_tensor_first: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum ConvertError {
    GGuf(GGufError),
    Io(io::Error),
}

impl ConvertArgs {
    pub fn convert(self) -> Result<Vec<FileInfo>, ConvertError> {
        let Self {
            input_files,
            output_dir,
            output_name,
            operations,
            split_tensor_count,
            split_file_size,
            split_no_tensor_first,
        } = self;

        let files = input_files
            .into_iter()
            .map(|path| File::open(path).and_then(|f| unsafe { Mmap::map(&f) }))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ConvertError::Io)?;
        let files = read_files(files.iter().map(|m| &**m)).map_err(ConvertError::GGuf)?;

        let mut content = Content::new(&files);

        for op in operations {
            content.apply(op);
        }

        content
            .write_files(
                &output_dir,
                &output_name,
                split_tensor_count,
                split_file_size,
                split_no_tensor_first,
            )
            .map_err(ConvertError::Io)
    }
}

struct Content<'a> {
    alignment: usize,
    meta_kvs: IndexMap<String, MetaValue<'a>>,
    tensors: IndexMap<String, Tensor<'a>>,
}

struct MetaValue<'a> {
    ty: GGufMetaDataValueType,
    value: DataPromise<'a>,
}

struct Tensor<'a> {
    ty: GGmlType,
    shape: Vec<u64>,
    data: DataPromise<'a>,
}

#[derive(Clone)]
#[repr(transparent)]
struct DataPromise<'a>(Arc<dyn DataFuture + Send + Sync + 'a>);

impl Deref for DataPromise<'_> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.0.get()
    }
}

trait DataFuture {
    fn get(&self) -> &[u8];
}

impl DataFuture for &[u8] {
    #[inline]
    fn get(&self) -> &[u8] {
        self
    }
}

impl<F: FnOnce() -> Mmap> DataFuture for LazyLock<Mmap, F> {
    #[inline]
    fn get(&self) -> &[u8] {
        &**self
    }
}
