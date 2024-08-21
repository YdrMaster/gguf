mod file_info;
mod name_pattern;
mod operator;
mod read;
mod write;

use file_info::FileInfo;
use ggus::{GGmlType, GGufError, GGufFileName, GGufMetaDataValueType, GGufReader};
use indexmap::IndexMap;
use memmap2::{Mmap, MmapMut};
use read::read_files;
use std::{
    borrow::Cow,
    fs::File,
    io,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock},
};

pub(crate) use file_info::{show_file_info, MemSize};
pub(crate) use name_pattern::compile_patterns;
pub(crate) use operator::Operator;

pub(crate) struct OutputConfig<'a> {
    pub dir: PathBuf,
    pub name: GGufFileName<'a>,
    pub shard_max_tensor_count: usize,
    pub shard_max_file_size: MemSize,
    pub shard_no_tensor_first: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum OperateError {
    GGuf(GGufError),
    Io(io::Error),
}

pub(crate) fn operate<T: AsRef<Path>>(
    input_files: impl IntoIterator<Item = T>,
    operations: impl IntoIterator<Item = Operator>,
    out: OutputConfig,
) -> Result<Vec<FileInfo>, OperateError> {
    let files = input_files
        .into_iter()
        .map(|path| File::open(path).and_then(|f| unsafe { Mmap::map(&f) }))
        .collect::<Result<Vec<_>, _>>()
        .map_err(OperateError::Io)?;

    let files = read_files(files.iter().map(|m| &**m)).map_err(OperateError::GGuf)?;
    let mut content = Content::new(&files);
    for op in operations {
        content.apply(op);
    }
    content.write_files(out).map_err(OperateError::Io)
}

struct Content<'a> {
    alignment: usize,
    meta_kvs: IndexMap<String, MetaValue<'a>>,
    tensors: IndexMap<String, Tensor<'a>>,
}

struct MetaValue<'a> {
    ty: GGufMetaDataValueType,
    value: Cow<'a, [u8]>,
}

impl MetaValue<'_> {
    #[inline]
    fn value_reader(&self) -> GGufReader {
        GGufReader::new(&self.value)
    }
}

struct Tensor<'a> {
    ty: GGmlType,
    shape: Vec<u64>,
    data: DataPromise<'a>,
}

#[derive(Clone)]
enum DataPromise<'a> {
    Borrowed(&'a [u8]),
    Lazy(Arc<dyn LazyData + Send + Sync + 'a>),
}

impl<'a> DataPromise<'a> {
    fn lazy(f: impl FnOnce() -> MmapMut + Send + Sync + 'a) -> Self {
        Self::Lazy(Arc::new(LazyLock::new(f)))
    }
}

impl ggus::DataFuture for DataPromise<'_> {
    #[inline]
    fn get(&self) -> &[u8] {
        match self {
            Self::Borrowed(data) => data,
            Self::Lazy(data) => data.get(),
        }
    }
}

trait LazyData {
    fn get(&self) -> &[u8];
}

impl<F: FnOnce() -> MmapMut> LazyData for LazyLock<MmapMut, F> {
    #[inline]
    fn get(&self) -> &[u8] {
        self
    }
}
