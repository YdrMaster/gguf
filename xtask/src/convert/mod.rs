mod read;
mod write;

use crate::file_info::FileInfo;
use digit_layout::DigitLayout;
use ggus::{GGufError, GENERAL_ALIGNMENT};
use memmap2::Mmap;
use read::read_files;
use regex::Regex;
use std::{fs::File, io, path::PathBuf};

pub struct ConvertArgs {
    pub input_files: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub output_name: String,

    pub filter_meta: Regex,
    pub filter_tensor: Regex,
    pub cast_data: Option<DigitLayout>,
    pub optimize: Vec<String>,

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
            filter_meta,
            filter_tensor,
            cast_data,
            optimize,
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

        assert!(cast_data.is_none());
        assert!(optimize.is_empty());

        let align = files.iter().map(|f| f.meta_kvs.alignment()).max().unwrap();
        let meta_kvs = files
            .iter()
            .flat_map(|f| f.meta_kvs.kvs())
            .filter(|kv| {
                let k = kv.key();
                k != GENERAL_ALIGNMENT && !k.starts_with("split.") && filter_meta.is_match(k)
            })
            .collect::<Vec<_>>();
        let tensors = files
            .iter()
            .enumerate()
            .flat_map(|(i, f)| {
                let data = files[i].data;
                f.tensors.iter().map(move |t| (t, data))
            })
            .filter(|(t, _)| filter_tensor.is_match(t.name()))
            .collect::<Vec<_>>();

        write::write_files(
            &meta_kvs,
            &tensors,
            &output_dir,
            &output_name,
            align,
            split_tensor_count,
            split_file_size,
            split_no_tensor_first,
        )
        .map_err(ConvertError::Io)
    }
}
