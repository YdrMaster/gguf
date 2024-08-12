use crate::{convert::ConvertArgs, file_info::show_file_info, name_pattern::compile_patterns};
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct FilterArgs {
    /// The file to filter
    file: PathBuf,
    /// Output directory for merged file
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
    /// Meta to keep
    #[clap(long, short = 'm', default_value = "*")]
    filter_meta: String,
    /// Tensors to keep
    #[clap(long, short = 't', default_value = "*")]
    filter_tensor: String,
}

impl FilterArgs {
    pub fn filter(self) {
        let Self {
            file: file_path,
            output_dir,
            filter_meta,
            filter_tensor,
        } = self;

        let files = ConvertArgs {
            output_name: file_path.file_stem().unwrap().to_str().unwrap().to_string() + ".part",
            input_files: vec![file_path],
            output_dir: output_dir.unwrap_or_else(|| std::env::current_dir().unwrap()),
            filter_meta: compile_patterns(&filter_meta),
            filter_tensor: compile_patterns(&filter_tensor),
            cast_data: None,
            optimize: vec![],
            split_tensor_count: usize::MAX,
            split_file_size: usize::MAX,
            split_no_tensor_first: false,
        }
        .convert()
        .unwrap();
        show_file_info(&files);
    }
}
