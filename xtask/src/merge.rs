use crate::{convert::ConvertArgs, name_pattern::compile_patterns, shards::Shards};
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct MergeArgs {
    /// One of the shards to merge.
    file: PathBuf,
    /// Output directory for merged file
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
}

impl MergeArgs {
    pub fn merge(self) {
        let Self { file, output_dir } = self;

        let shards = Shards::from(&*file);
        if shards.count < 2 {
            println!("Model does not need to merge.");
            return;
        }

        let files = ConvertArgs {
            input_files: shards.iter_all().collect(),
            output_dir: output_dir.unwrap_or_else(|| std::env::current_dir().unwrap()),
            output_name: shards.name.into(),
            filter_meta: compile_patterns("*"),
            filter_tensor: compile_patterns("*"),
            cast_data: None,
            optimize: Vec::new(),
            split_tensor_count: usize::MAX,
            split_file_size: usize::MAX,
            split_no_tensor_first: false,
        }
        .convert()
        .unwrap();

        let [file] = &*files else {
            unreachable!();
        };
        println!("{file}");
    }
}
