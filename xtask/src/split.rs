use crate::{convert::ConvertArgs, name_pattern::compile_patterns, shards::Shards};
use std::{path::PathBuf, str::from_utf8};

#[derive(Args, Default)]
pub struct SplitArgs {
    /// File to split
    file: PathBuf,
    /// Output directory for splited shards
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
    /// Max count of tensors per shard
    #[clap(long, short = 't')]
    max_tensors: Option<usize>,
    /// Max size in bytes per shard
    #[clap(long, short = 's')]
    max_bytes: Option<String>,
    /// If set, the first shard will not contain any tensor
    #[clap(long, short)]
    no_tensor_first: bool,
}

impl SplitArgs {
    pub fn split(self) {
        let Self {
            file,
            output_dir,
            max_tensors,
            max_bytes,
            no_tensor_first,
        } = self;

        let shards = Shards::from(&*file);
        if shards.count > 1 {
            println!("Model has already been splited");
            return;
        }

        fn parse_size_num(num: &[u8], k: usize) -> Option<usize> {
            from_utf8(num).ok()?.parse().ok().map(|n: usize| n << k)
        }

        let files = ConvertArgs {
            output_name: shards.name.into(),
            input_files: vec![file],
            output_dir: output_dir.unwrap_or_else(|| std::env::current_dir().unwrap()),
            filter_meta: compile_patterns("*"),
            filter_tensor: compile_patterns("*"),
            cast_data: None,
            optimize: Vec::new(),
            split_tensor_count: max_tensors.unwrap_or(usize::MAX),
            split_file_size: match max_bytes {
                Some(s) => match s.trim().as_bytes() {
                    [num @ .., b'G'] => parse_size_num(num, 30),
                    [num @ .., b'M'] => parse_size_num(num, 20),
                    [num @ .., b'K'] => parse_size_num(num, 10),
                    num => parse_size_num(num, 0),
                }
                .unwrap_or_else(|| panic!("Invalid max bytes format: \"{s}\"")),
                None => usize::MAX,
            },
            split_no_tensor_first: no_tensor_first,
        }
        .convert()
        .unwrap();

        for file in files {
            println!("{file}");
            println!();
        }
    }
}
