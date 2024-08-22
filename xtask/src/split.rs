use crate::utils::{operate, show_file_info, OutputConfig};
use ggus::GGufFileName;
use std::{ops::Deref, path::PathBuf};

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

        let name: GGufFileName = file.deref().try_into().unwrap();
        if name.shard_count() > 1 {
            println!("Model has already been splited");
            return;
        }

        let files = operate(
            name,
            [&file],
            [],
            OutputConfig {
                dir: output_dir,
                shard_max_tensor_count: max_tensors.unwrap_or(usize::MAX),
                shard_max_file_size: max_bytes.map_or(Default::default(), |s| s.parse().unwrap()),
                shard_no_tensor_first: no_tensor_first,
            },
        )
        .unwrap();

        show_file_info(&files);
    }
}
