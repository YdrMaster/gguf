use crate::utils::{operate, show_file_info, OutputConfig};
use ggus::GGufFileName;
use std::{ops::Deref, path::PathBuf};

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

        let dir = file.parent().unwrap();
        let name: GGufFileName = file.deref().try_into().unwrap();
        if name.shard_count() == 1 {
            println!("Model does not need to merge.");
            return;
        }

        let files = operate(
            name.clone(),
            name.iter_all().map(|name| dir.join(name.to_string())),
            [],
            OutputConfig {
                dir: output_dir,
                shard_max_tensor_count: usize::MAX,
                shard_max_file_size: Default::default(),
                shard_no_tensor_first: false,
            },
        )
        .unwrap();

        show_file_info(&files);
    }
}
