use crate::utils::{operate, show_file_info, OutputConfig, Shards};
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

        let files = operate(
            shards.iter_all(),
            [],
            OutputConfig {
                dir: output_dir.unwrap_or_else(|| std::env::current_dir().unwrap()),
                name: shards.name.into(),
                shard_max_tensor_count: usize::MAX,
                shard_max_file_size: Default::default(),
                shard_no_tensor_first: false,
            },
        )
        .unwrap();

        show_file_info(&files);
    }
}
