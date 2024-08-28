use crate::{
    utils::{operate, show_file_info, Operator, OutputConfig},
    LogArgs,
};
use ggus::GGufFileName;
use std::{fs::read_to_string, path::PathBuf};

#[derive(Args, Default)]
pub struct SetMetaArgs {
    /// File to set metadata
    file: PathBuf,
    /// Meta data to set for the file
    meta_kvs: PathBuf,
    /// Output directory for changed file
    #[clap(long, short)]
    output_dir: Option<PathBuf>,

    #[clap(flatten)]
    log: LogArgs,
}

impl SetMetaArgs {
    pub fn set_meta(self) {
        let Self {
            file,
            meta_kvs,
            output_dir,
            log,
        } = self;
        log.init();

        let cfg = read_to_string(meta_kvs).unwrap();
        // 消除 utf-8 BOM
        let cfg = if cfg.as_bytes()[..3] == [0xef, 0xbb, 0xbf] {
            &cfg[3..]
        } else {
            &cfg[..]
        };

        let files = operate(
            GGufFileName::try_from(&*file).unwrap(),
            [&file],
            [Operator::set_meta_by_cfg(cfg)],
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
