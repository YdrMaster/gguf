use crate::{
    utils::{operate, show_file_info, Operator, OutputConfig},
    LogArgs,
};
use ggus::GGufFileName;
use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

#[derive(Args, Default)]
pub struct SetMetaArgs {
    /// File to set metadata
    file: PathBuf,
    /// Meta data to set for the file
    meta_kvs: String,
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

        let path = Path::new(&meta_kvs);
        let cfg = if path.is_file() {
            let cfg = read_to_string(path).unwrap();
            // 消除 utf-8 BOM
            if cfg.as_bytes()[..3] == [0xef, 0xbb, 0xbf] {
                cfg[3..].to_string()
            } else {
                cfg
            }
        } else {
            meta_kvs
        };

        let files = operate(
            GGufFileName::try_from(&*file).unwrap(),
            [&file],
            [Operator::set_meta_by_cfg(&cfg)],
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
