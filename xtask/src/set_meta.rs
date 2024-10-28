use crate::{
    utils::{operate, show_file_info, Operator, OutputArgs},
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

    #[clap(flatten)]
    output: OutputArgs,
    #[clap(flatten)]
    log: LogArgs,
}

impl SetMetaArgs {
    pub fn set_meta(self) {
        let Self {
            file,
            meta_kvs,
            output,
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
            output.into(),
        )
        .unwrap();

        show_file_info(&files);
    }
}
