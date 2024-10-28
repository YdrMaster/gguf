use crate::{
    utils::{operate, show_file_info, OutputArgs},
    LogArgs,
};
use ggus::GGufFileName;
use std::{ops::Deref, path::PathBuf};

#[derive(Args, Default)]
pub struct SplitArgs {
    /// File to split
    file: PathBuf,

    #[clap(flatten)]
    output: OutputArgs,
    #[clap(flatten)]
    log: LogArgs,
}

impl SplitArgs {
    pub fn split(self) {
        let Self { file, output, log } = self;
        log.init();

        let name: GGufFileName = file.deref().try_into().unwrap();
        if name.shard_count() > 1 {
            println!("Model has already been splited");
            return;
        }

        let files = operate(name, [&file], [], output.into()).unwrap();
        show_file_info(&files);
    }
}
