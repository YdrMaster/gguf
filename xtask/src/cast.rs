use crate::{
    utils::OutputArgs,
    utils::{operate, show_file_info, Operator},
    LogArgs,
};
use ggus::GGufFileName;
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct CastArgs {
    /// File to convert
    file: PathBuf,
    #[clap(long, short)]
    types: String,

    #[clap(flatten)]
    output: OutputArgs,
    #[clap(flatten)]
    log: LogArgs,
}

impl CastArgs {
    pub fn cast(self) {
        let Self {
            file,
            types,
            output,
            log,
        } = self;
        log.init();

        let name = GGufFileName::try_from(&*file).unwrap();
        let dir = file.parent().unwrap();
        let files = operate(
            name.clone(),
            name.iter_all().map(|name| dir.join(name.to_string())),
            [Operator::cast(&types)],
            output.into(),
        )
        .unwrap();

        show_file_info(&files);
    }
}
