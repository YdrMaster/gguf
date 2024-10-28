use crate::{
    utils::{operate, show_file_info, Operator, OutputArgs},
    LogArgs,
};
use ggus::GGufFileName;
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct ToLlamaArgs {
    /// File to convert
    file: PathBuf,
    /// Extra metadata for convertion
    #[clap(long, short = 'x')]
    extra: Option<String>,

    #[clap(flatten)]
    output: OutputArgs,
    #[clap(flatten)]
    log: LogArgs,
}

impl ToLlamaArgs {
    pub fn convert_to_llama(self) {
        let Self {
            file,
            extra,
            output,
            log,
        } = self;
        log.init();

        let name = GGufFileName::try_from(&*file).unwrap();
        let dir = file.parent().unwrap();
        let files = operate(
            name.clone(),
            name.iter_all().map(|name| dir.join(name.to_string())),
            [Operator::ToLlama(extra)],
            output.into(),
        )
        .unwrap();

        show_file_info(&files);
    }
}
