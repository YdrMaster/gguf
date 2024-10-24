use crate::{
    utils::{operate, show_file_info, Operator, OutputConfig},
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
    /// Output directory for converted files
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

    #[clap(flatten)]
    log: LogArgs,
}

impl ToLlamaArgs {
    pub fn convert_to_llama(self) {
        let Self {
            file,
            extra,
            output_dir,
            max_tensors,
            max_bytes,
            no_tensor_first,
            log,
        } = self;
        log.init();

        let name = GGufFileName::try_from(&*file).unwrap();
        let dir = file.parent().unwrap();
        let files = operate(
            name.clone(),
            name.iter_all().map(|name| dir.join(name.to_string())),
            [Operator::ToLlama(extra)],
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
