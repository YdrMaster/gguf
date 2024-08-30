use crate::{
    utils::{operate, show_file_info, Operator, OutputConfig},
    LogArgs,
};
use ggus::GGufFileName;
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct ConvertArgs {
    /// File to convert
    file: PathBuf,
    /// Output directory for converted files
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
    /// Steps to apply, separated by "->", maybe "sort", "merge-linear", "split-linear", "filter-meta:<key>", "filter-tensor:<name>", "cast:<dtype>" or "distribute:<n>"
    #[clap(long, short = 'x')]
    steps: String,
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

impl ConvertArgs {
    pub fn convert(self) {
        let Self {
            file,
            output_dir,
            steps,
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
            steps.split("->").map(|op| match op.trim() {
                "sort" => Operator::SortTensors,
                "merge-linear" => Operator::MergeLinear(true),
                "split-linear" | "!merge-linear" => Operator::MergeLinear(false),
                op => match op.split_once(':') {
                    Some(("filter-meta", key)) => Operator::filter_meta_key(key),
                    Some(("filter-tensor", name)) => Operator::filter_tensor_name(name),
                    Some(("cast", dtype)) => Operator::quantize(dtype),
                    Some(("distribute", n)) => Operator::distribute(n),
                    _ => panic!("Unsupported operation: {op}"),
                },
            }),
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
