use crate::utils::{operate, show_file_info, Operator, OutputConfig};
use ggus::GGufFileName;
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct ConvertArgs {
    /// File to convert
    file: PathBuf,
    /// Output directory for converted files
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
    /// Operations to apply, separated by "->"
    #[clap(long)]
    ops: String,
    /// Max count of tensors per shard
    #[clap(long, short = 't')]
    max_tensors: Option<usize>,
    /// Max size in bytes per shard
    #[clap(long, short = 's')]
    max_bytes: Option<String>,
    /// If set, the first shard will not contain any tensor
    #[clap(long, short)]
    no_tensor_first: bool,
}

impl ConvertArgs {
    pub fn convert(self) {
        let Self {
            file,
            output_dir,
            ops,
            max_tensors,
            max_bytes,
            no_tensor_first,
        } = self;

        let name = GGufFileName::try_from(&*file).unwrap();
        let dir = file.parent().unwrap();
        let files = operate(
            name.clone(),
            name.clone()
                .iter_all()
                .map(|name| dir.join(name.to_string())),
            ops.split("->").map(|op| {
                let op = op.trim();
                if let Some(content) = op.strip_prefix("filter-meta:") {
                    Operator::filter_meta_key(content)
                } else if let Some(content) = op.strip_prefix("filter-tensor:") {
                    Operator::filter_tensor_name(content)
                } else if let Some(content) = op.strip_prefix("cast:") {
                    Operator::cast(content)
                } else if let Some(content) = op.strip_prefix("merge:") {
                    Operator::merge_linear(content)
                } else {
                    panic!("Unsupported operation: {}", op)
                }
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
