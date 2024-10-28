use crate::{
    utils::OutputArgs,
    utils::{operate, show_file_info, Operator},
    LogArgs,
};
use ggus::GGufFileName;
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct ConvertArgs {
    /// File to convert
    file: PathBuf,
    /// Steps to apply, separated by "->", maybe "sort", "merge-linear", "split-linear", "filter-meta:<key>" or "filter-tensor:<name>"
    #[clap(long, short = 'x')]
    steps: String,

    #[clap(flatten)]
    output: OutputArgs,
    #[clap(flatten)]
    log: LogArgs,
}

impl ConvertArgs {
    pub fn convert(self) {
        let Self {
            file,
            steps,
            output,
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
                    _ => panic!("Unsupported operation: {op}"),
                },
            }),
            output.into(),
        )
        .unwrap();

        show_file_info(&files);
    }
}
