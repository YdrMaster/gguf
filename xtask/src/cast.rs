use crate::{
    utils::OutputArgs,
    utils::{operate, show_file_info, Operator},
    LogArgs,
};
use ggus::{GGmlType, GGufFileName};
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct CastArgs {
    /// File to convert
    file: PathBuf,
    #[clap(long)]
    embd: Option<String>,
    #[clap(long)]
    norm: Option<String>,
    #[clap(long)]
    mat: Option<String>,

    #[clap(flatten)]
    output: OutputArgs,
    #[clap(flatten)]
    log: LogArgs,
}

impl CastArgs {
    pub fn cast(self) {
        let Self {
            file,
            embd,
            norm,
            mat,
            output,
            log,
        } = self;
        log.init();

        let name = GGufFileName::try_from(&*file).unwrap();
        let dir = file.parent().unwrap();
        let files = operate(
            name.clone(),
            name.iter_all().map(|name| dir.join(name.to_string())),
            [Operator::Cast {
                embd: embd.map(parse),
                norm: norm.map(parse),
                mat: mat.map(parse),
            }],
            output.into(),
        )
        .unwrap();

        show_file_info(&files);
    }
}

#[rustfmt::skip]
fn parse(s: String) -> GGmlType {
    use GGmlType as Ty;
    match s.to_ascii_uppercase().as_str() {
        "F32"      => Ty::F32,
        "F16"      => Ty::F16,
        "Q4_0"     => Ty::Q4_0,
        "Q4_1"     => Ty::Q4_1,
        "Q5_0"     => Ty::Q5_0,
        "Q5_1"     => Ty::Q5_1,
        "Q8_0"     => Ty::Q8_0,
        "Q8_1"     => Ty::Q8_1,
        "Q2K"      => Ty::Q2K,
        "Q3K"      => Ty::Q3K,
        "Q4K"      => Ty::Q4K,
        "Q5K"      => Ty::Q5K,
        "Q6K"      => Ty::Q6K,
        "Q8K"      => Ty::Q8K,
        "IQ2XXS"   => Ty::IQ2XXS,
        "IQ2XS"    => Ty::IQ2XS,
        "IQ3XXS"   => Ty::IQ3XXS,
        "IQ1S"     => Ty::IQ1S,
        "IQ4NL"    => Ty::IQ4NL,
        "IQ3S"     => Ty::IQ3S,
        "IQ2S"     => Ty::IQ2S,
        "IQ4XS"    => Ty::IQ4XS,
        "I8"       => Ty::I8,
        "I16"      => Ty::I16,
        "I32"      => Ty::I32,
        "I64"      => Ty::I64,
        "F64"      => Ty::F64,
        "IQ1M"     => Ty::IQ1M,
        "BF16"     => Ty::BF16,
        "Q4_0_4_4" => Ty::Q4_0_4_4,
        "Q4_0_4_8" => Ty::Q4_0_4_8,
        "Q4_0_8_8" => Ty::Q4_0_8_8,
        _          => todo!(),
    }
}
