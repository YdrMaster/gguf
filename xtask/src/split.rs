use std::path::PathBuf;

#[derive(Args, Default)]
pub struct SplitArgs {
    #[clap(long, short)]
    file: PathBuf,
}

impl SplitArgs {
    pub fn split(self) {
        todo!()
    }
}
