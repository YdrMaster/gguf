mod loose_shards;
mod show;

use clap::Parser;

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Show(args) => args.show(),
        Split => todo!(),
        Merge => todo!(),
    }
}

#[derive(Parser)]
#[clap(name = "gguf-utils")]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Show(show::ShowArgs),
    Split,
    Merge,
}
