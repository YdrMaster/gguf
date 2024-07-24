mod gguf_file;
mod loose_shards;
mod merge;
mod show;
mod split;

#[macro_use]
extern crate clap;
use clap::Parser;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Show(args) => args.show(),
        Split(args) => args.split(),
        Merge(args) => args.merge(),
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
    Split(split::SplitArgs),
    Merge(merge::MergeArgs),
}
