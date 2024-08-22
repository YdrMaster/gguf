mod convert;
mod filter;
mod merge;
mod set_meta;
mod show;
mod split;
mod utils;

#[macro_use]
extern crate clap;
use clap::Parser;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Show(args) => args.show(),
        Split(args) => args.split(),
        Merge(args) => args.merge(),
        Filter(args) => args.filter(),
        Convert(args) => args.convert(),
        SetMeta(args) => args.set_meta(),
    }
}

/// gguf-utils is a command-line tool for working with gguf files.
#[derive(Parser)]
#[clap(version)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show the contents of gguf files
    Show(show::ShowArgs),
    /// Split gguf files into shards
    Split(split::SplitArgs),
    /// Merge shards into a single gguf file
    Merge(merge::MergeArgs),
    /// Filter gguf files based on wildcard patterns
    Filter(filter::FilterArgs),
    /// Convert gguf files to different format
    Convert(convert::ConvertArgs),
    /// Set metadata of gguf files
    SetMeta(set_meta::SetMetaArgs),
}
