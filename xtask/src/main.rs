mod file_info;
mod filter;
mod gguf_file;
mod loose_shards;
mod merge;
mod name_pattern;
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
        Filter(args) => args.filter(),
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
    Filter(filter::FilterArgs),
}

const YES: &str = "✔️  ";
const ERR: &str = "❌  ";

use gguf_file::pad;
use ggus::{GGufTensorInfo, GGufWriter};
use std::{io::Write, iter::zip};

fn write_tensors<T: Write>(
    writer: &mut GGufWriter<T>,
    tensors: &[GGufTensorInfo],
    align: usize,
    data: &[u8],
) {
    if tensors.is_empty() {
        return;
    }

    let mut cursor = 0;
    let mut paddings = Vec::with_capacity(tensors.len() + 1);
    paddings.push(0);

    for t in tensors {
        writer
            .write_tensor_info(t.name(), t.shape(), t.ggml_type(), cursor)
            .unwrap();

        cursor += t.nbytes();
        let padding = pad(cursor, align);

        cursor += padding;
        paddings.push(padding);
    }

    paddings.pop();
    paddings[0] = pad(writer.written_bytes(), align);

    for (t, padding) in zip(tensors, paddings) {
        for _ in 0..padding {
            writer.write(0u8).unwrap();
        }
        writer
            .write_bytes(&data[t.offset()..][..t.nbytes()])
            .unwrap();
    }
}
