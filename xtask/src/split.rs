use crate::{
    file_info::print_file_info,
    gguf_file::{pad, GGufFile},
    loose_shards::LooseShards,
    YES,
};
use ggus::{GGufFileHeader, GGufTensorInfo, GGufWriter};
use std::{
    fs::File,
    io::{Result as IoResult, Write},
    iter::zip,
    mem::size_of,
    path::PathBuf,
};

#[derive(Args, Default)]
pub struct SplitArgs {
    /// File to split
    file: PathBuf,
    /// Output directory for splited shards
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
}

impl SplitArgs {
    pub fn split(self) {
        // 解析参数
        let Self {
            file,
            output_dir,
            max_tensors,
            max_bytes,
            no_tensor_first,
        } = self;
        let shards = LooseShards::from(&*file);
        if shards.count() > 1 {
            println!("Model has already been splited");
            return;
        }
        let max_tensors = max_tensors.unwrap_or(usize::MAX);
        let max_bytes = match max_bytes {
            Some(s) => match s.trim().as_bytes() {
                [num @ .., b'G'] => parse_size_num(num, 30),
                [num @ .., b'M'] => parse_size_num(num, 20),
                [num @ .., b'K'] => parse_size_num(num, 10),
                num => parse_size_num(num, 0),
            }
            .unwrap_or_else(|| panic!("Invalid max bytes format: \"{s}\"")),
            None => usize::MAX,
        };
        // 打开文件
        let file = File::open(&file)
            .map_err(|e| {
                eprintln!("Failed to open");
                eprintln!("  file: {}", file.display());
                eprintln!("  cause: {e}");
            })
            .unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let num_tensors =
            unsafe { mmap.as_ptr().cast::<GGufFileHeader>().read() }.tensor_count as usize;
        if mmap.len() <= max_bytes && num_tensors <= max_tensors {
            println!("Model is already small enough");
            return;
        }
        // 解析文件
        let source = GGufFile::new(&mmap).unwrap();
        // 生成分片方案
        let mut file_format = shards;
        let first_size = source.header.nbytes() + source.meta_kvs.nbytes();
        let others_size = others_size();
        let mut shards = vec![ShardMeta {
            n_tensors: 0,
            n_bytes: first_size,
        }];
        for info in source.tensors.iter() {
            let tensor_size = tensor_size(&info);
            match &mut *shards {
                [_] if no_tensor_first => shards.push(ShardMeta {
                    n_tensors: 1,
                    n_bytes: others_size + tensor_size,
                }),
                [.., last] => {
                    if last.n_tensors < max_tensors && last.n_bytes + tensor_size < max_bytes {
                        last.n_tensors += 1;
                        last.n_bytes += tensor_size;
                    } else {
                        shards.push(ShardMeta {
                            n_tensors: 1,
                            n_bytes: others_size + tensor_size,
                        });
                    }
                }
                [] => unreachable!(),
            }
        }
        // 准备写入
        let filter_split = source
            .meta_kvs
            .keys()
            .filter(|k| !k.starts_with("split."))
            .count();
        let align = source.meta_kvs.alignment();
        let tensors = source.tensors.iter().collect::<Vec<_>>();
        file_format.shards_count = shards.len();
        if let Some(dir) = output_dir {
            if let Err(e) = std::fs::create_dir_all(&dir) {
                panic!("Failed to create output directory: {e}");
            }
            file_format.dir = dir;
        }
        // 写入第一个分片
        {
            let n_tensors = shards[0].n_tensors;
            let path = file_format.get(0).unwrap();
            let header = GGufFileHeader::new(3, n_tensors as _, filter_split as _);
            let mut writer = GGufWriter::new(File::create(&path).unwrap(), header).unwrap();
            for kv in source.meta_kvs.kvs() {
                if !kv.key().starts_with("split.") {
                    writer
                        .write_meta_kv(kv.key(), kv.ty(), kv.value_bytes())
                        .unwrap();
                }
            }
            write_tensors(&mut writer, &tensors[..n_tensors], align, &source);

            println!("{YES}Shard is written to: \"{}\"", path.display());
            print_file_info(n_tensors, filter_split, writer.written_bytes());
        }
        // 写入其他分片
        let mut used_tensors = shards[0].n_tensors;
        for (i, shard) in shards.into_iter().enumerate().skip(1) {
            let n_tensors = shard.n_tensors;
            let path = file_format.get(i).unwrap();
            let header = GGufFileHeader::new(3, n_tensors as _, 1);
            let mut writer = GGufWriter::new(File::create(&path).unwrap(), header).unwrap();
            writer.write_alignment(align).unwrap();

            write_tensors(
                &mut writer,
                &tensors[used_tensors..][..n_tensors],
                align,
                &source,
            );
            used_tensors += n_tensors;

            println!("{YES}Shard is written to: \"{}\"", path.display());
            print_file_info(n_tensors, 1, writer.written_bytes());
        }
    }
}

fn parse_size_num(num: &[u8], k: usize) -> Option<usize> {
    std::str::from_utf8(num)
        .ok()?
        .parse::<usize>()
        .ok()
        .map(|n| n << k)
}

struct NWriter;
impl Write for NWriter {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

fn others_size() -> usize {
    let mut writer = GGufWriter::new(NWriter, GGufFileHeader::new(3, 0, 0)).unwrap();
    writer.write_alignment(0).unwrap();
    writer.written_bytes()
}

fn tensor_size(info: &GGufTensorInfo) -> usize {
    let mut writer = GGufWriter::new(NWriter, GGufFileHeader::new(3, 0, 0)).unwrap();
    writer
        .write_tensor_info(info.name(), info.shape(), info.ggml_type(), info.offset())
        .unwrap();
    writer.written_bytes() + info.nbytes() - size_of::<GGufFileHeader>()
}

#[derive(Clone, Copy, Debug)]
struct ShardMeta {
    n_tensors: usize,
    n_bytes: usize,
}

fn write_tensors<T: Write>(
    writer: &mut GGufWriter<T>,
    tensors: &[GGufTensorInfo],
    align: usize,
    source: &GGufFile,
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
            .write_bytes(&source.data[t.offset()..][..t.nbytes()])
            .unwrap();
    }
}
