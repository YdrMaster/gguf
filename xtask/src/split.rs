use crate::{file_info::print_file_info, gguf_file::GGufFile, loose_shards::LooseShards, YES};
use ggus::{GGufFileHeader, GGufMetaWriter, GGufSimulator, GENERAL_ALIGNMENT};
use std::{fs::File, path::PathBuf};

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
        let mut shards = vec![0usize];

        let mut simulator = GGufSimulator::new();
        simulator.write(source.meta_kvs.nbytes());
        if !source.meta_kvs.contains(GENERAL_ALIGNMENT) {
            simulator.write_alignment();
        }

        for info in source.tensors.iter() {
            match &mut *shards {
                [_] if no_tensor_first => {
                    simulator = GGufSimulator::new();
                    simulator.write_alignment();
                    simulator.write_tensor(&info);
                    shards.push(1);
                }
                [.., last] => {
                    simulator.write_tensor(&info);
                    if *last < max_tensors && simulator.written_bytes() < max_bytes {
                        *last += 1;
                    } else {
                        simulator = GGufSimulator::new();
                        simulator.write_alignment();
                        simulator.write_tensor(&info);
                        shards.push(1);
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
            let n_tensors = shards[0];
            let path = file_format.get(0).unwrap();
            let header = GGufFileHeader::new(3, n_tensors as _, filter_split as _);

            let mut writer = GGufMetaWriter::new(File::create(&path).unwrap(), header).unwrap();
            for kv in source.meta_kvs.kvs() {
                if !kv.key().starts_with("split.") {
                    writer
                        .write_meta_kv(kv.key(), kv.ty(), kv.value_bytes())
                        .unwrap();
                }
            }

            let mut writer = writer.finish();
            for t in &tensors[..n_tensors] {
                writer.write_tensor(t, source.data).unwrap();
            }
            let n_bytes = writer.finish().unwrap();

            println!("{YES}Shard is written to: \"{}\"", path.display());
            print_file_info(n_tensors, filter_split, n_bytes);
        }
        // 写入其他分片
        let mut tensors = &tensors[shards[0]..];
        for (i, n_tensors) in shards.into_iter().enumerate().skip(1) {
            let (current, others) = tensors.split_at(n_tensors);
            tensors = others;

            let path = file_format.get(i).unwrap();
            let header = GGufFileHeader::new(3, n_tensors as _, 1);

            let mut writer = GGufMetaWriter::new(File::create(&path).unwrap(), header).unwrap();
            writer.write_alignment(align).unwrap();

            let mut writer = writer.finish();
            for t in current {
                writer.write_tensor(t, source.data).unwrap();
            }
            let n_bytes = writer.finish().unwrap();

            println!("{YES}Shard is written to: \"{}\"", path.display());
            print_file_info(n_tensors, 1, n_bytes);
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
