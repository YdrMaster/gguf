use super::{Content, FileInfo, OutputConfig};
use ggus::{GGufFileHeader, GGufFileSimulator, GGufFileWriter};
use std::{fs::File, io, iter::zip, path::PathBuf, thread};

impl Content<'_> {
    pub fn write_files(self, out: OutputConfig) -> Result<Vec<FileInfo>, io::Error> {
        let Self {
            name,
            alignment,
            meta_kvs,
            tensors,
        } = self;
        let OutputConfig {
            dir,
            shard_max_tensor_count,
            shard_max_file_size,
            shard_no_tensor_first,
        } = out;

        // 规划分片方案

        let mut simulator = GGufFileSimulator::with_alignment(alignment);
        for (k, v) in &meta_kvs {
            simulator.write_meta_kv(k, v.ty, &v.value);
        }

        let mut simulator = simulator.finish();
        let mut shards = vec![vec![]];
        for (name, tensor) in tensors {
            match &mut *shards {
                [_] if shard_no_tensor_first => {
                    simulator = GGufFileSimulator::with_alignment(alignment).finish();
                    simulator.write_tensor(&name, tensor.ty, &tensor.shape);
                    shards.push(vec![(name, tensor)]);
                }
                [.., current] => {
                    simulator.write_tensor(&name, tensor.ty, &tensor.shape);
                    if current.len() < shard_max_tensor_count
                        && simulator.written_bytes() < shard_max_file_size.nbytes()
                    {
                        current.push((name, tensor));
                    } else {
                        simulator = GGufFileSimulator::with_alignment(alignment).finish();
                        simulator.write_tensor(&name, tensor.ty, &tensor.shape);
                        shards.push(vec![(name, tensor)]);
                    }
                }
                [] => unreachable!(),
            }
        }

        // 生成迭代器

        let meta_kvs = &meta_kvs;
        let dir = dir.unwrap_or_else(|| std::env::current_dir().unwrap());
        let path = name
            .split_n(shards.len())
            .map(|name| dir.join(name.to_string()));

        // 并行写入文件

        std::fs::create_dir_all(&dir)?;
        thread::scope(|s| {
            zip(shards, path)
                .enumerate()
                .map(|(i, (tensors, path))| {
                    s.spawn(move || -> Result<FileInfo, io::Error> {
                        let path = find_path(path);

                        let n_meta_kvs = if i == 0 { meta_kvs.len() + 1 } else { 1 };
                        let n_tensors = tensors.len();
                        let header = GGufFileHeader::new(3, n_tensors as _, n_meta_kvs as _);

                        let mut writer = GGufFileWriter::new(File::create(&path)?, header)?;
                        writer.write_alignment(alignment)?;
                        if i == 0 {
                            for (k, v) in meta_kvs {
                                writer.write_meta_kv(k, v.ty, &v.value)?;
                            }
                        }

                        let mut writer = writer.finish();
                        for (name, tensor) in tensors {
                            writer.write_tensor(&name, tensor.ty, &tensor.shape, tensor.data)?;
                        }
                        writer.finish().map(|n_bytes| FileInfo {
                            path,
                            n_tensors,
                            n_meta_kvs,
                            n_bytes,
                        })
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|j| j.join().unwrap())
                .collect::<Result<Vec<_>, _>>()
        })
    }
}

/// 找到一个未被占用的文件名
fn find_path(path: PathBuf) -> PathBuf {
    if !path.exists() {
        return path;
    }

    let dir = path.parent().unwrap();
    let name = path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .strip_suffix(".gguf")
        .unwrap();

    for i in 1..usize::MAX {
        let path = dir.join(format!("{name} ({i}).gguf"));
        if !path.exists() {
            return path;
        }
    }

    unreachable!()
}
