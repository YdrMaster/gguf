use super::Content;
use crate::{file_info::FileInfo, shards::Shards};
use ggus::{GGufFileHeader, GGufMetaWriter, GGufSimulator};
use std::{fs::File, io, iter::zip, path::Path, thread};

impl Content<'_> {
    pub fn write_files(
        self,
        output_dir: &Path,
        output_name: &str,
        split_tensor_count: usize,
        split_file_size: usize,
        split_no_tensor_first: bool,
    ) -> Result<Vec<FileInfo>, io::Error> {
        let Self {
            alignment,
            meta_kvs,
            tensors,
        } = self;

        // 规划分片方案

        let mut simulator = GGufSimulator::with_alignment(alignment);
        for (k, v) in &meta_kvs {
            simulator.write_meta_kv(k, v.ty, &v.value);
        }

        let mut shards = vec![vec![]];
        for (name, tensor) in tensors {
            match &mut *shards {
                [_] if split_no_tensor_first => {
                    simulator = GGufSimulator::with_alignment(alignment);
                    simulator.write_tensor(&name, tensor.ty, &tensor.shape);
                    shards.push(vec![(name, tensor)]);
                }
                [.., current] => {
                    simulator.write_tensor(&name, tensor.ty, &tensor.shape);
                    if current.len() < split_tensor_count
                        && simulator.written_bytes() < split_file_size
                    {
                        current.push((name, tensor));
                    } else {
                        simulator = GGufSimulator::with_alignment(alignment);
                        simulator.write_tensor(&name, tensor.ty, &tensor.shape);
                        shards.push(vec![(name, tensor)]);
                    }
                }
                [] => unreachable!(),
            }
        }

        // 生成迭代器

        let meta_kvs = &meta_kvs;
        let names = Shards {
            dir: output_dir,
            name: output_name,
            index: 0,
            count: shards.len(),
            format: 5,
        };

        // 并行写入文件

        std::fs::create_dir_all(output_dir)?;
        thread::scope(|s| {
            zip(shards, names)
                .enumerate()
                .map(|(i, (tensors, path))| {
                    s.spawn(move || -> Result<FileInfo, io::Error> {
                        let n_meta_kvs = if i == 0 { meta_kvs.len() + 1 } else { 1 };
                        let n_tensors = tensors.len();
                        let header = GGufFileHeader::new(3, n_tensors as _, n_meta_kvs as _);

                        let mut writer = GGufMetaWriter::new(File::create(&path)?, header)?;
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
