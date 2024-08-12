use crate::{file_info::FileInfo, shards::Shards};
use ggus::{GGufFileHeader, GGufMetaKV, GGufMetaWriter, GGufSimulator, GGufTensorInfo};
use std::{fs::File, io, path::Path, thread};

#[allow(clippy::too_many_arguments)]
pub(super) fn write_files<T: AsRef<[u8]> + Sync>(
    meta_kvs: &[GGufMetaKV],
    tensors: &[(GGufTensorInfo, T)],
    output_dir: &Path,
    output_name: &str,
    align: usize,
    split_tensor_count: usize,
    split_file_size: usize,
    split_no_tensor_first: bool,
) -> Result<Vec<FileInfo>, io::Error> {
    // 规划分片方案

    let mut simulator = GGufSimulator::with_alignment(align);
    for kv in meta_kvs {
        simulator.write_meta_kv(kv.key(), kv.ty(), kv.value_bytes());
    }

    let mut shards = vec![0usize];
    for (t, _) in tensors {
        match &mut *shards {
            [_] if split_no_tensor_first => {
                simulator = GGufSimulator::with_alignment(align);
                simulator.write_tensor(t);
                shards.push(1);
            }
            [.., count] => {
                simulator.write_tensor(t);
                if *count < split_tensor_count && simulator.written_bytes() < split_file_size {
                    *count += 1;
                } else {
                    simulator = GGufSimulator::with_alignment(align);
                    simulator.write_tensor(t);
                    shards.push(1);
                }
            }
            [] => unreachable!(),
        }
    }

    // 生成迭代器

    let mut tensors = tensors;
    let iter = shards
        .iter()
        .map(|&n| {
            let (head, tail) = tensors.split_at(n);
            tensors = tail;
            head
        })
        .zip(Shards {
            dir: output_dir,
            name: output_name,
            index: 0,
            count: shards.len(),
            format: 5,
        })
        .enumerate();

    // 并行写入文件

    std::fs::create_dir_all(output_dir)?;
    thread::scope(|s| {
        iter.map(|(i, (tensors, path))| {
            s.spawn(move || -> Result<FileInfo, io::Error> {
                let n_meta_kvs = if i == 0 { meta_kvs.len() + 1 } else { 1 };
                let header = GGufFileHeader::new(3, tensors.len() as _, n_meta_kvs as _);

                let mut writer = GGufMetaWriter::new(File::create(&path)?, header)?;
                writer.write_alignment(align)?;
                if i == 0 {
                    for kv in meta_kvs {
                        writer.write_meta_kv(kv.key(), kv.ty(), kv.value_bytes())?;
                    }
                }

                let mut writer = writer.finish();
                for (t, data) in tensors {
                    writer.write_tensor(t, data.as_ref())?;
                }
                writer.finish().map(|n_bytes| FileInfo {
                    path,
                    n_tensors: tensors.len(),
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
