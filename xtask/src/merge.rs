use crate::{
    gguf_file::{pad, GGufFile},
    loose_shards::LooseShards,
};
use ggus::{GGufFileHeader, GGufMetaDataValueType, GGufWriter};
use indexmap::{IndexMap, IndexSet};
use std::{fs::File, iter::zip, path::PathBuf, thread};

#[derive(Args, Default)]
pub struct MergeArgs {
    #[clap(long, short)]
    file: PathBuf,
}

impl MergeArgs {
    pub fn merge(self) {
        let shards = LooseShards::from(&*self.file);
        if shards.count() < 2 {
            println!("Model does not need to be merged.");
            return;
        }

        let mut files = Vec::new();
        for path in &shards {
            match File::open(&path) {
                Ok(file) => files.push(unsafe { memmap2::Mmap::map(&file).unwrap() }),
                Err(e) => {
                    eprintln!("Failed to open");
                    eprintln!("  file: {}", path.display());
                    eprintln!("  cause: {e}");
                    return;
                }
            }
        }

        let files = thread::scope(|s| {
            files
                .iter()
                .map(|data| s.spawn(|| GGufFile::new(data).unwrap()))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });

        let kvs = files
            .iter()
            .flat_map(|file| file.get_meta_kvs().kvs())
            .filter(|kv| {
                let key = kv.key();
                !key.starts_with("split.") && key != "general.alignment"
            })
            .collect::<IndexSet<_>>();
        let tensors = files
            .iter()
            .flat_map(|file| file.get_tensors_as_indexmap())
            .collect::<IndexMap<_, _>>();

        let out = File::create(shards.single_file()).unwrap();
        let header = GGufFileHeader::new(3, tensors.len() as _, (kvs.len() + 1) as _);
        let mut writer = GGufWriter::new(out, header).unwrap();

        let align = files
            .iter()
            .map(|file| file.get_meta_kvs().alignment())
            .max()
            .unwrap();

        writer
            .write_meta_kv(
                "general.alignment",
                GGufMetaDataValueType::U32,
                (align as u32).to_le_bytes(),
            )
            .unwrap();

        for kv in kvs {
            writer
                .write_meta_kv(kv.key(), kv.ty(), kv.value_bytes())
                .unwrap();
        }

        let mut cursor = 0;
        let mut paddings = Vec::with_capacity(tensors.len() + 1);
        paddings.push(0);

        for t in tensors.keys() {
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

        for ((t, data), padding) in zip(tensors, paddings) {
            for _ in 0..padding {
                writer.write(0u8).unwrap();
            }
            writer
                .write_bytes(&data[t.offset()..][..t.nbytes()])
                .unwrap();
        }
    }
}
