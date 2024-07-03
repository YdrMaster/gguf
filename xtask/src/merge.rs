use ggus::{
    GGufFileHeader, GGufMetaDataValueType, GGufMetaKVPairs, GGufReadError, GGufTensors, GGufWriter,
};
use indexmap::IndexSet;

use crate::loose_shards::LooseShards;
use std::{fs::File, path::PathBuf};

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
        let files = files;

        let files = files
            .iter()
            .map(|data| GGufFile::new(data).unwrap())
            .collect::<Vec<_>>();

        let tensor_count = files.iter().map(|file| file.tensors.len()).sum::<usize>();
        let kvs = files
            .iter()
            .flat_map(|file| file.meta_kvs.kvs())
            .filter(|kv| {
                let key = kv.key();
                !key.starts_with("split.")
                    && !key.starts_with("shard.")
                    && key != "general.alignment"
            })
            .collect::<IndexSet<_>>();

        let out = File::create(shards.single_file()).unwrap();
        let header = GGufFileHeader::new(3, tensor_count as _, (kvs.len() + 1) as _);
        let mut writer: GGufWriter<File> = GGufWriter::new(out, header).unwrap();

        let alignment = files
            .iter()
            .map(|file| file.meta_kvs.alignment())
            .max()
            .unwrap();

        writer
            .write_meta_kv(
                "general.alignment",
                GGufMetaDataValueType::U64,
                (alignment as u64).to_le_bytes(),
            )
            .unwrap();

        for kv in kvs {
            writer
                .write_meta_kv(kv.key(), kv.ty(), kv.value_bytes())
                .unwrap();
        }

        drop(writer);
        todo!()
    }
}

struct GGufFile<'a> {
    meta_kvs: GGufMetaKVPairs<'a>,
    tensors: GGufTensors<'a>,
    data: &'a [u8],
}

#[derive(Debug)]
enum GGufError<'a> {
    MagicMismatch,
    EndianNotSupport,
    VersionNotSupport,
    Reading(GGufReadError<'a>),
}

impl<'a> GGufFile<'a> {
    fn new(data: &'a [u8]) -> Result<Self, GGufError<'a>> {
        let header = unsafe { data.as_ptr().cast::<GGufFileHeader>().read() };
        if !header.is_magic_correct() {
            return Err(GGufError::MagicMismatch);
        }
        if !header.is_native_endian() {
            return Err(GGufError::EndianNotSupport);
        }
        if header.version != 3 {
            return Err(GGufError::VersionNotSupport);
        }

        let cursor = header.nbytes();
        let meta_kvs = GGufMetaKVPairs::scan(header.metadata_kv_count, &data[cursor..])
            .map_err(GGufError::Reading)?;

        let cursor = cursor + meta_kvs.nbytes();
        let tensors =
            GGufTensors::scan(header.tensor_count, &data[cursor..]).map_err(GGufError::Reading)?;

        let align = meta_kvs.alignment();
        let cursor = (cursor + tensors.nbytes() + align - 1) / align * align;
        Ok(Self {
            meta_kvs,
            tensors,
            data: &data[cursor..],
        })
    }
}
