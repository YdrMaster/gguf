use crate::gguf_file::{pad, GGufError, GGufFile};
use ggus::{GGufFileHeader, GGufMetaDataValueType, GGufMetaKVPairs, GGufWriter};
use std::{fs::File, path::PathBuf};

const GGUF_VERSION: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const LLM_KV_SPLIT_NO: &str = "split.no";
const LLM_KV_SPLIT_COUNT: &str = "split.count";
const LLM_KV_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

#[derive(Args, Default)]
pub struct SplitArgs {
    #[clap(long)]
    input: PathBuf,
    #[clap(long)]
    output: Option<String>,
    // default 128 tensors
    #[clap(long)]
    split_max_tensors: Option<u64>,
    #[clap(long)]
    split_max_size: Option<String>,
    #[clap(long)]
    no_tensor_first_split: bool,
    n_bytes_split: u64,
    n_split_tensors: u64,
}

#[derive(Clone)]
struct GGufFileInfo<'a> {
    output_path: String,
    header: GGufFileHeader,
    meta_kvs: GGufMetaKVPairs<'a>,
    new_kv_tuples: Vec<(String, GGufMetaDataValueType, u64)>,
}

impl<'a> GGufFileInfo<'a> {
    fn new_empty() -> Self {
        let header = GGufFileHeader::new(GGUF_VERSION, 0, 0);
        let meta_kvs = GGufMetaKVPairs::new(0);
        let new_kv_tuples: Vec<(String, GGufMetaDataValueType, u64)> = Vec::new();
        let output_path = "".to_string();
        Self {
            output_path,
            header,
            meta_kvs,
            new_kv_tuples,
        }
    }
}

impl SplitArgs {
    pub fn split(self) {
        let file = File::open(&self.input)
            .map_err(|e| {
                println!("Failed to open");
                eprintln!("  file: {}", self.input.display());
                eprintln!("  cause: {e}");
            })
            .unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let ctx_gguf: GGufFile = GGufFile::new(&mmap).unwrap();

        let align = ctx_gguf.get_meta_kvs().alignment();
        let ggufs = self.split_strategy(ctx_gguf.clone()).unwrap();

        let tensors = ctx_gguf.get_tensors_as_indexmap();
        let mut tensor_iter: indexmap::map::Iter<ggus::GGufTensorInfo, &[u8]> = tensors.iter();

        for gguf in ggufs {
            let out = File::create(gguf.output_path).unwrap();

            let header = gguf.header;
            let tensor_count: u64 = header.tensor_count;
            let mut writer = GGufWriter::new(out, header).unwrap();

            let kvs = gguf.meta_kvs.kvs();
            for kv in kvs {
                if kv.key() == LLM_KV_SPLIT_TENSORS_COUNT
                    || kv.key() == LLM_KV_SPLIT_COUNT
                    || kv.key() == LLM_KV_SPLIT_NO
                {
                    continue;
                }
                writer
                    .write_meta_kv(kv.key(), kv.ty(), kv.value_bytes())
                    .unwrap();
            }

            for kv in gguf.new_kv_tuples {
                match kv.1 {
                    GGufMetaDataValueType::U16 => writer
                        .write_meta_kv(kv.0, kv.1, (kv.2 as u16).to_le_bytes())
                        .unwrap(),
                    GGufMetaDataValueType::I32 => writer
                        .write_meta_kv(kv.0, kv.1, (kv.2 as i32).to_le_bytes())
                        .unwrap(),
                    _ => (),
                }
            }

            let mut cursor = 0;
            let mut paddings = Vec::with_capacity(tensor_count as usize + 1);
            paddings.push(0);

            let mut tensor_info_iter: indexmap::map::Iter<ggus::GGufTensorInfo, &[u8]> =
                tensor_iter.clone();

            for _ in 0..tensor_count {
                let (tensor_info, _) = tensor_info_iter.next().unwrap();

                writer
                    .write_tensor_info(
                        tensor_info.name(),
                        tensor_info.shape(),
                        tensor_info.ggml_type(),
                        cursor,
                    )
                    .unwrap();

                cursor += tensor_info.nbytes();
                let padding = pad(cursor, align);

                cursor += padding;
                paddings.push(padding);
            }

            paddings.pop();
            if !paddings.is_empty() {
                paddings[0] = pad(writer.written_bytes(), GGUF_DEFAULT_ALIGNMENT);
            }

            for padding in paddings {
                for _ in 0..padding {
                    writer.write(0u8).unwrap();
                }

                let (t, data) = tensor_iter.next().unwrap();
                writer
                    .write_bytes(&data[t.offset()..][..t.nbytes()])
                    .unwrap();
            }

            let end_padding = pad(writer.written_bytes(), GGUF_DEFAULT_ALIGNMENT);
            for _ in 0..end_padding {
                writer.write(0u8).unwrap();
            }
        }
    }

    fn split_strategy(mut self, ctx_gguf: GGufFile) -> Result<Vec<GGufFileInfo>, GGufError> {
        use GGufMetaDataValueType as ty;
        match (self.split_max_size.clone(), self.split_max_tensors) {
            (Some(_), Some(_)) => {
                return Err(GGufError::SplitModeRepeated);
            }
            (Some(max_size), None) => {
                fn parse_split_max_size(split_max_size: String) -> Option<u64> {
                    if split_max_size.is_empty() {
                        return None;
                    }
                    let symbol = split_max_size.chars().last().unwrap();
                    let len = split_max_size.len();
                    let size = split_max_size[..len - 1].parse::<u64>();
                    match size {
                        Ok(num) => {
                            if symbol == 'M' {
                                Some(num * 1000 * 1000)
                            } else if symbol == 'G' {
                                Some(num * 1000 * 1000 * 1000)
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    }
                }

                match parse_split_max_size(max_size) {
                    Some(size) => {
                        self.n_bytes_split = size;
                    }
                    None => {
                        // 错误的格式
                        return Err(GGufError::FileSizeError);
                    }
                }
            }
            _ => match self.split_max_tensors {
                Some(tensors) => {
                    self.n_bytes_split = 0;
                    self.n_split_tensors = tensors;
                }
                None => {
                    self.n_bytes_split = 0;
                    self.n_split_tensors = 128;
                }
            },
        }

        let tensors = ctx_gguf.get_tensors_as_indexmap();

        let mut ggufs: Vec<GGufFileInfo> = Vec::new();
        let n_tensors: u64 = ctx_gguf.get_header().tensor_count;

        let setup_gguf_file = |i_split: u64, n_tensors: u64| {
            let mut gguf_file = GGufFileInfo::new_empty();
            gguf_file
                .new_kv_tuples
                .push((LLM_KV_SPLIT_NO.to_string(), ty::U16, i_split));
            gguf_file
                .new_kv_tuples
                .push((LLM_KV_SPLIT_COUNT.to_string(), ty::U16, 0));
            gguf_file.new_kv_tuples.push((
                LLM_KV_SPLIT_TENSORS_COUNT.to_string(),
                ty::I32,
                n_tensors,
            ));
            gguf_file.header.metadata_kv_count += 3;
            gguf_file
        };

        let mut i_split: u64 = 1;
        let mut gguf_file = setup_gguf_file(i_split, n_tensors);
        gguf_file.meta_kvs = ctx_gguf.get_meta_kvs().clone();

        if gguf_file.meta_kvs.get(LLM_KV_SPLIT_NO).is_some() {
            gguf_file.header.metadata_kv_count -= 1;
        }
        if gguf_file.meta_kvs.get(LLM_KV_SPLIT_COUNT).is_some() {
            gguf_file.header.metadata_kv_count -= 1;
        }
        if gguf_file.meta_kvs.get(LLM_KV_SPLIT_TENSORS_COUNT).is_some() {
            gguf_file.header.metadata_kv_count -= 1;
        }

        gguf_file.header.metadata_kv_count += ctx_gguf.get_header().metadata_kv_count;

        if self.no_tensor_first_split {
            ggufs.push(gguf_file);
            gguf_file = setup_gguf_file(i_split, n_tensors);
        }

        let mut curr_tensors_size: u64 = 0;
        let mut i_tensor = 0;

        for t in tensors.keys() {
            i_tensor += 1;
            let tensor_size = t.nbytes();
            let n_bytes = (pad(tensor_size, GGUF_DEFAULT_ALIGNMENT) + tensor_size) as u64;
            let next_tensor_size = curr_tensors_size + n_bytes;

            if self.should_split(i_tensor, n_tensors, next_tensor_size) {
                ggufs.push(gguf_file.clone());
                i_tensor = 0;
                i_split += 1;
                gguf_file = setup_gguf_file(i_split, n_tensors);
                curr_tensors_size = n_bytes;
            } else {
                curr_tensors_size = next_tensor_size;
            }
            gguf_file.header.tensor_count += 1;
        }

        ggufs.push(gguf_file);

        let tensor_count = ggufs.len() as u64;
        let output_path = &self.output.unwrap();
        let mut index = 0;
        while index < ggufs.len() {
            ggufs[index].output_path = split_path(output_path, index + 1, ggufs.len());
            ggufs[index].new_kv_tuples[1] = (LLM_KV_SPLIT_COUNT.to_string(), ty::U16, tensor_count);
            index += 1;
        }

        Ok(ggufs)
    }

    fn should_split(&self, i_tensor: u64, n_tensors: u64, next_size: u64) -> bool {
        if self.n_bytes_split > 0 {
            next_size > self.n_bytes_split
        } else {
            i_tensor > 0 && i_tensor < n_tensors && i_tensor % self.n_split_tensors == 0
        }
    }
}

fn split_path(path_prefix: &String, split_no: usize, split_count: usize) -> String {
    format!("{}-{:05}-of-{:05}.gguf", path_prefix, split_no, split_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let input = PathBuf::from("/home/qinyiqun/gguf/xtask/src/test/rust/rust_ori.gguf");
        let output = Some("/home/qinyiqun/gguf/xtask/src/test/rust/rust_ori_oi".to_string());
        let split_args = SplitArgs {
            input,
            output,
            split_max_tensors: None,
            split_max_size: Some("300M".to_string()),
            no_tensor_first_split: true,
            n_bytes_split: 0,
            n_split_tensors: 0,
        };

        split_args.split();
    }
}
