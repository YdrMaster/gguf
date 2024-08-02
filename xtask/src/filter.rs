use crate::{
    file_info::print_file_info, gguf_file::GGufFile, name_pattern::compile_patterns, write_tensors,
    YES,
};
use ggus::{GGufFileHeader, GGufWriter};
use std::{fs::File, path::PathBuf};

#[derive(Args, Default)]
pub struct FilterArgs {
    /// The file to filter
    file: PathBuf,
    /// Output directory for merged file
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
    /// Meta to keep
    #[clap(long, short = 'm', default_value = "*")]
    filter_meta: String,
    /// Tensors to keep
    #[clap(long, short = 't', default_value = "*")]
    filter_tensor: String,
}

impl FilterArgs {
    pub fn filter(self) {
        let Self {
            file: file_path,
            output_dir,
            filter_meta,
            filter_tensor,
        } = self;
        // 打开文件
        let file = File::open(&file_path)
            .map_err(|e| {
                eprintln!("Failed to open");
                eprintln!("  file: {}", file_path.display());
                eprintln!("  cause: {e}");
            })
            .unwrap();
        let file = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let file = GGufFile::new(&file).unwrap();
        // 编译过滤器
        let filter_meta = compile_patterns(&filter_meta);
        let filter_tensor = compile_patterns(&filter_tensor);
        // 过滤元信息和张量
        let meta_kvs = file
            .meta_kvs
            .kvs()
            .filter(move |t| filter_meta.is_match(t.key()))
            .collect::<Vec<_>>();
        let tensors = file
            .tensors
            .iter()
            .filter(move |t| filter_tensor.is_match(t.name()))
            .collect::<Vec<_>>();
        if meta_kvs.len() == file.meta_kvs.len() && tensors.len() == file.tensors.len() {
            eprintln!("Nothing to filter");
            return;
        }
        // 创建存储目录
        let mut out = file_path;
        if let Some(dir) = output_dir {
            if let Err(e) = std::fs::create_dir_all(&dir) {
                panic!("Failed to create output directory: {e}");
            }
            out = dir.join(out.file_name().unwrap());
        }
        // 写入文件
        out = out.with_extension("part.gguf");

        let header = GGufFileHeader::new(3, tensors.len() as _, meta_kvs.len() as _);
        let mut writer = GGufWriter::new(File::create(&out).unwrap(), header).unwrap();
        for kv in &meta_kvs {
            writer
                .write_meta_kv(kv.key(), kv.ty(), kv.value_bytes())
                .unwrap();
        }
        write_tensors(&mut writer, &tensors, file.meta_kvs.alignment(), file.data);

        println!("{YES}Shard is written to: \"{}\"", out.display());
        print_file_info(tensors.len(), meta_kvs.len(), writer.written_bytes());
    }
}
