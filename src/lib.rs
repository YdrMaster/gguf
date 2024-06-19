mod header;
mod metadata;
mod name;

pub use header::GGufFileHeader;
pub use metadata::{GGufMetaDataValueType, MetaReader};
pub use name::GGufFileName;

#[test]
fn test_gguf() {
    use std::{fs::File, mem::size_of};

    let Some(args) = std::option_env!("ARGS") else {
        return;
    };

    let file = File::open(args).unwrap();
    let file = unsafe { memmap2::Mmap::map(&file) }.unwrap();

    let header = unsafe { file.as_ptr().cast::<GGufFileHeader>().read() };
    println!("{header:?} {}", header.is_magic_correct());

    fn read_metadata(dt: GGufMetaDataValueType, buf: &mut MetaReader) {
        match dt {
            GGufMetaDataValueType::U8 => {
                let val = buf.read::<u8>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::I8 => {
                let val = buf.read::<i8>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::U16 => {
                let val = buf.read::<u16>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::I16 => {
                let val = buf.read::<i16>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::U32 => {
                let val = buf.read::<u32>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::I32 => {
                let val = buf.read::<i32>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::F32 => {
                let val = buf.read::<f32>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::BOOL => {
                let val = buf.read::<bool>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::STRING => {
                let val = buf.read_str().unwrap();
                println!("#\"{val}\"#");
            }
            GGufMetaDataValueType::ARRAY => {
                let (dt, len) = buf.read_arr_header().unwrap();
                println!("[{dt:?}; {len}]");
                for _ in 0..len {
                    read_metadata(dt, buf);
                }
            }
            GGufMetaDataValueType::U64 => {
                let val = buf.read::<u64>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::I64 => {
                let val = buf.read::<i64>().unwrap();
                println!("{val}");
            }
            GGufMetaDataValueType::F64 => {
                let val = buf.read::<f64>().unwrap();
                println!("{val}");
            }
        }
    }

    let mut buf = MetaReader::new(&file[size_of::<GGufFileHeader>()..]);
    for _ in 0..header.metadata_kv_count() {
        let (key, ty) = buf.read_kv_header().unwrap();
        print!("{key}: ");
        read_metadata(ty, &mut buf);
    }
}
