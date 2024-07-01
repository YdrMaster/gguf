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

    let (keys, _bytes) = metadata::scan(
        header.metadata_kv_count(),
        &file[size_of::<GGufFileHeader>()..],
    )
    .unwrap();

    for kv in keys {
        println!("{} {:?}", kv.key(), kv.ty());
    }
}
