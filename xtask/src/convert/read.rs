use crate::gguf_file::{GGufError, GGufFile};

pub(super) fn read_files<'a>(
    files: impl IntoIterator<Item = &'a [u8]> + 'a,
) -> Result<Vec<GGufFile<'a>>, GGufError> {
    std::thread::scope(|s| {
        files
            .into_iter()
            .map(|data| s.spawn(|| GGufFile::new(data)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect::<Result<Vec<_>, _>>()
    })
}
