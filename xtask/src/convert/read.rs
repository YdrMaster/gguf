use ggus::{GGuf, GGufError};

pub(super) fn read_files<'a>(
    files: impl IntoIterator<Item = &'a [u8]> + 'a,
) -> Result<Vec<GGuf<'a>>, GGufError> {
    std::thread::scope(|s| {
        files
            .into_iter()
            .map(|data| s.spawn(|| GGuf::scan(data)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect::<Result<Vec<_>, _>>()
    })
}
