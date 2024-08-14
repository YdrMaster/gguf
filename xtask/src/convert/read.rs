use super::{Content, DataPromise, MetaValue, Tensor};
use ggus::{GGuf, GGufError, GENERAL_ALIGNMENT};
use std::borrow::Cow;

impl<'a> Content<'a> {
    pub fn new(files: &[GGuf<'a>]) -> Self {
        let mut ans = Self {
            alignment: 0,
            meta_kvs: Default::default(),
            tensors: Default::default(),
        };

        for f in files {
            ans.alignment = ans.alignment.max(f.alignment);

            for (&k, kv) in &f.meta_kvs {
                if k != GENERAL_ALIGNMENT && !k.starts_with("split.") {
                    ans.meta_kvs.insert(
                        k.into(),
                        MetaValue {
                            ty: kv.ty(),
                            value: Cow::Borrowed(kv.value_bytes()),
                        },
                    );
                }
            }

            for (&name, tensor) in &f.tensors {
                let tensor = tensor.to_info();
                ans.tensors.insert(
                    name.into(),
                    Tensor {
                        ty: tensor.ty(),
                        shape: tensor.shape().to_vec(),
                        data: DataPromise::Borrowed(&f.data[tensor.offset()..][..tensor.nbytes()]),
                    },
                );
            }
        }

        ans
    }
}

pub(super) fn read_files<'a>(
    files: impl IntoIterator<Item = &'a [u8]> + 'a,
) -> Result<Vec<GGuf<'a>>, GGufError> {
    std::thread::scope(|s| {
        files
            .into_iter()
            .map(|data| s.spawn(|| GGuf::new(data)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect::<Result<Vec<_>, _>>()
    })
}
