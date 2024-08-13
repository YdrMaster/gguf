use std::sync::Arc;

use super::{Content, DataPromise, MetaValue, Tensor};
use ggus::{GGuf, GGufError, GENERAL_ALIGNMENT};

impl<'a> Content<'a> {
    pub fn new(files: &[GGuf<'a>]) -> Self {
        let mut ans = Self {
            alignment: 0,
            meta_kvs: Default::default(),
            tensors: Default::default(),
        };

        for f in files {
            ans.alignment = ans.alignment.max(f.meta_kvs.alignment());

            for kv in f.meta_kvs.kvs() {
                let key = kv.key();
                if key != GENERAL_ALIGNMENT && !key.starts_with("split.") {
                    ans.meta_kvs.insert(
                        key.into(),
                        MetaValue {
                            ty: kv.ty(),
                            value: DataPromise(Arc::new(kv.value_bytes())),
                        },
                    );
                }
            }

            for tensor in f.tensors.iter() {
                ans.tensors.insert(
                    tensor.name().into(),
                    Tensor {
                        ty: tensor.ggml_type(),
                        shape: tensor.shape().to_vec(),
                        data: DataPromise(Arc::new(&f.data[tensor.offset()..][..tensor.nbytes()])),
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
            .map(|data| s.spawn(|| GGuf::scan(data)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect::<Result<Vec<_>, _>>()
    })
}
