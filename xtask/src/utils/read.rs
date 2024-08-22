use super::{Content, DataPromise, MetaValue, Tensor};
use ggus::{GGuf, GGufError, GGufFileName, GENERAL_ALIGNMENT};

impl<'a> Content<'a> {
    pub fn new(
        name: GGufFileName<'a>,
        files: impl IntoIterator<Item = &'a [u8]> + 'a,
    ) -> Result<Self, GGufError> {
        std::thread::scope(|s| {
            let mut ans = Self {
                name,
                alignment: 0,
                meta_kvs: Default::default(),
                tensors: Default::default(),
            };

            for thread in files
                .into_iter()
                .map(|data| s.spawn(|| GGuf::new(data)))
                .collect::<Vec<_>>()
                .into_iter()
            {
                thread
                    .join()
                    .unwrap()
                    .and_then(|gguf| ans.merge_file(gguf))?;
            }

            Ok(ans)
        })
    }

    fn merge_file(&mut self, others: GGuf<'a>) -> Result<(), GGufError> {
        self.alignment = self.alignment.max(others.alignment);

        for (k, kv) in others.meta_kvs {
            if k == GENERAL_ALIGNMENT || k.starts_with("split.") {
                continue;
            }
            let value = MetaValue {
                ty: kv.ty(),
                value: kv.value_bytes().into(),
            };
            if self.meta_kvs.insert(k.into(), value).is_some() {
                return Err(GGufError::DuplicateMetaKey(k.into()));
            }
        }

        for (name, tensor) in others.tensors {
            let tensor = tensor.to_info();
            let tensor = Tensor {
                ty: tensor.ty(),
                shape: tensor.shape().to_vec(),
                data: DataPromise::Borrowed(&others.data[tensor.offset()..][..tensor.nbytes()]),
            };
            if self.tensors.insert(name.into(), tensor).is_some() {
                return Err(GGufError::DuplicateTensorName(name.into()));
            }
        }

        Ok(())
    }
}
